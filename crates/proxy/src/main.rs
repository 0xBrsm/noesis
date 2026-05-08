//! noesis-proxy — OpenAI Responses API proxy with RAG memory injection.
//!
//! Forwards `/v1/responses` to an upstream Responses API endpoint, streaming
//! the response back unchanged while teeing the assembled assistant turn into
//! the canonical conversations store. Before forwarding, retrieves relevant
//! memory chunks (embedding shortlist → optional reranker) and injects them
//! as a developer-role input item.

use anyhow::Result;
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::post,
};
use bytes::Bytes;
use clap::Parser;
use futures::StreamExt;
use noesis_memory::{
    Config as MemConfig, Embedder, LocalLLM, Memory, RemoteLLM, Reranker, SearchRow, LLM,
};
use reqwest::Client;
use serde_json::{Value, json};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "noesis-proxy", about = "OpenAI Responses API proxy with RAG memory injection")]
struct Cli {
    /// Address to bind the proxy on.
    #[arg(long, default_value = "127.0.0.1:8787")]
    bind: SocketAddr,

    /// Upstream Responses API base URL (e.g. https://api.openai.com).
    #[arg(long, env = "NOESIS_UPSTREAM", default_value = "https://api.openai.com")]
    upstream: String,

    /// Workspace dir for journal/topic files (defaults to ~/noesis).
    #[arg(long, env = "NOESIS_WORKSPACE")]
    workspace: Option<PathBuf>,

    /// Data dir for db + cached models (defaults to ~/.noesis).
    #[arg(long, env = "NOESIS_DATA_DIR")]
    data_dir: Option<PathBuf>,

    /// Use local fastembed embedder instead of remote API.
    #[arg(long, env = "NOESIS_LOCAL_EMBED")]
    local_embed: bool,
}

struct AppState {
    upstream: String,
    http: Client,
    memory: Arc<Mutex<Memory>>,
    embedder: Arc<Embedder>,
    reranker: Option<Arc<Reranker>>,
    cfg: MemConfig,
    session_id: String,
    next_turn: AtomicUsize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();

    let mut cfg = MemConfig::default();
    if let Some(ws) = cli.workspace.clone() {
        cfg.workspace = ws;
    }
    if let Some(dd) = cli.data_dir.clone() {
        cfg.data_dir = dd;
    }
    if cli.local_embed {
        cfg.local_embed = true;
    }

    let model_name = if cfg.local_embed {
        cfg.local_embed_model.as_str()
    } else {
        cfg.embed_model.as_str()
    };
    let memory = Memory::open(
        &cfg.workspace,
        &cfg.data_dir,
        model_name,
        cfg.decay_half_life_days,
        cfg.semantic_weight,
        cfg.lexical_weight,
    )?;

    let embedder = if cfg.local_embed {
        Embedder::Local(LocalLLM::new(
            &cfg.data_dir.join("models"),
            &cfg.local_embed_model,
        )?)
    } else {
        Embedder::Remote(RemoteLLM::new(
            &cfg.base_url,
            &cfg.api_key,
            &cfg.chat_model,
            &cfg.embed_model,
        ))
    };

    let reranker = match Reranker::new(&cfg.data_dir.join("models"), &cfg.rerank_model) {
        Ok(r) => Some(Arc::new(r)),
        Err(e) => {
            tracing::warn!("reranker disabled: {e}");
            None
        }
    };

    let state = Arc::new(AppState {
        upstream: cli.upstream.clone(),
        http: Client::new(),
        memory: Arc::new(Mutex::new(memory)),
        embedder: Arc::new(embedder),
        reranker,
        cfg,
        session_id: Uuid::new_v4().to_string(),
        next_turn: AtomicUsize::new(0),
    });

    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .with_state(state);

    tracing::info!(bind = %cli.bind, upstream = %cli.upstream, "noesis-proxy starting");
    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_responses(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, StatusCode> {
    let mut req: Value = serde_json::from_slice(&body).map_err(|_| StatusCode::BAD_REQUEST)?;
    let is_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let user_input = extract_input_text(&req);
    let prev_id = req
        .get("previous_response_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // Retrieval + injection (best-effort; failures don't block the request).
    let body = if user_input.trim().is_empty() {
        body
    } else {
        let rows = retrieve_context(&state, &user_input).await;
        if rows.is_empty() {
            body
        } else {
            inject_memory(&mut req, &format_context_block(&rows));
            match serde_json::to_vec(&req) {
                Ok(v) => Bytes::from(v),
                Err(e) => {
                    tracing::warn!("reserialize request: {e}");
                    body
                }
            }
        }
    };

    let url = format!("{}/v1/responses", state.upstream.trim_end_matches('/'));
    let mut req_builder = state.http.post(&url);
    for (name, value) in headers.iter() {
        let n = name.as_str();
        if matches!(n, "host" | "content-length" | "connection" | "accept-encoding") {
            continue;
        }
        req_builder = req_builder.header(name, value);
    }

    let upstream_resp = req_builder.body(body).send().await.map_err(|e| {
        tracing::error!("upstream error: {e}");
        StatusCode::BAD_GATEWAY
    })?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();

    if is_stream {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
        let memory = state.memory.clone();
        let session_id = state.session_id.clone();
        let user_idx = state.next_turn.fetch_add(2, Ordering::SeqCst);

        tokio::spawn(async move {
            let mut stream = upstream_resp.bytes_stream();
            let mut buf = Vec::<u8>::new();
            let mut assistant_text = String::new();
            let mut response_id: Option<String> = None;

            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(bytes) => {
                        buf.extend_from_slice(&bytes);
                        while let Some(idx) = find_event_boundary(&buf) {
                            let event: Vec<u8> = buf.drain(..idx + 2).collect();
                            parse_sse_event(&event[..idx], &mut assistant_text, &mut response_id);
                        }
                        if tx.send(Ok(bytes)).await.is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let io_err = std::io::Error::other(e);
                        let _ = tx.send(Err(io_err)).await;
                        return;
                    }
                }
            }
            if !buf.is_empty() {
                parse_sse_event(&buf, &mut assistant_text, &mut response_id);
            }

            persist_turns(
                &memory,
                &session_id,
                user_idx,
                &user_input,
                prev_id.as_deref(),
                &assistant_text,
                response_id.as_deref(),
            )
            .await;
        });

        let stream = ReceiverStream::new(rx);
        let body = Body::from_stream(stream);
        Ok(build_response(status, &resp_headers, body))
    } else {
        let resp_body = upstream_resp
            .bytes()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        if let Ok(resp_json) = serde_json::from_slice::<Value>(&resp_body) {
            let resp_id = resp_json
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            let assistant_text = extract_assistant_text(&resp_json);
            let user_idx = state.next_turn.fetch_add(2, Ordering::SeqCst);
            persist_turns(
                &state.memory,
                &state.session_id,
                user_idx,
                &user_input,
                prev_id.as_deref(),
                &assistant_text,
                resp_id.as_deref(),
            )
            .await;
        }

        Ok(build_response(status, &resp_headers, Body::from(resp_body)))
    }
}

fn build_response(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: Body,
) -> Response {
    let mut resp = Response::new(body);
    *resp.status_mut() = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK);
    for (k, v) in headers.iter() {
        if matches!(k.as_str(), "content-length" | "transfer-encoding") {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            axum::http::HeaderName::from_bytes(k.as_str().as_bytes()),
            axum::http::HeaderValue::from_bytes(v.as_bytes()),
        ) {
            resp.headers_mut().insert(name, value);
        }
    }
    resp
}

async fn persist_turns(
    memory: &Mutex<Memory>,
    session_id: &str,
    user_idx: usize,
    user_input: &str,
    prev_id: Option<&str>,
    assistant_text: &str,
    response_id: Option<&str>,
) {
    let mem = memory.lock().await;
    if let Err(e) = mem.insert_turn(session_id, user_idx, "user", user_input, prev_id) {
        tracing::warn!("insert user turn: {e}");
    }
    if let Err(e) = mem.insert_turn(
        session_id,
        user_idx + 1,
        "assistant",
        assistant_text,
        response_id,
    ) {
        tracing::warn!("insert assistant turn: {e}");
    }
}

// ── Retrieval + injection ───────────────────────────────────────────────────

async fn retrieve_context(state: &AppState, query: &str) -> Vec<SearchRow> {
    // Step 1: figure out whether vector search is available, then drop the lock
    // before awaiting the embedding (rusqlite Connection is !Sync, so we must
    // not hold a Memory reference across an await point).
    let vec_available = {
        let mem = state.memory.lock().await;
        mem.vec_available()
    };

    let query_vec = if vec_available {
        match state.embedder.embed(query).await {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("embed failed: {e}");
                None
            }
        }
    } else {
        None
    };

    let mem = state.memory.lock().await;
    match mem.search_for_context_with_vec(
        query,
        query_vec.as_deref(),
        state.reranker.as_deref(),
        state.cfg.retrieval_candidates,
        state.cfg.retrieval_limit,
        state.cfg.retrieval_threshold,
    ) {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!("retrieval failed: {e}");
            Vec::new()
        }
    }
}

fn format_context_block(rows: &[SearchRow]) -> String {
    let mut s = String::from(
        "<retrieved_memory>\nThe following notes from the user's persistent memory may be relevant. \
         Use them when they apply; ignore them when they don't.\n\n",
    );
    for row in rows {
        s.push_str(&format!(
            "[{} L{}-{}]\n{}\n\n",
            row.path, row.start_line, row.end_line, row.text
        ));
    }
    s.push_str("</retrieved_memory>");
    s
}

/// Inject memory as a developer-role input item, preceding any existing input.
/// Handles both string and array `input` shapes.
fn inject_memory(req: &mut Value, context: &str) {
    let memory_item = json!({ "role": "developer", "content": context });
    match req.get_mut("input") {
        Some(input @ Value::String(_)) => {
            let user_text = input.as_str().unwrap_or("").to_string();
            *input = Value::Array(vec![
                memory_item,
                json!({ "role": "user", "content": user_text }),
            ]);
        }
        Some(Value::Array(arr)) => {
            arr.insert(0, memory_item);
        }
        _ => {
            req["input"] = Value::Array(vec![memory_item]);
        }
    }
}

// ── Request/response parsing ────────────────────────────────────────────────

fn extract_input_text(req: &Value) -> String {
    match req.get("input") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => {
            for item in items.iter().rev() {
                if item.get("role").and_then(|v| v.as_str()) == Some("user")
                    && let Some(content) = item.get("content")
                {
                    return extract_text_from_content(content);
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

fn extract_text_from_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                p.get("text")
                    .and_then(|t| t.as_str())
                    .map(str::to_string)
                    .or_else(|| p.as_str().map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn extract_assistant_text(resp: &Value) -> String {
    let mut out = String::new();
    let Some(output) = resp.get("output").and_then(|v| v.as_array()) else {
        return out;
    };
    for item in output {
        if item.get("type").and_then(|v| v.as_str()) != Some("message") {
            continue;
        }
        let Some(content) = item.get("content").and_then(|v| v.as_array()) else {
            continue;
        };
        for c in content {
            if c.get("type").and_then(|v| v.as_str()) == Some("output_text")
                && let Some(t) = c.get("text").and_then(|v| v.as_str())
            {
                out.push_str(t);
            }
        }
    }
    out
}

// ── SSE parsing ─────────────────────────────────────────────────────────────

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

fn parse_sse_event(event: &[u8], assistant_text: &mut String, response_id: &mut Option<String>) {
    for line in event.split(|&b| b == b'\n') {
        let Ok(line) = std::str::from_utf8(line) else { continue };
        let line = line.trim_end_matches('\r');
        let Some(data) = line.strip_prefix("data:") else { continue };
        let data = data.trim_start();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        match v.get("type").and_then(|t| t.as_str()) {
            Some("response.output_text.delta") => {
                if let Some(d) = v.get("delta").and_then(|d| d.as_str()) {
                    assistant_text.push_str(d);
                }
            }
            Some("response.completed") => {
                if let Some(id) = v.pointer("/response/id").and_then(|x| x.as_str()) {
                    *response_id = Some(id.to_string());
                }
            }
            _ => {}
        }
    }
}
