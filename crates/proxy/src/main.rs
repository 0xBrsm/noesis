//! noesis-proxy — OpenAI Responses API proxy with RAG memory injection.
//!
//! Forwards `/v1/responses` to an upstream Responses API endpoint, streaming
//! the response back unchanged while teeing the assembled assistant turn into
//! the canonical conversations store. Before forwarding, retrieves relevant
//! memory chunks (embedding shortlist → optional reranker) and injects them
//! as a developer-role input item.

use anyhow::{Context, Result};
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
    AutoDream, Config as MemConfig, Embedder, Journaler, LLM, LocalLLM, Memory, Message,
    RemoteLLM, Reranker, Role, SUMMARIZER_PROMPT, SearchRow, apply_plan, run_consolidation,
};
use reqwest::Client;
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "noesis-proxy", about = "OpenAI Responses API proxy with RAG memory injection")]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    /// All other settings live in the config file.
    #[arg(long)]
    config: Option<PathBuf>,
}

struct AppState {
    upstream: String,
    http: Client,
    memory: Arc<Mutex<Memory>>,
    embedder: Arc<Embedder>,
    chat: Arc<RemoteLLM>,
    reranker: Option<Arc<Reranker>>,
    journaler: Arc<Journaler>,
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

    let config_path = cli
        .config
        .or_else(MemConfig::default_path)
        .context("could not resolve config path: $HOME unset and --config not given")?;
    let cfg = MemConfig::load(&config_path)?;

    let model_name = if cfg.local_embed {
        cfg.local_embed_model.as_str()
    } else {
        cfg.embed_model.as_str()
    };
    let mut memory = Memory::open(
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

    let chat = Arc::new(RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.chat_model,
        &cfg.embed_model,
    ));
    let journaler = Arc::new(Journaler::new(&cfg.data_dir));

    match memory.index(&embedder).await {
        Ok(r) => tracing::info!(
            indexed = r.indexed,
            skipped = r.skipped,
            deleted = r.deleted,
            "startup index complete"
        ),
        Err(e) => tracing::warn!("startup index failed: {e}"),
    }

    let bind = cfg.bind;
    let upstream = cfg.upstream.clone();

    let state = Arc::new(AppState {
        upstream: upstream.clone(),
        http: Client::new(),
        memory: Arc::new(Mutex::new(memory)),
        embedder: Arc::new(embedder),
        chat,
        reranker,
        journaler,
        cfg,
        session_id: Uuid::new_v4().to_string(),
        next_turn: AtomicUsize::new(0),
    });

    spawn_journal_loop(state.clone());

    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .with_state(state);

    tracing::info!(%bind, %upstream, config = %config_path.display(), "noesis-proxy starting");
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Background journal summarizer ───────────────────────────────────────────

fn spawn_journal_loop(state: Arc<AppState>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(300));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Skip the initial immediate tick so we don't hammer at startup.
        tick.tick().await;
        loop {
            tick.tick().await;
            run_journal_tick(state.clone()).await;
        }
    });
}

async fn run_journal_tick(state: Arc<AppState>) {
    let st = state.journaler.load_state().await;
    let last_ts = state.journaler.last_ts(&st);

    let turns = {
        let mem = state.memory.lock().await;
        match mem.load_turns_since(last_ts) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("journal: load turns: {e}");
                return;
            }
        }
    };

    if !state.journaler.should_run(&st, turns.len()) {
        return;
    }

    let excerpt = turns
        .iter()
        .map(|(role, content, _ts)| format!("[{role}]: {content}"))
        .collect::<Vec<_>>()
        .join("\n\n");

    let messages = vec![
        Message {
            role: Role::System,
            content: SUMMARIZER_PROMPT.to_string(),
        },
        Message {
            role: Role::User,
            content: excerpt,
        },
    ];

    let summary = match state.chat.respond(&messages, None, false).await {
        Ok((text, _id)) => text,
        Err(e) => {
            tracing::warn!("journal: LLM call failed: {e}");
            return;
        }
    };

    let trimmed = summary.trim();
    if trimmed.is_empty() || trimmed == "SKIP" {
        if let Err(e) = state.journaler.mark_done().await {
            tracing::warn!("journal: mark_done after skip: {e}");
        }
        return;
    }

    let path = match state.journaler.append_entry(&summary).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("journal: append: {e}");
            return;
        }
    };

    {
        let mut mem = state.memory.lock().await;
        if let Err(e) = mem.index(state.embedder.as_ref()).await {
            tracing::warn!("journal: re-index: {e}");
        }
    }

    if let Err(e) = state.journaler.mark_done().await {
        tracing::warn!("journal: mark_done: {e}");
    }

    tracing::info!(
        path = %path.display(),
        turns = turns.len(),
        "journal entry appended"
    );

    // Try a dream consolidation; gates inside ensure it only runs when due.
    tokio::spawn(run_dream_if_due(state));
}

// ── Background dream consolidation ──────────────────────────────────────────

async fn run_dream_if_due(state: Arc<AppState>) {
    let auto_dream = AutoDream::new(&state.cfg.data_dir);
    let dream_state = auto_dream.load_state().await;
    let last_ts = dream_state.last_as_ts();

    let sessions = {
        let mem = state.memory.lock().await;
        match mem.sessions_since(last_ts) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!("dream: sessions_since: {e}");
                return;
            }
        }
    };

    match auto_dream.should_run(sessions).await {
        Ok(true) => {}
        Ok(false) => return,
        Err(e) => {
            tracing::warn!("dream: gate check: {e}");
            return;
        }
    }

    tracing::info!("dream: starting consolidation");

    let plan = match run_consolidation(&state.cfg.data_dir, state.chat.as_ref(), last_ts).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("dream: consolidation failed: {e}");
            let _ = auto_dream.finish().await;
            return;
        }
    };

    let updates = plan.updates.len();
    let deletes = plan.deletes.len();

    let changed = match apply_plan(&state.cfg.data_dir, &plan).await {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("dream: apply plan: {e}");
            let _ = auto_dream.finish().await;
            return;
        }
    };

    if changed > 0 {
        let mut mem = state.memory.lock().await;
        if let Err(e) = mem.index(state.embedder.as_ref()).await {
            tracing::warn!("dream: re-index: {e}");
        }
    }

    if let Err(e) = auto_dream.finish().await {
        tracing::warn!("dream: finish: {e}");
    }

    tracing::info!(updates, deletes, changed, "dream consolidation done");
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
        if matches!(
            n,
            "host" | "content-length" | "connection" | "accept-encoding" | "authorization"
        ) {
            continue;
        }
        req_builder = req_builder.header(name, value);
    }
    req_builder = req_builder.bearer_auth(&state.cfg.api_key);

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
