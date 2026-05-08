mod config;
mod db;
mod llm;
mod memory;

use anyhow::Result;
use clap::Parser;

use crate::config::Command;
use crate::db::SearchRow;
use crate::llm::{Embedder, LLM, LocalLLM, RemoteLLM, Reranker};
use crate::memory::Memory;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = config::Cli::parse();
    let cfg = config::load(&cli)?;

    let embedder: Embedder = if cli.local_embed {
        println!("loading local embedder (all-MiniLM-L6-v2-Q)...");
        Embedder::Local(LocalLLM::new()?)
    } else {
        Embedder::Remote(RemoteLLM::new(
            &cfg.base_url,
            &cfg.api_key,
            &cfg.chat_model,
            &cfg.embed_model,
        ))
    };

    let reranker: Option<Reranker> = if cli.local_embed {
        println!("loading reranker (ms-marco-MiniLM-L-6-v2)...");
        Some(Reranker::new()?)
    } else {
        None
    };

    // Chat always uses a remote LLM regardless of embed mode
    let chat_llm = RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.chat_model,
        &cfg.embed_model,
    );

    match &cli.command {
        Command::Index { force } => cmd_index(&cfg, &embedder, *force).await,
        Command::Search { query, limit, lexical_only, full } => {
            cmd_search(&cfg, &embedder, query, *limit, *lexical_only, reranker.as_ref(), *full).await
        }
        Command::Chat => cmd_chat(&cfg, &embedder, reranker.as_ref(), &chat_llm).await,
    }
}

async fn cmd_index<L: LLM>(cfg: &config::Config, llm: &L, force: bool) -> Result<()> {
    let mut mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.decay_half_life_days,
        cfg.semantic_weight,
        cfg.lexical_weight,
    )?;

    if force {
        let db_path = cfg.workspace.join("memory.db");
        if db_path.exists() {
            std::fs::remove_file(&db_path)?;
            mem = Memory::open(
                &cfg.workspace,
                &cfg.embed_model,
                cfg.decay_half_life_days,
                cfg.semantic_weight,
                cfg.lexical_weight,
            )?;
        }
    }

    let r = mem.index(llm).await?;
    println!(
        "indexed: {}  skipped: {}  deleted: {}",
        r.indexed, r.skipped, r.deleted
    );
    Ok(())
}

async fn cmd_search<L: LLM>(
    cfg: &config::Config,
    llm: &L,
    query: &str,
    limit: usize,
    lexical_only: bool,
    reranker: Option<&Reranker>,
    full: bool,
) -> Result<()> {
    let mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.decay_half_life_days,
        cfg.semantic_weight,
        cfg.lexical_weight,
    )?;

    let results = if lexical_only {
        mem.search_keyword(query, limit)?
    } else {
        mem.search(query, limit, llm, reranker).await?
    };

    if results.is_empty() {
        println!("no results");
        return Ok(());
    }

    for (i, r) in results.iter().enumerate() {
        let score_detail = match (r.vector_score, r.text_score) {
            (Some(v), Some(t)) => format!("sem={v:.3} lex={t:.3}"),
            (Some(v), None)    => format!("sem={v:.3}"),
            (None,    Some(t)) => format!("lex={t:.3}"),
            (None,    None)    => String::new(),
        };

        println!(
            "[{}] score={:.3}  {}  {}:{}-{}",
            i + 1,
            r.score,
            score_detail,
            r.path,
            r.start_line,
            r.end_line,
        );

        let text = r.text.trim();
        let display = if full || text.len() <= 200 {
            text.to_string()
        } else {
            format!("{}…", &text[..200])
        };

        for line in display.lines() {
            println!("    {line}");
        }
        println!();
    }

    Ok(())
}

async fn cmd_chat<E: LLM>(
    cfg: &config::Config,
    embedder: &E,
    reranker: Option<&Reranker>,
    chat_llm: &RemoteLLM,
) -> Result<()> {
    use crate::llm::{Message, Role};
    use rustyline::error::ReadlineError;

    let mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.decay_half_life_days,
        cfg.semantic_weight,
        cfg.lexical_weight,
    )?;

    let context_md = crate::memory::load_context_md(&cfg.workspace);

    let session_id = uuid::Uuid::new_v4().to_string();

    // Load prior history and last response ID for stateful chaining
    let mut history: Vec<Message> = mem.load_recent_turns()?
        .into_iter()
        .map(|(role, content)| Message {
            role: match role.as_str() {
                "assistant" => Role::Assistant,
                "system"    => Role::System,
                _           => Role::User,
            },
            content,
        })
        .collect();

    let mut previous_response_id: Option<String> = mem.last_response_id()?;
    let mut turn_index = 0usize;
    let mut rl = rustyline::DefaultEditor::new()?;

    println!("Chat ready. Ctrl-D to exit.");

    loop {
        let line = match rl.readline("> ") {
            Ok(l) => l,
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => break,
            Err(e) => return Err(e.into()),
        };
        let input = line.trim().to_string();
        if input.is_empty() { continue; }
        let _ = rl.add_history_entry(&input);

        let t0 = std::time::Instant::now();

        let chunks = mem.search_for_context(
            &input,
            embedder,
            reranker,
            cfg.retrieval_candidates,
            cfg.retrieval_limit,
            cfg.retrieval_threshold,
        ).await?;

        if cfg.debug {
            eprintln!("[debug] retrieval: {:.2}s  chunks injected: {}", t0.elapsed().as_secs_f64(), chunks.len());
            for c in &chunks {
                eprintln!("[debug]   score={:.3}  {}:{}-{}", c.score, c.path, c.start_line, c.end_line);
            }
        }

        let system = build_system_prompt(&chunks, context_md.as_deref());
        let mut messages = vec![Message { role: Role::System, content: system }];
        messages.extend_from_slice(&history);
        messages.push(Message { role: Role::User, content: input.clone() });

        if cfg.debug {
            eprintln!("[debug] full prompt ({} messages), prev_id={:?}", messages.len(), previous_response_id);
            for m in &messages {
                eprintln!("[debug] [{:?}] {}", m.role, m.content.chars().take(300).collect::<String>());
            }
            eprintln!("[debug] sending to {} at {:.2}s", cfg.chat_model, t0.elapsed().as_secs_f64());
        }

        let (response, response_id) = chat_llm.respond(
            &messages,
            previous_response_id.as_deref(),
            cfg.debug,
        ).await?;

        if cfg.debug {
            eprintln!("[debug] turn total: {:.2}s", t0.elapsed().as_secs_f64());
        }

        // Persist both turns; store response_id on assistant turn for chaining
        mem.insert_turn(&session_id, turn_index,     "user",      &input,     None)?;
        mem.insert_turn(&session_id, turn_index + 1, "assistant", &response,  Some(&response_id))?;
        turn_index += 2;
        previous_response_id = Some(response_id);

        history.push(Message { role: Role::User,      content: input });
        history.push(Message { role: Role::Assistant, content: response });

    }

    Ok(())
}

fn build_system_prompt(chunks: &[SearchRow], context_md: Option<&str>) -> String {
    let mut s = match context_md {
        Some(ctx) if !ctx.trim().is_empty() => format!("{}\n", ctx.trim()),
        _ => String::from("You are a helpful assistant with access to the user's memory store.\n"),
    };
    if chunks.is_empty() {
        return s;
    }
    s.push_str("\nRelevant memory:\n");
    for chunk in chunks {
        s.push_str(&format!(
            "\n---\nSource: {} (lines {}-{})\n{}\n",
            chunk.path, chunk.start_line, chunk.end_line, chunk.text.trim()
        ));
    }
    s.push_str("---\n");
    s
}

