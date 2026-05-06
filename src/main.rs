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

    match &cli.command {
        Command::Index { force } => cmd_index(&cfg, &embedder, *force).await,
        Command::Search { query, limit, keyword_only, full } => {
            cmd_search(&cfg, &embedder, query, *limit, *keyword_only, reranker.as_ref(), *full).await
        }
        Command::Chat => cmd_chat(&cfg, &embedder, reranker.as_ref()).await,
    }
}

async fn cmd_index<L: LLM>(cfg: &config::Config, llm: &L, force: bool) -> Result<()> {
    let mut mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.chunk_tokens,
        cfg.chunk_overlap,
    )?;

    if force {
        let db_path = cfg.workspace.join("memory.db");
        if db_path.exists() {
            std::fs::remove_file(&db_path)?;
            mem = Memory::open(
                &cfg.workspace,
                &cfg.embed_model,
                cfg.chunk_tokens,
                cfg.chunk_overlap,
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
    keyword_only: bool,
    reranker: Option<&Reranker>,
    full: bool,
) -> Result<()> {
    let mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.chunk_tokens,
        cfg.chunk_overlap,
    )?;

    let results = if keyword_only {
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
            (Some(v), Some(t)) => format!("vec={v:.3} fts={t:.3}"),
            (Some(v), None)    => format!("vec={v:.3}"),
            (None,    Some(t)) => format!("fts={t:.3}"),
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

async fn cmd_chat<L: LLM>(
    cfg: &config::Config,
    llm: &L,
    reranker: Option<&Reranker>,
) -> Result<()> {
    use crate::llm::{Message, Role};
    use rustyline::error::ReadlineError;

    let mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.chunk_tokens,
        cfg.chunk_overlap,
    )?;

    let mut history: Vec<Message> = vec![];
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

        let chunks = mem.search_for_context(
            &input,
            llm,
            reranker,
            cfg.context_candidates,
            cfg.context_limit,
            cfg.context_threshold,
        ).await?;

        let system = build_system_prompt(&chunks);
        let mut messages = vec![Message { role: Role::System, content: system }];
        messages.extend_from_slice(&history);
        messages.push(Message { role: Role::User, content: input.clone() });

        let response = llm.chat_stream(&messages).await?;

        history.push(Message { role: Role::User, content: input });
        history.push(Message { role: Role::Assistant, content: response });

        trim_history(&mut history, cfg.history_chars);
    }

    Ok(())
}

fn build_system_prompt(chunks: &[SearchRow]) -> String {
    let mut s = String::from(
        "You are a helpful assistant with access to the user's memory store.\n"
    );
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

fn trim_history(history: &mut Vec<crate::llm::Message>, char_budget: usize) {
    while history.iter().map(|m| m.content.len()).sum::<usize>() > char_budget
        && history.len() >= 2
    {
        history.drain(0..2);
    }
}
