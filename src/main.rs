mod config;
mod db;
mod llm;
mod memory;

use anyhow::Result;
use clap::Parser;

use crate::config::Command;
use crate::llm::{Embedder, LLM, LocalLLM, RemoteLLM};
use crate::memory::Memory;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = config::Cli::parse();
    let cfg = config::load(&cli)?;

    let embedder: Embedder = if cli.local_embed {
        println!("loading local embedder (nomic-embed-text-v1.5)...");
        Embedder::Local(LocalLLM::new()?)
    } else {
        Embedder::Remote(RemoteLLM::new(
            &cfg.base_url,
            &cfg.api_key,
            &cfg.chat_model,
            &cfg.embed_model,
        ))
    };

    match &cli.command {
        Command::Index { force } => cmd_index(&cfg, &embedder, *force).await,
        Command::Search { query, limit, keyword_only, full } => {
            cmd_search(&cfg, &embedder, query, *limit, *keyword_only, *full).await
        }
        Command::Chat => {
            eprintln!("chat not yet implemented — use index and search for now");
            Ok(())
        }
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
        let db_path = cfg.workspace.join(".llmchat").join("memory.db");
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
        mem.search(query, limit, llm).await?
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
