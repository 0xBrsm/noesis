mod config;
mod db;
mod dream;
mod llm;
mod memory;
mod tui;

use anyhow::Result;
use clap::Parser;

use crate::config::Command;
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
        Command::Chat => {
            let mut mem = Memory::open(
                &cfg.workspace,
                &cfg.embed_model,
                cfg.decay_half_life_days,
                cfg.semantic_weight,
                cfg.lexical_weight,
            )?;
            tui::run(&cfg, &embedder, reranker.as_ref(), &chat_llm, &mut mem).await
        }
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
            i + 1, r.score, score_detail, r.path, r.start_line, r.end_line,
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
