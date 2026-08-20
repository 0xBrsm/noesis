//! noesis-query — interactive REPL that shows exactly what the LLM sees.
//!
//! Loads the same Memory/Embedder/Reranker the proxy builds, reads queries
//! from stdin, and prints the verbatim `<retrieved_memory>` block that would
//! be injected as the developer-role input item.

use anyhow::Result;
use clap::Parser;
use noesis_memory::{Config as MemConfig, Embedder, LLM, Memory, Reranker, format_context_block};
use std::io::{BufRead, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "noesis-query", about = "Interactive retrieval REPL — prints the exact text the LLM sees")]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Also print the per-row score table (path, decayed, raw, vec, text) above
    /// the `<retrieved_memory>` block.
    #[arg(long, default_value_t = false)]
    debug: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = MemConfig::load_or_default(cli.config)?;
    let memory = Memory::open_from_config(&cfg)?;
    let embedder = Embedder::from_config(&cfg)?;

    let reranker = match Reranker::new(&cfg.data_dir.join("models"), &cfg.rerank_model) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("reranker disabled: {e}");
            None
        }
    };

    eprintln!("noesis-query — type a query and press Enter. Ctrl-D or 'exit' to quit.");
    eprintln!(
        "config: candidates={} limit={} threshold={} half_life={}d sem/lex={}/{}",
        cfg.retrieval_candidates,
        cfg.retrieval_limit,
        cfg.retrieval_threshold,
        cfg.decay_half_life_days,
        cfg.semantic_weight,
        cfg.lexical_weight,
    );

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        eprint!("\n> ");
        stdout.flush().ok();
        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            eprintln!();
            break;
        }
        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        if matches!(query, "exit" | "quit") {
            break;
        }

        let query_vec = if memory.vec_available() {
            match embedder.embed(query).await {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("embed failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let rows = memory.search_for_context_with_vec(
            query,
            query_vec.as_deref(),
            reranker.as_ref(),
            cfg.retrieval_candidates,
            cfg.retrieval_limit,
            cfg.retrieval_threshold,
        )?;

        if cli.debug {
            eprintln!(
                "── {} rows (after threshold {}) ──",
                rows.len(),
                cfg.retrieval_threshold
            );
            for r in &rows {
                eprintln!(
                    "  {:>5.3} (raw {:>5.3})  vec={:?} text={:?}  {} L{}-{}",
                    r.score, r.raw_score, r.vector_score, r.text_score,
                    r.path, r.start_line, r.end_line,
                );
            }
            eprintln!("──");
        }

        println!("{}", format_context_block(&rows));
    }

    Ok(())
}
