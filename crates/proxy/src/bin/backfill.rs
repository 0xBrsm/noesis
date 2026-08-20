//! noesis-backfill — populate dated journal files from existing conversation
//! history. Walks the `conversations` table, buckets turns by UTC date, runs
//! the journal prompt (`data_dir/prompts/journal.md`, with compiled default
//! fallback) per date, writes `journal/YYYY-MM-DD.md`.
//!
//! Existing journal files are not overwritten.

use anyhow::Result;
use chrono::NaiveDate;
use clap::Parser;
use noesis_memory::{
    BackfillOptions, Config as MemConfig, Embedder, Journaler, Memory, RemoteLLM,
    backfill_by_date, load_journal_prompt,
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "noesis-backfill", about = "Backfill dated journal entries from stored turns")]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Inclusive UTC start date (YYYY-MM-DD). Defaults to earliest turn.
    #[arg(long)]
    from: Option<NaiveDate>,

    /// Inclusive UTC end date (YYYY-MM-DD). Defaults to latest turn.
    #[arg(long)]
    to: Option<NaiveDate>,

    /// Re-index memory after backfill so new journal files become searchable.
    #[arg(long, default_value_t = true)]
    reindex: bool,

    /// Stop the run once cumulative input chars (across all chunks) exceeds
    /// this number. Used to stay under a daily token quota — rough conversion
    /// is ~3.5 chars/token English. Resume by re-running; written entries skip.
    #[arg(long)]
    daily_input_char_budget: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = MemConfig::load_or_default(cli.config)?;
    let mut memory = Memory::open_from_config(&cfg)?;

    let summarizer = RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.summarizer_model,
        &cfg.embed_model,
    );
    let journaler = Journaler::new(&cfg.data_dir);

    let turns = memory.load_turns_since(0.0)?;
    tracing::info!(
        turns = turns.len(),
        data_dir = %cfg.data_dir.display(),
        summarizer = %cfg.summarizer_model,
        "backfill: starting"
    );

    let date_filter = match (cli.from, cli.to) {
        (Some(a), Some(b)) => Some((a, b)),
        (Some(a), None) => Some((a, NaiveDate::MAX)),
        (None, Some(b)) => Some((NaiveDate::MIN, b)),
        (None, None) => None,
    };

    let journal_prompt = load_journal_prompt(&cfg.data_dir);
    let opts = BackfillOptions {
        date_filter,
        max_chunk_chars: cfg.summarizer_max_chunk_chars,
        daily_input_char_budget: cli.daily_input_char_budget,
    };
    let stats = backfill_by_date(&journaler, &summarizer, &journal_prompt, &turns, &opts).await?;
    tracing::info!(?stats, "backfill: done");

    if cli.reindex {
        let embedder = Embedder::from_config(&cfg)?;
        let r = memory.index(&embedder).await?;
        tracing::info!(
            indexed = r.indexed,
            skipped = r.skipped,
            deleted = r.deleted,
            "backfill: reindex complete"
        );
    }

    Ok(())
}
