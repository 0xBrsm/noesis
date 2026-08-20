//! noesis-dream-backfill — replay dream consolidation across already-imported
//! journal entries, one UTC day at a time. Mirrors how the live gate would
//! have fired for an active user (≥1 dream/day once user-message volume crosses
//! the threshold), letting topic files accrete in date order.
//!
//! Stateful: persists `last_consolidated_at` in `.consolidation_state.json`,
//! so re-running picks up where it left off. Days whose journal file is
//! missing or empty are skipped. Days whose end-of-day timestamp is ≤ the
//! persisted watermark are skipped.

use anyhow::Result;
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use clap::Parser;
use noesis_memory::{
    Config as MemConfig, Embedder, Memory, RemoteLLM, apply_plan, load_topic_prompt,
    run_dream_with_text,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "noesis-dream-backfill", about = "Replay dream consolidation per UTC day over imported journals")]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Inclusive UTC start date (YYYY-MM-DD). Defaults to earliest journal file.
    #[arg(long)]
    from: Option<NaiveDate>,

    /// Inclusive UTC end date (YYYY-MM-DD). Defaults to latest journal file.
    #[arg(long)]
    to: Option<NaiveDate>,

    /// Don't apply the plan or advance state — just log what would happen.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Stop once cumulative input chars across phases exceed this budget.
    /// Tune to fit gpt-5-mini's daily token quota (~5.7 chars/token observed).
    #[arg(long)]
    daily_input_char_budget: Option<usize>,

    /// Re-index after backfill so topic file changes become searchable.
    #[arg(long, default_value_t = true)]
    reindex: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ConsolidationStateFile {
    last_consolidated_at: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = MemConfig::load_or_default(cli.config)?;

    let summarizer = RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.summarizer_model,
        &cfg.embed_model,
    );
    let topic_prompt = load_topic_prompt(&cfg.data_dir);

    let state_path = cfg.data_dir.join(".consolidation_state.json");
    let mut state = load_state(&state_path).await;
    let initial_watermark = state.last_consolidated_at.unwrap_or(0);

    let journal_dir = cfg.data_dir.join("journal");
    let mut dates = collect_journal_dates(&journal_dir).await?;
    dates.sort();

    let dates: Vec<NaiveDate> = dates
        .into_iter()
        .filter(|d| match (cli.from, cli.to) {
            (Some(a), Some(b)) => *d >= a && *d <= b,
            (Some(a), None) => *d >= a,
            (None, Some(b)) => *d <= b,
            (None, None) => true,
        })
        .filter(|d| end_of_day_ts(*d) > initial_watermark)
        .collect();

    tracing::info!(
        candidates = dates.len(),
        data_dir = %cfg.data_dir.display(),
        summarizer = %cfg.summarizer_model,
        dry_run = cli.dry_run,
        "dream-backfill: starting"
    );

    let mut processed = 0usize;
    let mut empty_skipped = 0usize;
    let mut failed = 0usize;
    let mut topic_changes = 0usize;
    let mut input_chars_used = 0usize;
    let mut stopped_on_budget = false;

    let total = dates.len();
    for (idx, date) in dates.iter().enumerate() {
        let journal_path = journal_dir.join(format!("{date}.md"));
        let body = match fs::read_to_string(&journal_path).await {
            Ok(s) => s,
            Err(_) => {
                empty_skipped += 1;
                continue;
            }
        };
        if body.trim().is_empty() {
            empty_skipped += 1;
            continue;
        }

        // Format identical to what live `read_journal_since` produces.
        let journal_text = format!("=== {}.md ===\n{}\n\n", date, body);

        // Conservative pre-flight cost: phases 1+2+3 each include the system
        // prompt and topic summaries; phase 2 also includes the journal text.
        // Estimating 3× system prompt is overshoot but keeps us under budget.
        let est_cost = journal_text.len() + 4_000; // ~4KB for prompts + topics list
        if let Some(budget) = cli.daily_input_char_budget
            && input_chars_used + est_cost > budget
        {
            tracing::warn!(
                used = input_chars_used,
                budget,
                date = %date,
                "dream-backfill: daily input budget reached, stopping cleanly"
            );
            stopped_on_budget = true;
            break;
        }

        tracing::info!(
            date = %date,
            progress = format!("{}/{}", idx + 1, total),
            journal_chars = journal_text.len(),
            "dream-backfill: consolidating"
        );

        let plan = match run_dream_with_text(
            &cfg.data_dir,
            &summarizer,
            &topic_prompt,
            &journal_text,
        )
        .await
        {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(date = %date, "dream-backfill: consolidation failed: {e}");
                failed += 1;
                continue;
            }
        };
        input_chars_used += est_cost;

        let updates = plan.updates.len();
        let deletes = plan.deletes.len();

        if cli.dry_run {
            tracing::info!(
                date = %date,
                updates,
                deletes,
                "dream-backfill: DRY RUN, plan not applied"
            );
        } else {
            match apply_plan(&cfg.data_dir, &plan).await {
                Ok(changed) => {
                    topic_changes += changed;
                    tracing::info!(
                        date = %date,
                        updates,
                        deletes,
                        changed,
                        "dream-backfill: applied plan"
                    );
                }
                Err(e) => {
                    tracing::warn!(date = %date, "dream-backfill: apply failed: {e}");
                    failed += 1;
                    continue;
                }
            }
            state.last_consolidated_at = Some(end_of_day_ts(*date));
            save_state(&state_path, &state).await?;
        }
        processed += 1;
    }

    tracing::info!(
        processed,
        empty_skipped,
        failed,
        topic_changes,
        input_chars_used,
        stopped_on_budget,
        "dream-backfill: done"
    );

    if !cli.dry_run && cli.reindex {
        let mut memory = Memory::open_from_config(&cfg)?;
        let embedder = Embedder::from_config(&cfg)?;
        let r = memory.index(&embedder).await?;
        tracing::info!(
            indexed = r.indexed,
            skipped = r.skipped,
            deleted = r.deleted,
            "dream-backfill: reindex complete"
        );
    }

    Ok(())
}

fn end_of_day_ts(date: NaiveDate) -> u64 {
    let end = NaiveDateTime::new(date, NaiveTime::from_hms_opt(23, 59, 59).unwrap());
    Utc.from_utc_datetime(&end).timestamp() as u64
}

async fn collect_journal_dates(journal_dir: &PathBuf) -> Result<Vec<NaiveDate>> {
    let mut dates = Vec::new();
    if !journal_dir.exists() {
        return Ok(dates);
    }
    let mut rd = fs::read_dir(journal_dir).await?;
    while let Some(e) = rd.next_entry().await? {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        if let Ok(d) = NaiveDate::parse_from_str(stem, "%Y-%m-%d") {
            dates.push(d);
        }
    }
    Ok(dates)
}

async fn load_state(path: &PathBuf) -> ConsolidationStateFile {
    match fs::read_to_string(path).await {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => ConsolidationStateFile::default(),
    }
}

async fn save_state(path: &PathBuf, state: &ConsolidationStateFile) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).await?;
    }
    fs::write(path, serde_json::to_string_pretty(state)?).await?;
    Ok(())
}
