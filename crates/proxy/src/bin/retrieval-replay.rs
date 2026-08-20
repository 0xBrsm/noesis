//! noesis-retrieval-replay — build GBT training data from conversation history.
//!
//! Walks user turns oldest-first, runs the live retrieval pipeline for each,
//! and logs every candidate to the `retrievals` table with a weak relevance
//! label: cosine similarity between the candidate chunk's stored embedding and
//! the assistant response that actually followed the user turn. Resumable —
//! turns that already have retrieval rows are skipped.
//!
//! Temporal note: journal chunks dated after the query are masked out by
//! default. Topic files are not versioned, so they leak future knowledge into
//! past queries; acceptable for score/rank features, not for content features.

use anyhow::Result;
use chrono::{DateTime, NaiveDate};
use clap::Parser;
use noesis_memory::{
    Config as MemConfig, Embedder, LLM, Memory, Reranker, RetrievalEvent, SearchRow,
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

/// Cap embedding input so oversized assistant responses don't blow API limits.
const MAX_EMBED_CHARS: usize = 8_000;

#[derive(Parser, Debug)]
#[command(
    name = "noesis-retrieval-replay",
    about = "Replay conversation history through retrieval, logging labeled candidates"
)]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Max user turns to replay this run. 0 = all remaining.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Candidates to log per query (more than production injects, so the
    /// training data includes negatives).
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// Drop journal chunks dated after the query (avoids temporal leakage).
    #[arg(long, default_value_t = true)]
    journal_mask: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = MemConfig::load_or_default(cli.config)?;
    let memory = Memory::open_from_config(&cfg)?;
    let embedder = Embedder::from_config(&cfg)?;
    let reranker = match Reranker::new(&cfg.data_dir.join("models"), &cfg.rerank_model) {
        Ok(r) => Some(r),
        Err(e) => {
            tracing::warn!("reranker disabled: {e}");
            None
        }
    };

    let pairs = memory.unreplayed_qa_pairs(cli.limit)?;
    tracing::info!(
        pairs = pairs.len(),
        top_k = cli.top_k,
        journal_mask = cli.journal_mask,
        "replay: starting"
    );

    let mut replayed = 0usize;
    let mut skipped_empty = 0usize;
    let mut rows_logged = 0usize;
    let mut rows_labeled = 0usize;

    let total = pairs.len();
    for (idx, pair) in pairs.iter().enumerate() {
        let query = pair.question.trim();
        if query.is_empty() || pair.answer.trim().is_empty() {
            skipped_empty += 1;
            continue;
        }

        let query_vec = match embedder.embed(&clip(query)).await {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("replay: query embed failed: {e}");
                None
            }
        };

        let mut rows = memory.search_for_context_with_vec(
            query,
            query_vec.as_deref(),
            reranker.as_ref(),
            cfg.retrieval_candidates.max(cli.top_k),
            cli.top_k,
            0.0,
        )?;

        if cli.journal_mask {
            let query_date = DateTime::from_timestamp(pair.ts as i64, 0)
                .map(|dt| dt.date_naive())
                .unwrap_or(NaiveDate::MAX);
            rows.retain(|r| journal_date(&r.path).is_none_or(|d| d <= query_date));
        }
        if rows.is_empty() {
            skipped_empty += 1;
            continue;
        }

        let answer_vec = match embedder.embed(&clip(&pair.answer)).await {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("replay: answer embed failed: {e}");
                None
            }
        };

        let labels: Vec<Option<f32>> = rows
            .iter()
            .map(|r| label_for(&memory, r, answer_vec.as_deref()))
            .collect();

        let events: Vec<RetrievalEvent<'_>> = rows
            .iter()
            .zip(&labels)
            .enumerate()
            .map(|(rank, (row, label))| RetrievalEvent {
                row,
                rank,
                injected: rank < cfg.retrieval_limit,
                label: *label,
            })
            .collect();

        memory.log_retrievals(&pair.session_id, pair.turn_index, query, &events)?;
        replayed += 1;
        rows_logged += events.len();
        rows_labeled += labels.iter().flatten().count();

        if (idx + 1) % 50 == 0 {
            tracing::info!(
                progress = format!("{}/{}", idx + 1, total),
                replayed,
                rows_logged,
                "replay: progress"
            );
        }
    }

    tracing::info!(
        replayed,
        skipped_empty,
        rows_logged,
        rows_labeled,
        "replay: done"
    );
    Ok(())
}

fn clip(s: &str) -> String {
    if s.len() <= MAX_EMBED_CHARS {
        return s.to_string();
    }
    let mut end = MAX_EMBED_CHARS;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

/// `journal/2023-04-10.md` → that date; None for non-journal paths.
fn journal_date(path: &str) -> Option<NaiveDate> {
    let stem = path.strip_prefix("journal/")?.strip_suffix(".md")?;
    NaiveDate::parse_from_str(stem, "%Y-%m-%d").ok()
}

fn label_for(memory: &Memory, row: &SearchRow, answer_vec: Option<&[f32]>) -> Option<f32> {
    let answer_vec = answer_vec?;
    let chunk_vec = memory.chunk_embedding(&row.id).ok().flatten()?;
    cosine(answer_vec, &chunk_vec)
}

fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na * nb))
}
