//! Background journal summarizer. Compresses the recent transcript delta
//! into dated markdown entries that the existing chunker indexes for retrieval.
//!
//! The DB transcript is canonical. The journal is a denser, retrieval-friendly
//! projection of it; the dream consolidation later distills journal entries
//! into evergreen topic files.
//!
//! Prompt framing borrows from claurst's `EXTRACTION_SYSTEM_PROMPT`
//! (src-rust/crates/query/src/session_memory.rs:360). The output target differs:
//! claurst emits typed `MEMORY: <category> | <confidence> | <fact>` lines for
//! direct ingestion into a categorized memory store; we emit narrative markdown
//! that the header-based chunker re-splits into retrieval chunks.

use anyhow::Result;
use chrono::{DateTime, Local, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::llm::{LLM, Message, Role};

pub const JOURNAL_PROMPT: &str = "\
You are a memory extraction assistant. Your job is to identify key facts, \
preferences, patterns, and decisions from a recent conversation between a user \
and an AI assistant that would be useful to remember for future interactions. \
Be precise, concise, and only extract genuinely useful information. Do not \
extract trivial or transient details.

The output is a journal entry that feeds back into long-term retrieval, so \
favor concrete, searchable phrasing.

Capture only what's worth remembering across future sessions:
- Decisions made or preferences expressed
- Topics, files, systems, or projects the user is working on
- Information the user explicitly wants remembered
- Open questions or unresolved threads

Style: short bullets or terse paragraphs in markdown. Skip small talk, \
debugging back-and-forth, and transient implementation details. If nothing in \
the excerpt is worth recording, output exactly: SKIP

Output only the entry (or SKIP). No preamble, no explanation.";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JournalState {
    pub last_journaled_ts: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct JournalConfig {
    /// Minimum minutes between journal entries.
    pub min_minutes: f64,
    /// Minimum new transcript turns required before summarizing.
    pub min_turns: usize,
}

impl Default for JournalConfig {
    fn default() -> Self {
        Self {
            min_minutes: 30.0,
            min_turns: 4,
        }
    }
}

pub struct Journaler {
    pub config: JournalConfig,
    state_file: PathBuf,
    journal_dir: PathBuf,
}

impl Journaler {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            config: JournalConfig::default(),
            state_file: data_dir.join(".journal_state.json"),
            journal_dir: data_dir.join("journal"),
        }
    }

    pub async fn load_state(&self) -> JournalState {
        match fs::read_to_string(&self.state_file).await {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => JournalState::default(),
        }
    }

    /// Lower-bound timestamp (unix seconds, as f64) for "turns since last entry".
    pub fn last_ts(&self, state: &JournalState) -> f64 {
        state.last_journaled_ts.unwrap_or(0) as f64
    }

    /// Both gates: enough new turns AND enough minutes since last entry.
    pub fn should_run(&self, state: &JournalState, new_turn_count: usize) -> bool {
        if new_turn_count < self.config.min_turns {
            return false;
        }
        match state.last_journaled_ts {
            None => true,
            Some(last) => (now_secs().saturating_sub(last) as f64) / 60.0 >= self.config.min_minutes,
        }
    }

    /// Append a timestamped entry to today's journal file. Creates the file
    /// (with date H1) if missing. Returns the file path so the caller can
    /// re-index just the affected file.
    pub async fn append_entry(&self, summary: &str) -> Result<PathBuf> {
        fs::create_dir_all(&self.journal_dir).await?;
        let today = Local::now().format("%Y-%m-%d").to_string();
        let path = self.journal_dir.join(format!("{today}.md"));
        let stamp = Local::now().format("%H:%M").to_string();
        let entry = format!("\n## {stamp}\n\n{}\n", summary.trim());

        if path.exists() {
            let mut f = fs::OpenOptions::new().append(true).open(&path).await?;
            f.write_all(entry.as_bytes()).await?;
            f.flush().await?;
        } else {
            let header = format!("# {today}\n");
            fs::write(&path, format!("{header}{entry}")).await?;
        }
        Ok(path)
    }

    /// Write (or replace) the journal file for a specific date. Used by the
    /// by-date backfill importer — not the live append flow.
    pub async fn write_dated_entry(&self, date: NaiveDate, body: &str) -> Result<PathBuf> {
        fs::create_dir_all(&self.journal_dir).await?;
        let date_str = date.format("%Y-%m-%d").to_string();
        let path = self.journal_dir.join(format!("{date_str}.md"));
        let content = format!("# {date_str}\n\n{}\n", body.trim());
        fs::write(&path, content).await?;
        Ok(path)
    }

    /// Returns true if a journal file exists for the given date.
    pub fn dated_entry_exists(&self, date: NaiveDate) -> bool {
        self.journal_dir
            .join(format!("{}.md", date.format("%Y-%m-%d")))
            .exists()
    }

    pub async fn mark_done(&self) -> Result<()> {
        let state = JournalState {
            last_journaled_ts: Some(now_secs()),
        };
        if let Some(p) = self.state_file.parent() {
            fs::create_dir_all(p).await?;
        }
        fs::write(&self.state_file, serde_json::to_string_pretty(&state)?).await?;
        Ok(())
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── By-date backfill ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct BackfillOptions {
    /// Inclusive UTC date range; `None` means all dates.
    pub date_filter: Option<(NaiveDate, NaiveDate)>,
    pub max_chunk_chars: usize,
    /// Stop cleanly once cumulative input chars exceed this.
    pub daily_input_char_budget: Option<usize>,
}

#[derive(Debug, Default)]
pub struct BackfillStats {
    pub dates_processed: usize,
    pub dates_skipped_existing: usize,
    pub dates_skipped_empty_summary: usize,
    pub dates_failed: usize,
    pub entries_written: usize,
    pub chunks_summarized: usize,
    pub input_chars_used: usize,
    pub stopped_on_budget: bool,
}

/// Read every turn from `turns` (already chronological), bucket by UTC date,
/// chunk each day's transcript to fit `opts.max_chunk_chars`, and run the
/// supplied system prompt once per chunk. Multiple chunks for the same day
/// become `## HH:MM` blocks within `journal/YYYY-MM-DD.md`.
///
/// If `opts.daily_input_char_budget` is set, the run stops cleanly once
/// cumulative input chars exceed it — leaving already-written days intact
/// for resume. Existing journal files for a date are left untouched.
///
/// `turns` items are `(role, content, ts_unix_secs)`. Pass `JOURNAL_PROMPT`
/// (or a user override loaded via `load_journal_prompt`) as `system_prompt`.
pub async fn backfill_by_date<L: LLM>(
    journaler: &Journaler,
    llm: &L,
    system_prompt: &str,
    turns: &[(String, String, f64)],
    opts: &BackfillOptions,
) -> Result<BackfillStats> {
    use std::collections::BTreeMap;
    let mut by_date: BTreeMap<NaiveDate, Vec<(&str, &str, f64)>> = BTreeMap::new();
    for (role, content, ts) in turns {
        let dt = DateTime::<Utc>::from_timestamp(*ts as i64, 0)
            .ok_or_else(|| anyhow::anyhow!("bad ts {ts}"))?;
        let date = dt.date_naive();
        if let Some((from, to)) = opts.date_filter
            && (date < from || date > to)
        {
            continue;
        }
        by_date
            .entry(date)
            .or_default()
            .push((role.as_str(), content.as_str(), *ts));
    }

    let mut stats = BackfillStats::default();
    let total_dates = by_date.len();
    let mut idx = 0;
    'days: for (date, day_turns) in by_date {
        idx += 1;
        if journaler.dated_entry_exists(date) {
            stats.dates_skipped_existing += 1;
            continue;
        }

        let chunks = chunk_turns(&day_turns, opts.max_chunk_chars);
        let mut blocks: Vec<String> = Vec::new();

        for (chunk_i, chunk) in chunks.iter().enumerate() {
            let excerpt = chunk
                .iter()
                .map(|(role, content, _)| format!("[{role}]: {content}"))
                .collect::<Vec<_>>()
                .join("\n\n");
            let excerpt_chars = excerpt.len();

            // Local quota check before issuing the call.
            if let Some(budget) = opts.daily_input_char_budget
                && stats.input_chars_used + excerpt_chars > budget
            {
                tracing::warn!(
                    used = stats.input_chars_used,
                    budget,
                    date = %date,
                    "backfill: daily input budget reached, stopping cleanly"
                );
                stats.stopped_on_budget = true;
                break 'days;
            }

            let messages = vec![
                Message { role: Role::System, content: system_prompt.to_string() },
                Message { role: Role::User, content: excerpt },
            ];

            tracing::info!(
                date = %date,
                progress = format!("{idx}/{total_dates}"),
                chunk = format!("{}/{}", chunk_i + 1, chunks.len()),
                turns = chunk.len(),
                chars = excerpt_chars,
                "backfill: summarizing"
            );

            let summary = match llm.chat(&messages).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(date = %date, "backfill: LLM call failed: {e}");
                    stats.dates_failed += 1;
                    continue 'days;
                }
            };

            stats.input_chars_used += excerpt_chars;
            stats.chunks_summarized += 1;

            let trimmed = summary.trim();
            if trimmed.is_empty() || trimmed == "SKIP" {
                continue;
            }

            let first_ts = chunk.first().map(|(_, _, t)| *t).unwrap_or(0.0);
            let stamp = DateTime::<Utc>::from_timestamp(first_ts as i64, 0)
                .map(|d| d.format("%H:%M").to_string())
                .unwrap_or_else(|| "00:00".to_string());
            blocks.push(format!("## {stamp}\n\n{trimmed}"));
        }

        if blocks.is_empty() {
            stats.dates_skipped_empty_summary += 1;
            continue;
        }

        let body = blocks.join("\n\n");
        match journaler.write_dated_entry(date, &body).await {
            Ok(path) => {
                tracing::info!(
                    date = %date,
                    path = %path.display(),
                    blocks = blocks.len(),
                    "backfill: wrote entry"
                );
                stats.entries_written += 1;
            }
            Err(e) => {
                tracing::warn!(date = %date, "backfill: write failed: {e}");
                stats.dates_failed += 1;
            }
        }
        stats.dates_processed += 1;
    }

    Ok(stats)
}

/// Pack consecutive turns into chunks no larger than `max_chars` (when
/// rendered as `[role]: content` joined by blank lines). A single turn
/// exceeding the limit becomes its own chunk (truncation is the caller's
/// problem if that overflows the model).
fn chunk_turns<'a>(
    turns: &'a [(&'a str, &'a str, f64)],
    max_chars: usize,
) -> Vec<Vec<(&'a str, &'a str, f64)>> {
    const SEP_OVERHEAD: usize = 6; // "\n\n" + "[role]: "
    let mut chunks: Vec<Vec<(&str, &str, f64)>> = Vec::new();
    let mut current: Vec<(&str, &str, f64)> = Vec::new();
    let mut current_chars: usize = 0;
    for &(role, content, ts) in turns {
        let cost = role.len() + content.len() + SEP_OVERHEAD;
        if !current.is_empty() && current_chars + cost > max_chars {
            chunks.push(std::mem::take(&mut current));
            current_chars = 0;
        }
        current.push((role, content, ts));
        current_chars += cost;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}
