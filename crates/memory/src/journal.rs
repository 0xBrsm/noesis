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
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub const SUMMARIZER_PROMPT: &str = "\
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
