// Adapted from claurst/src-rust/crates/query/src/auto_dream.rs
// Gate logic is verbatim; session counting uses our SQLite conversations table
// instead of claurst's JSONL transcript directory scan.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;

#[derive(Debug, Clone)]
pub struct AutoDreamConfig {
    pub min_hours: f64,
    pub min_sessions: usize,
}

impl Default for AutoDreamConfig {
    fn default() -> Self {
        Self { min_hours: 24.0, min_sessions: 5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsolidationState {
    pub last_consolidated_at: Option<u64>,
}

impl ConsolidationState {
    pub fn last_as_ts(&self) -> f64 {
        self.last_consolidated_at.unwrap_or(0) as f64
    }
}

pub struct AutoDream {
    config: AutoDreamConfig,
    lock_file: PathBuf,
    state_file: PathBuf,
}

impl AutoDream {
    pub fn new(data_dir: &PathBuf) -> Self {
        Self {
            config: AutoDreamConfig::default(),
            lock_file: data_dir.join(".consolidation_lock"),
            state_file: data_dir.join(".consolidation_state.json"),
        }
    }

    pub async fn load_state(&self) -> ConsolidationState {
        match fs::read_to_string(&self.state_file).await {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => ConsolidationState::default(),
        }
    }

    // Gate 1: enough time elapsed since last consolidation
    fn time_gate(&self, state: &ConsolidationState) -> bool {
        match state.last_consolidated_at {
            None => true,
            Some(last) => {
                let hours = now_secs().saturating_sub(last) as f64 / 3600.0;
                hours >= self.config.min_hours
            }
        }
    }

    // Gate 2: caller supplies session count from DB (avoids duplicate DB open)
    fn session_gate(&self, sessions_since: usize) -> bool {
        sessions_since >= self.config.min_sessions
    }

    // Gate 3: no active lock (stale after 1h)
    async fn lock_gate(&self) -> Result<bool> {
        if !self.lock_file.exists() {
            return Ok(true);
        }
        match fs::metadata(&self.lock_file).await {
            Ok(meta) => {
                let age = SystemTime::now()
                    .duration_since(meta.modified().unwrap_or(UNIX_EPOCH))
                    .unwrap_or(Duration::ZERO)
                    .as_secs();
                Ok(age > 3600)
            }
            Err(_) => Ok(true),
        }
    }

    /// Returns true (and acquires lock) if all gates pass.
    /// `sessions_since` = distinct sessions in DB since last_consolidated_at.
    pub async fn should_run(&self, sessions_since: usize) -> Result<bool> {
        let state = self.load_state().await;
        if !self.time_gate(&state) {
            return Ok(false);
        }
        if !self.session_gate(sessions_since) {
            return Ok(false);
        }
        if !self.lock_gate().await? {
            return Ok(false);
        }
        self.acquire_lock().await?;
        Ok(true)
    }

    pub async fn acquire_lock(&self) -> Result<()> {
        if let Some(p) = self.lock_file.parent() {
            fs::create_dir_all(p).await?;
        }
        fs::write(&self.lock_file, now_secs().to_string()).await?;
        Ok(())
    }

    pub async fn finish(&self) -> Result<()> {
        let state = ConsolidationState { last_consolidated_at: Some(now_secs()) };
        let json = serde_json::to_string_pretty(&state)?;
        if let Some(p) = self.state_file.parent() {
            fs::create_dir_all(p).await?;
        }
        fs::write(&self.state_file, json).await?;
        if self.lock_file.exists() {
            fs::remove_file(&self.lock_file).await?;
        }
        Ok(())
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}
