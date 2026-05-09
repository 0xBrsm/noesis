//! Strict TOML-driven config. Every field is required — there are no
//! in-code defaults. Copy `config.example.toml` from the repo to
//! `~/.noesis/config.toml` and edit it.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    // Server
    pub bind: SocketAddr,
    pub upstream: String,

    // LLM
    pub base_url: String,
    pub api_key: String,
    pub chat_model: String,
    pub embed_model: String,
    pub local_embed: bool,
    pub local_embed_model: String,
    pub rerank_model: String,

    // Path (`~/...` is expanded against $HOME). All noesis state lives here:
    // memory.db, models/, journal/, topics/, context.md, .consolidation_*, etc.
    pub data_dir: PathBuf,

    // Search
    pub semantic_weight: f32,
    pub lexical_weight: f32,

    // Retrieval
    pub retrieval_candidates: usize,
    pub retrieval_limit: usize,
    pub retrieval_threshold: f32,

    // Memory decay
    pub decay_half_life_days: f32,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| {
            format!(
                "reading config at {} \
                 (copy config.example.toml from the repo to that path and edit it)",
                path.display()
            )
        })?;
        let cfg: Config = toml::from_str(&text)
            .with_context(|| format!("parsing TOML config at {}", path.display()))?;
        Ok(expand_paths(cfg))
    }

    /// `$HOME/.noesis/config.toml`, or `None` if `$HOME` is unset.
    pub fn default_path() -> Option<PathBuf> {
        let home = std::env::var("HOME").ok()?;
        Some(PathBuf::from(home).join(".noesis").join("config.toml"))
    }
}

fn expand_paths(mut cfg: Config) -> Config {
    cfg.data_dir = expand_tilde(cfg.data_dir);
    cfg
}

fn expand_tilde(p: PathBuf) -> PathBuf {
    let Some(s) = p.to_str() else {
        return p;
    };
    if let Some(rest) = s.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    p
}
