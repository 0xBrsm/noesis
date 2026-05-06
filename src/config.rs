use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub base_url: String,
    pub api_key: String,
    pub chat_model: String,
    pub embed_model: String,
    pub workspace: PathBuf,
    pub chunk_tokens: usize,
    pub chunk_overlap: usize,
    pub search_limit: usize,
    pub flush_every: usize,
    pub context_candidates: usize,
    pub context_limit: usize,
    pub context_threshold: f32,
    pub history_chars: usize,
}

impl Default for Config {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
            api_key: "ollama".to_string(),
            chat_model: "llama3.2".to_string(),
            embed_model: "nomic-embed-text".to_string(),
            workspace: PathBuf::from(home).join(".llmchat"),
            chunk_tokens: 400,
            chunk_overlap: 80,
            search_limit: 5,
            flush_every: 10,
            context_candidates: 10,
            context_limit: 3,
            context_threshold: 0.5,
            history_chars: 16_000,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct TomlConfig {
    base_url: Option<String>,
    api_key: Option<String>,
    chat_model: Option<String>,
    embed_model: Option<String>,
    workspace: Option<PathBuf>,
    chunk_tokens: Option<usize>,
    chunk_overlap: Option<usize>,
    search_limit: Option<usize>,
    flush_every: Option<usize>,
}

#[derive(Parser, Debug)]
#[command(name = "llmchat", about = "Local LLM chat with persistent memory")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Path to config TOML
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Workspace directory for memory files
    #[arg(long, global = true)]
    pub workspace: Option<PathBuf>,

    /// Use local fastembed (nomic-embed-text-v1.5) instead of remote API for embeddings
    #[arg(long, global = true)]
    pub local_embed: bool,

    /// OpenAI-compatible API base URL
    #[arg(long, global = true)]
    pub base_url: Option<String>,

    /// API key
    #[arg(long, global = true)]
    pub api_key: Option<String>,

    /// Embedding model name
    #[arg(long, global = true)]
    pub embed_model: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Scan workspace and index memory files
    Index {
        /// Force re-index even if files are unchanged
        #[arg(long)]
        force: bool,
    },
    /// Search memory and print ranked results
    Search {
        /// Search query
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Use keyword (FTS5) search only, skip vector search
        #[arg(long)]
        keyword_only: bool,

        /// Show full chunk text instead of a truncated snippet
        #[arg(long)]
        full: bool,
    },
    /// Interactive chat (coming soon)
    Chat,
}

pub fn load(cli: &Cli) -> Result<Config> {
    let mut cfg = Config::default();

    let toml_path = cli.config.clone().or_else(default_config_path);
    if let Some(path) = toml_path {
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let toml: TomlConfig = toml::from_str(&text)?;
            if let Some(v) = toml.base_url     { cfg.base_url     = v; }
            if let Some(v) = toml.api_key       { cfg.api_key      = v; }
            if let Some(v) = toml.chat_model    { cfg.chat_model   = v; }
            if let Some(v) = toml.embed_model   { cfg.embed_model  = v; }
            if let Some(v) = toml.workspace     { cfg.workspace    = v; }
            if let Some(v) = toml.chunk_tokens  { cfg.chunk_tokens = v; }
            if let Some(v) = toml.chunk_overlap { cfg.chunk_overlap = v; }
            if let Some(v) = toml.search_limit  { cfg.search_limit = v; }
            if let Some(v) = toml.flush_every   { cfg.flush_every  = v; }
        }
    }

    if let Some(v) = &cli.base_url    { cfg.base_url    = v.clone(); }
    if let Some(v) = &cli.api_key     { cfg.api_key     = v.clone(); }
    if let Some(v) = &cli.embed_model { cfg.embed_model = v.clone(); }
    if let Some(v) = &cli.workspace   { cfg.workspace   = v.clone(); }

    Ok(cfg)
}

fn default_config_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".config").join("llmchat").join("config.toml"))
}
