use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    // LLM
    pub base_url: String,
    pub api_key: String,
    pub chat_model: String,
    pub embed_model: String,

    // Workspace
    pub workspace: PathBuf,

    // Indexing
    pub chunk_tokens: usize,
    pub chunk_overlap: usize,

    // Search (noesis search command — results shown to user)
    pub search_results: usize,
    pub semantic_weight: f32,
    pub lexical_weight: f32,

    // Retrieval (chat context injection)
    pub retrieval_candidates: usize,  // candidates fetched before reranking
    pub retrieval_limit: usize,       // max chunks injected into prompt
    pub retrieval_threshold: f32,     // min score after reranking to include

    // Chat
    pub history_chars: usize,
    pub flush_every: usize,

    // Memory
    pub decay_half_life_days: f32,

    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
            api_key: "ollama".to_string(),
            chat_model: "llama3.2".to_string(),
            embed_model: "nomic-embed-text".to_string(),
            workspace: PathBuf::from(home).join(".noesis"),
            chunk_tokens: 400,
            chunk_overlap: 80,
            search_results: 5,
            semantic_weight: 0.7,
            lexical_weight: 0.3,
            retrieval_candidates: 10,
            retrieval_limit: 3,
            retrieval_threshold: 0.5,
            history_chars: 16_000,
            flush_every: 10,
            decay_half_life_days: 30.0,
            debug: false,
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
    search_results: Option<usize>,
    semantic_weight: Option<f32>,
    lexical_weight: Option<f32>,
    retrieval_candidates: Option<usize>,
    retrieval_limit: Option<usize>,
    retrieval_threshold: Option<f32>,
    history_chars: Option<usize>,
    flush_every: Option<usize>,
    decay_half_life_days: Option<f32>,
}

#[derive(Parser, Debug)]
#[command(name = "noesis", about = "Local LLM chat with persistent memory")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Path to config TOML
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Workspace directory for memory files
    #[arg(long, global = true)]
    pub workspace: Option<PathBuf>,

    /// Use local fastembed (all-MiniLM-L6-v2-Q) instead of remote API for embeddings
    #[arg(long, global = true)]
    pub local_embed: bool,

    /// Print debug timing and full prompts to stderr
    #[arg(long, global = true)]
    pub debug: bool,

    /// OpenAI-compatible API base URL
    #[arg(long, global = true)]
    pub base_url: Option<String>,

    /// API key
    #[arg(long, global = true)]
    pub api_key: Option<String>,

    /// Chat model name
    #[arg(long, global = true)]
    pub chat_model: Option<String>,

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

        /// Use lexical (FTS5) search only, skip vector search
        #[arg(long)]
        lexical_only: bool,

        /// Show full chunk text instead of a truncated snippet
        #[arg(long)]
        full: bool,
    },
    /// Interactive chat
    Chat,
}

pub fn load(cli: &Cli) -> Result<Config> {
    let mut cfg = Config::default();

    // Load .env from current directory if present
    if let Ok(text) = std::fs::read_to_string(".env") {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some((k, v)) = line.trim_start_matches("export ").split_once('=') {
                let v = v.trim_matches('"').trim_matches('\'');
                unsafe { std::env::set_var(k.trim(), v) };
            }
        }
    }

    let toml_path = cli.config.clone().or_else(default_config_path);
    if let Some(path) = toml_path {
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let toml: TomlConfig = toml::from_str(&text)?;
            if let Some(v) = toml.base_url             { cfg.base_url             = v; }
            if let Some(v) = toml.api_key              { cfg.api_key              = v; }
            if let Some(v) = toml.chat_model           { cfg.chat_model           = v; }
            if let Some(v) = toml.embed_model          { cfg.embed_model          = v; }
            if let Some(v) = toml.workspace            { cfg.workspace            = v; }
            if let Some(v) = toml.chunk_tokens         { cfg.chunk_tokens         = v; }
            if let Some(v) = toml.chunk_overlap        { cfg.chunk_overlap        = v; }
            if let Some(v) = toml.search_results       { cfg.search_results       = v; }
            if let Some(v) = toml.semantic_weight      { cfg.semantic_weight      = v; }
            if let Some(v) = toml.lexical_weight       { cfg.lexical_weight       = v; }
            if let Some(v) = toml.retrieval_candidates { cfg.retrieval_candidates = v; }
            if let Some(v) = toml.retrieval_limit      { cfg.retrieval_limit      = v; }
            if let Some(v) = toml.retrieval_threshold  { cfg.retrieval_threshold  = v; }
            if let Some(v) = toml.history_chars        { cfg.history_chars        = v; }
            if let Some(v) = toml.flush_every          { cfg.flush_every          = v; }
            if let Some(v) = toml.decay_half_life_days { cfg.decay_half_life_days = v; }
        }
    }

    if let Some(v) = &cli.base_url    { cfg.base_url    = v.clone(); }
    if let Some(v) = &cli.api_key     { cfg.api_key     = v.clone(); }
    if let Some(v) = &cli.chat_model  { cfg.chat_model  = v.clone(); }
    if let Some(v) = &cli.embed_model { cfg.embed_model = v.clone(); }
    if let Some(v) = &cli.workspace   { cfg.workspace   = v.clone(); }
    cfg.debug = cli.debug;

    // OPENAI_API_KEY in environment overrides config
    if let Ok(k) = std::env::var("OPENAI_API_KEY") {
        if !k.is_empty() {
            cfg.api_key = k;
            if cfg.base_url == "http://localhost:11434/v1" {
                cfg.base_url = "https://api.openai.com/v1".to_string();
            }
            if cfg.chat_model == "llama3.2" {
                cfg.chat_model = "gpt-5-mini".to_string();
            }
        }
    }

    Ok(cfg)
}

fn default_config_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".config").join("noesis").join("config.toml"))
}
