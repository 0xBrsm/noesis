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
    pub embed_model: String,        // remote embed model
    pub local_embed: bool,
    pub local_embed_model: String,  // fastembed model name
    pub rerank_model: String,       // reranker model name

    // Paths
    pub workspace: PathBuf,         // ~/noesis  — user files
    pub data_dir: PathBuf,          // ~/.noesis — db, models, config

    // Search
    pub search_results: usize,
    pub semantic_weight: f32,
    pub lexical_weight: f32,

    // Retrieval (chat context injection)
    pub retrieval_candidates: usize,
    pub retrieval_limit: usize,
    pub retrieval_threshold: f32,

    // Memory
    pub decay_half_life_days: f32,

    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let home = PathBuf::from(home);
        Self {
            base_url: "http://localhost:11434/v1".to_string(),
            api_key: "ollama".to_string(),
            chat_model: "llama3.2".to_string(),
            embed_model: "nomic-embed-text".to_string(),
            local_embed: false,
            local_embed_model: "Xenova/all-MiniLM-L6-v2".to_string(),
            rerank_model: "Xenova/ms-marco-MiniLM-L-6-v2".to_string(),
            workspace: home.join("noesis"),
            data_dir: home.join(".noesis"),
            search_results: 5,
            semantic_weight: 0.7,
            lexical_weight: 0.3,
            retrieval_candidates: 10,
            retrieval_limit: 3,
            retrieval_threshold: 0.5,
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
    local_embed: Option<bool>,
    local_embed_model: Option<String>,
    rerank_model: Option<String>,
    workspace: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    search_results: Option<usize>,
    semantic_weight: Option<f32>,
    lexical_weight: Option<f32>,
    retrieval_candidates: Option<usize>,
    retrieval_limit: Option<usize>,
    retrieval_threshold: Option<f32>,
    decay_half_life_days: Option<f32>,
}

#[derive(Parser, Debug)]
#[command(name = "noesis", about = "Personal memory chat agent")]
#[command(subcommand_required = false, arg_required_else_help = false)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    #[arg(long, global = true)]
    pub workspace: Option<PathBuf>,

    #[arg(long, global = true)]
    pub data_dir: Option<PathBuf>,

    #[arg(long, global = true)]
    pub local_embed: bool,

    #[arg(long, global = true)]
    pub debug: bool,

    #[arg(long, global = true)]
    pub base_url: Option<String>,

    #[arg(long, global = true)]
    pub api_key: Option<String>,

    #[arg(long, global = true)]
    pub chat_model: Option<String>,

    #[arg(long, global = true)]
    pub embed_model: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Scan workspace and index memory files
    Index {
        #[arg(long)]
        force: bool,
    },
    /// Search memory and print ranked results
    Search {
        query: String,
        #[arg(short, long, default_value = "5")]
        limit: usize,
        #[arg(long)]
        lexical_only: bool,
        #[arg(long)]
        full: bool,
    },
    /// Interactive chat (default)
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

    // Config file: CLI --config > ~/.noesis/config.toml
    let toml_path = cli.config.clone().or_else(default_config_path);
    if let Some(path) = toml_path {
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let toml: TomlConfig = toml::from_str(&text)?;
            if let Some(v) = toml.base_url            { cfg.base_url            = v; }
            if let Some(v) = toml.api_key             { cfg.api_key             = v; }
            if let Some(v) = toml.chat_model          { cfg.chat_model          = v; }
            if let Some(v) = toml.embed_model         { cfg.embed_model         = v; }
            if let Some(v) = toml.local_embed         { cfg.local_embed         = v; }
            if let Some(v) = toml.local_embed_model   { cfg.local_embed_model   = v; }
            if let Some(v) = toml.rerank_model        { cfg.rerank_model        = v; }
            if let Some(v) = toml.workspace           { cfg.workspace           = v; }
            if let Some(v) = toml.data_dir            { cfg.data_dir            = v; }
            if let Some(v) = toml.search_results      { cfg.search_results      = v; }
            if let Some(v) = toml.semantic_weight     { cfg.semantic_weight     = v; }
            if let Some(v) = toml.lexical_weight      { cfg.lexical_weight      = v; }
            if let Some(v) = toml.retrieval_candidates{ cfg.retrieval_candidates= v; }
            if let Some(v) = toml.retrieval_limit     { cfg.retrieval_limit     = v; }
            if let Some(v) = toml.retrieval_threshold { cfg.retrieval_threshold = v; }
            if let Some(v) = toml.decay_half_life_days{ cfg.decay_half_life_days= v; }
        }
    }

    // CLI flags override
    if let Some(v) = &cli.base_url   { cfg.base_url   = v.clone(); }
    if let Some(v) = &cli.api_key    { cfg.api_key    = v.clone(); }
    if let Some(v) = &cli.chat_model { cfg.chat_model = v.clone(); }
    if let Some(v) = &cli.embed_model{ cfg.embed_model= v.clone(); }
    if let Some(v) = &cli.workspace  { cfg.workspace  = v.clone(); }
    if let Some(v) = &cli.data_dir   { cfg.data_dir   = v.clone(); }
    if cli.local_embed               { cfg.local_embed = true; }
    cfg.debug = cli.debug;

    // OPENAI_API_KEY auto-detect
    if let Ok(k) = std::env::var("OPENAI_API_KEY") {
        if !k.is_empty() {
            cfg.api_key = k;
            if cfg.base_url == "http://localhost:11434/v1" {
                cfg.base_url = "https://api.openai.com/v1".to_string();
            }
            if cfg.chat_model == "llama3.2" {
                cfg.chat_model = "gpt-5".to_string();
            }
            if cfg.embed_model == "nomic-embed-text" {
                cfg.embed_model = "text-embedding-3-small".to_string();
            }
        }
    }

    Ok(cfg)
}

fn default_config_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".noesis").join("config.toml"))
}
