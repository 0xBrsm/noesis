//! noesis-memory — RAG memory layer.
//!
//! - SQLite + sqlite-vec storage for chunked document embeddings
//! - Local (fastembed) and remote (OpenAI-compatible) embedders / rerankers
//! - Header-based markdown chunking with contextual embedding
//! - Hybrid lexical + semantic search with temporal decay
//! - Auto-dream consolidation: extract durable facts from chat history

pub mod config;
pub mod db;
pub mod dream;
pub mod journal;
pub mod llm;
pub mod memory;

pub use config::Config;
pub use db::SearchRow;
pub use dream::{
    AutoDream, ConsolidationPlan, DREAM_PROMPT, TopicUpdate, apply_plan, run_consolidation,
};
pub use journal::{Journaler, SUMMARIZER_PROMPT};
pub use llm::{Embedder, LLM, LocalLLM, Message, RemoteLLM, Reranker, Role};
pub use memory::{IndexResult, Memory, load_context_md};
