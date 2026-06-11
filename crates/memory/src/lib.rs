//! noesis-memory — RAG memory layer.
//!
//! - SQLite + sqlite-vec storage for chunked document embeddings
//! - Local (fastembed) and remote (OpenAI-compatible) embedders / rerankers
//! - Header-based markdown chunking with contextual embedding
//! - Hybrid lexical + semantic search with temporal decay
//! - Auto-dream consolidation: extract durable facts from chat history

pub mod config;
pub mod consolidate;
pub mod db;
pub mod dream;
pub mod journal;
pub mod llm;
pub mod memory;

pub use config::Config;
pub use consolidate::{
    CONSOLIDATE_PROMPT, ConsolidateStats, cluster_topics, consolidate_topics, topic_centroids,
};
pub use db::{RetrievalEvent, SearchRow};
pub use dream::{
    AutoDream, DreamPlan, Operation, TOPIC_PROMPT, TopicUpdate, apply_plan,
    run_dream, run_dream_with_text,
};
pub use journal::{BackfillOptions, BackfillStats, JOURNAL_PROMPT, Journaler, backfill_by_date};
pub use llm::{Embedder, LLM, LocalLLM, Message, RemoteLLM, Reranker, Role};
pub use memory::{
    IndexResult, Memory, format_context_block, load_consolidate_prompt, load_context_md,
    load_journal_prompt, load_prompt, load_topic_prompt,
};
