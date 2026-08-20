//! noesis-derive — turn stored sessions into titled conversations with
//! summaries and extracted facts.
//!
//! Two model calls' worth of work per session: one split, then one per
//! conversation. Results are banked in `derivations` against the session's
//! content hash, so a re-run skips sessions that have not changed.
//!
//! `--rerender` writes markdown from banked digests and makes no model calls at
//! all. Use it after a layout change; use a `PROMPT_FORMAT` bump when the
//! question itself changed.

use anyhow::Result;
use clap::Parser;
use noesis_memory::{
    Config as MemConfig, Embedder, Memory, RemoteLLM, SessionDigest, db, derive,
    derive_session, load_derive_prompt, render_digest,
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "noesis-derive",
    about = "Derive conversations, summaries, and facts from stored sessions"
)]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Re-render markdown from banked digests. No model calls.
    #[arg(long)]
    rerender: bool,

    /// Derive at most this many sessions, then stop. Re-run to continue.
    #[arg(long)]
    limit: Option<usize>,

    /// Skip sessions with fewer than this many messages.
    #[arg(long, default_value_t = 4)]
    min_messages: usize,

    /// Re-index after writing so the new files become searchable.
    #[arg(long, default_value_t = true)]
    reindex: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = MemConfig::load_or_default(cli.config.clone())?;
    let mut memory = Memory::open_from_config(&cfg)?;
    let out_dir = cfg.data_dir.join("conversations");
    std::fs::create_dir_all(&out_dir)?;

    let written = if cli.rerender {
        rerender(&memory, &out_dir)?
    } else {
        derive_all(&memory, &cfg, &cli, &out_dir).await?
    };

    tracing::info!(written, dir = %out_dir.display(), "derive: done");

    if written > 0 && cli.reindex {
        let embedder = Embedder::from_config(&cfg)?;
        let r = memory.index(&embedder).await?;
        tracing::info!(
            indexed = r.indexed,
            skipped = r.skipped,
            deleted = r.deleted,
            "derive: reindex complete"
        );
    }

    Ok(())
}

fn rerender(memory: &Memory, out_dir: &std::path::Path) -> Result<usize> {
    let banked = memory.all_derivations(derive::PROMPT_FORMAT)?;
    tracing::info!(
        digests = banked.len(),
        render_format = derive::RENDER_FORMAT,
        "derive: rerendering from banked digests"
    );
    let mut written = 0;
    for (session_id, raw) in banked {
        let digest: SessionDigest = match serde_json::from_str(&raw) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(session = %session_id, "skipping undecodable digest: {e}");
                continue;
            }
        };
        std::fs::write(out_dir.join(format!("{session_id}.md")), render_digest(&digest))?;
        written += 1;
    }
    Ok(written)
}

async fn derive_all(
    memory: &Memory,
    cfg: &MemConfig,
    cli: &Cli,
    out_dir: &std::path::Path,
) -> Result<usize> {
    let llm = RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.summarizer_model,
        &cfg.embed_model,
    );
    let instructions = load_derive_prompt(&cfg.data_dir);

    let session_ids = memory.session_ids()?;
    tracing::info!(
        sessions = session_ids.len(),
        prompt_format = derive::PROMPT_FORMAT,
        model = %cfg.summarizer_model,
        "derive: starting"
    );

    let mut written = 0;
    let mut skipped_banked = 0;
    let mut skipped_short = 0;
    let mut failed = 0;

    for session_id in session_ids {
        if let Some(limit) = cli.limit
            && written >= limit
        {
            tracing::info!(limit, "derive: limit reached, stopping");
            break;
        }

        let messages = memory.load_session(&session_id)?;
        if messages.len() < cli.min_messages {
            skipped_short += 1;
            continue;
        }
        let hash = db::session_content_hash(&messages);

        // A banked digest at this prompt format already answers the current
        // question for this exact content, so re-deriving would only cost money.
        if let Some(raw) = memory.load_derivation(&session_id, &hash, derive::PROMPT_FORMAT)? {
            if let Ok(digest) = serde_json::from_str::<SessionDigest>(&raw) {
                std::fs::write(
                    out_dir.join(format!("{session_id}.md")),
                    render_digest(&digest),
                )?;
                skipped_banked += 1;
                continue;
            }
        }

        let digest = match derive_session(&llm, &instructions, &session_id, &messages).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(session = %session_id, "derive failed: {e}");
                failed += 1;
                continue;
            }
        };

        memory.save_derivation(
            &session_id,
            &hash,
            derive::PROMPT_FORMAT,
            &serde_json::to_string(&digest)?,
        )?;
        std::fs::write(
            out_dir.join(format!("{session_id}.md")),
            render_digest(&digest),
        )?;
        written += 1;
        tracing::info!(
            session = %session_id,
            conversations = digest.conversations.len(),
            facts = digest.conversations.iter().map(|c| c.facts.len()).sum::<usize>(),
            "derive: wrote"
        );
    }

    tracing::info!(skipped_banked, skipped_short, failed, "derive: pass complete");
    Ok(written)
}
