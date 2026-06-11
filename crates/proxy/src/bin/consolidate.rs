//! noesis-consolidate — merge embedding-similar topic files.
//!
//! Dreams accrete topics; this is the pruning half. Topic centroids are
//! computed from already-indexed chunk embeddings, clustered by cosine
//! similarity (union-find, no LLM), then each cluster's full bodies go to
//! the summarizer model which merges them into fewer files or keeps them
//! separate. Run `--list` to preview clusters, `--dry-run` to see merge
//! plans without applying them.

use anyhow::Result;
use clap::Parser;
use noesis_memory::{
    Config as MemConfig, Embedder, Memory, RemoteLLM, cluster_topics, consolidate_topics,
    load_consolidate_prompt, topic_centroids,
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "noesis-consolidate", about = "Cluster and merge embedding-similar topic files")]
struct Cli {
    /// Path to config.toml. Defaults to ~/.noesis/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Minimum centroid cosine similarity for two topics to cluster.
    #[arg(long, default_value_t = 0.80)]
    threshold: f32,

    /// Print clusters and exit — no LLM calls.
    #[arg(long, default_value_t = false)]
    list: bool,

    /// Run LLM merge decisions but don't apply the plans.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Truncate clusters larger than this to their first N members,
    /// keeping per-call context bounded.
    #[arg(long, default_value_t = 8)]
    max_cluster: usize,

    /// Re-index after merging so topic file changes become searchable.
    #[arg(long, default_value_t = true)]
    reindex: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let cfg = MemConfig::load_or_default(cli.config)?;

    let mut memory = Memory::open_from_config(&cfg)?;
    let embeddings = memory.topic_embeddings()?;
    let centroids = topic_centroids(&embeddings);
    let mut clusters = cluster_topics(&centroids, cli.threshold);

    tracing::info!(
        topics = centroids.len(),
        clusters = clusters.len(),
        threshold = cli.threshold,
        "consolidate: clustered topic centroids"
    );

    if clusters.is_empty() {
        tracing::info!("consolidate: no clusters at this threshold, nothing to do");
        return Ok(());
    }

    for cluster in &mut clusters {
        if cluster.len() > cli.max_cluster {
            tracing::info!(
                size = cluster.len(),
                max = cli.max_cluster,
                "consolidate: truncating oversized cluster"
            );
            cluster.truncate(cli.max_cluster);
        }
    }

    if cli.list {
        for (i, cluster) in clusters.iter().enumerate() {
            println!("cluster {}: {}", i + 1, cluster.join(", "));
        }
        return Ok(());
    }

    let llm = RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.summarizer_model,
        &cfg.embed_model,
    );
    let prompt = load_consolidate_prompt(&cfg.data_dir);

    let stats = consolidate_topics(&cfg.data_dir, &llm, &prompt, &clusters, cli.dry_run).await?;
    tracing::info!(
        processed = stats.clusters_processed,
        kept_separate = stats.clusters_kept_separate,
        failed = stats.clusters_failed,
        written = stats.topics_written,
        deleted = stats.topics_deleted,
        dry_run = cli.dry_run,
        "consolidate: done"
    );

    if !cli.dry_run && cli.reindex && stats.topics_written + stats.topics_deleted > 0 {
        let embedder = Embedder::from_config(&cfg)?;
        let r = memory.index(&embedder).await?;
        tracing::info!(
            indexed = r.indexed,
            skipped = r.skipped,
            deleted = r.deleted,
            "consolidate: reindex complete"
        );
    }

    Ok(())
}
