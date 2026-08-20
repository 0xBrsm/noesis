//! Topic consolidation — the pruning half of memory maintenance.
//!
//! Dreams (dream.rs) integrate new journal signal into topics; they only
//! accrete. Consolidation compresses the topic corpus itself: topic centroids
//! are clustered by embedding similarity (mechanical, no LLM), then each
//! cluster's full bodies go to an LLM that merges them into fewer files or
//! keeps them separate. Merging is cheap because retrieval is chunk-level —
//! one well-structured file beats several overlapping fragments.

use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;
use tokio::fs;

use crate::dream::{apply_plan, parse_plan};
use crate::llm::{LLM, Message, Role};

pub const CONSOLIDATE_PROMPT: &str = "\
# Consolidate: Topic Merging

You maintain the user's long-term memory topic files. You are given a cluster \
of topic files whose embeddings are similar — they may cover one subject in \
fragments, or they may be legitimately distinct.

Decide whether to merge some or all of them into fewer, better-organized \
files, or keep them separate.

Output a JSON plan. Schema:
{\"updates\": [{\"name\": \"<kebab-case-slug>\", \"operation\": \"create|replace\", \"summary\": \"<stable 1-2 sentence topic summary>\", \"content\": \"<markdown body>\"}], \"deletes\": [\"<slug>\"]}

Rules:
- Merge when topics overlap, fragment one subject, or one is stale context \
for another. The merged file must preserve ALL durable facts from its \
sources — reorganize and deduplicate, but never drop information.
- Use markdown headers to structure merged files; retrieval operates on \
sections, so internal structure costs nothing.
- Every slug you merged from must appear in \"deletes\" (a slug you replace \
into may stay).
- Keep genuinely distinct topics separate: return {\"updates\": [], \"deletes\": []}.
- Never invent facts. Output ONLY the JSON object — no preamble, no code fences.";

#[derive(Debug, Default)]
pub struct ConsolidateStats {
    pub clusters_processed: usize,
    pub clusters_kept_separate: usize,
    pub clusters_failed: usize,
    pub topics_written: usize,
    pub topics_deleted: usize,
}

/// Average each topic's chunk embeddings into one l2-normalized centroid.
/// Input pairs are (chunk path, embedding); output pairs are (slug, centroid)
/// where slug is the path with `topics/` and `.md` stripped.
pub fn topic_centroids(chunk_embeddings: &[(String, Vec<f32>)]) -> Vec<(String, Vec<f32>)> {
    let mut by_topic: BTreeMap<&str, (Vec<f32>, usize)> = BTreeMap::new();
    for (path, vec) in chunk_embeddings {
        let slug = path
            .strip_prefix("topics/")
            .unwrap_or(path)
            .strip_suffix(".md")
            .unwrap_or(path);
        let entry = by_topic
            .entry(slug)
            .or_insert_with(|| (vec![0.0; vec.len()], 0));
        if entry.0.len() != vec.len() {
            continue; // mixed embedding dims (model change mid-index); skip
        }
        for (s, v) in entry.0.iter_mut().zip(vec) {
            *s += v;
        }
        entry.1 += 1;
    }

    by_topic
        .into_iter()
        .filter_map(|(slug, (sum, n))| {
            if n == 0 {
                return None;
            }
            let mut c: Vec<f32> = sum.iter().map(|s| s / n as f32).collect();
            let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                return None;
            }
            c.iter_mut().for_each(|x| *x /= norm);
            Some((slug.to_string(), c))
        })
        .collect()
}

/// Union-find clustering: topics whose centroid cosine ≥ `threshold` join the
/// same cluster. Returns only clusters with ≥ 2 members, largest first.
/// O(n²) over topic count — fine for the hundreds-of-topics scale.
pub fn cluster_topics(centroids: &[(String, Vec<f32>)], threshold: f32) -> Vec<Vec<String>> {
    let n = centroids.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            let root = find(parent, parent[i]);
            parent[i] = root;
        }
        parent[i]
    }

    for i in 0..n {
        for j in (i + 1)..n {
            // Centroids are normalized, so dot product = cosine.
            let dot: f32 = centroids[i]
                .1
                .iter()
                .zip(&centroids[j].1)
                .map(|(a, b)| a * b)
                .sum();
            if dot >= threshold {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (i, (slug, _)) in centroids.iter().enumerate() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(slug.clone());
    }
    let mut clusters: Vec<Vec<String>> = groups.into_values().filter(|g| g.len() >= 2).collect();
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters
}

/// Run one LLM merge decision per cluster and apply the resulting plans.
/// `clusters` contain topic slugs. With `dry_run`, plans are logged but not
/// applied. Deletes outside the cluster are dropped as a safety net.
pub async fn consolidate_topics<L: LLM>(
    data_dir: &Path,
    llm: &L,
    system_prompt: &str,
    clusters: &[Vec<String>],
    dry_run: bool,
) -> Result<ConsolidateStats> {
    let topics_dir = data_dir.join("topics");
    let mut stats = ConsolidateStats::default();

    for (idx, cluster) in clusters.iter().enumerate() {
        let mut sections = String::new();
        let mut present: Vec<&str> = Vec::new();
        for slug in cluster {
            let path = topics_dir.join(format!("{slug}.md"));
            match fs::read_to_string(&path).await {
                Ok(body) => {
                    sections.push_str(&format!("=== {slug} ===\n{body}\n\n"));
                    present.push(slug);
                }
                Err(_) => tracing::warn!(slug, "consolidate: topic file missing, skipping"),
            }
        }
        if present.len() < 2 {
            continue;
        }

        tracing::info!(
            progress = format!("{}/{}", idx + 1, clusters.len()),
            topics = ?present,
            "consolidate: evaluating cluster"
        );

        let messages = vec![
            Message { role: Role::System, content: system_prompt.to_string() },
            Message {
                role: Role::User,
                content: format!(
                    "Candidate cluster ({} embedding-similar topics):\n\n{sections}",
                    present.len()
                ),
            },
        ];

        let plan = match llm.chat(&messages).await.and_then(|t| parse_plan(&t)) {
            Ok(mut plan) => {
                plan.deletes.retain(|d| {
                    let keep = present.iter().any(|s| *s == d);
                    if !keep {
                        tracing::warn!(slug = %d, "consolidate: dropping delete outside cluster");
                    }
                    keep
                });
                plan
            }
            Err(e) => {
                tracing::warn!("consolidate: cluster failed: {e}");
                stats.clusters_failed += 1;
                continue;
            }
        };

        stats.clusters_processed += 1;
        if plan.updates.is_empty() && plan.deletes.is_empty() {
            tracing::info!("consolidate: keeping cluster separate");
            stats.clusters_kept_separate += 1;
            continue;
        }

        tracing::info!(
            updates = ?plan.updates.iter().map(|u| u.name.as_str()).collect::<Vec<_>>(),
            deletes = ?plan.deletes,
            dry_run,
            "consolidate: merge plan"
        );

        if !dry_run {
            apply_plan(data_dir, &plan).await?;
            stats.topics_written += plan.updates.len();
            stats.topics_deleted += plan.deletes.len();
        }
    }

    Ok(stats)
}
