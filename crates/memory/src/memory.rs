use anyhow::{Context, Result};
use chrono::Datelike;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::db::{self, SearchRow};
use crate::llm::{LLM, Reranker};

pub struct Memory {
    conn: Connection,
    data_dir: PathBuf,  // ~/.noesis — everything (db, models, journal/, topics/, ...)
    model: String,
    vector_weight: f32,
    text_weight: f32,
    decay_half_life_days: f32,
}

impl Memory {
    pub fn open(
        data_dir: &Path,
        model: &str,
        decay_half_life_days: f32,
        semantic_weight: f32,
        lexical_weight: f32,
    ) -> Result<Self> {
        let db_path = data_dir.join("memory.db");
        let _ = db::register_vec_extension();
        let conn = db::open(&db_path)?;

        Ok(Self {
            conn,
            data_dir: data_dir.to_path_buf(),
            model: model.to_string(),
            vector_weight: semantic_weight,
            text_weight: lexical_weight,
            decay_half_life_days,
        })
    }

    // ── Index ─────────────────────────────────────────────────────────────────

    pub async fn index<L: LLM>(&mut self, llm: &L) -> Result<IndexResult> {
        // Scan journal/ (dated entries, decay) and topics/ (evergreens, no decay).
        let mut files = Vec::new();
        for sub in &["journal", "topics"] {
            let dir = self.data_dir.join(sub);
            if !dir.exists() {
                std::fs::create_dir_all(&dir)?;
            }
            files.extend(collect_md_files(&dir));
        }
        let stored_paths: std::collections::HashSet<String> =
            db::list_file_paths(&self.conn)?.into_iter().collect();

        let mut result = IndexResult::default();

        for abs_path in &files {
            let rel = rel_path(abs_path, &self.data_dir)?;
            let content = std::fs::read_to_string(abs_path)
                .with_context(|| format!("reading {}", abs_path.display()))?;
            let hash = db::sha256(&content);
            let meta = abs_path.metadata()?;
            let mtime = meta
                .modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs_f64();

            // Skip if hash unchanged
            if let Some(stored_hash) = db::get_file_hash(&self.conn, &rel)? {
                if stored_hash == hash {
                    result.skipped += 1;
                    continue;
                }
            }

            self.index_file(&rel, &content, &hash, mtime, meta.len(), llm)
                .await?;
            result.indexed += 1;
        }

        // Prune stale files
        let disk_paths: std::collections::HashSet<String> = files
            .iter()
            .map(|p| rel_path(p, &self.data_dir))
            .collect::<Result<_>>()?;
        for stale in stored_paths.difference(&disk_paths) {
            db::delete_chunks(&self.conn, stale)?;
            self.conn
                .execute("DELETE FROM files WHERE path = ?", [stale])?;
            result.deleted += 1;
        }

        Ok(result)
    }

    async fn index_file<L: LLM>(
        &mut self,
        rel: &str,
        content: &str,
        hash: &str,
        mtime: f64,
        size: u64,
        llm: &L,
    ) -> Result<()> {
        // File record must exist before chunks (foreign key constraint)
        db::upsert_file(
            &self.conn,
            &db::FileEntry {
                path: rel.to_string(),
                hash: hash.to_string(),
                mtime,
                size,
            },
        )?;

        db::delete_chunks(&self.conn, rel)?;

        let chunks = db::chunk_markdown(content);
        let mut first_vec_dims: Option<usize> = None;

        for chunk in &chunks {
            let text_hash = db::sha256(&chunk.text);
            let id = db::chunk_id(rel, chunk.start_line, chunk.end_line, &text_hash);

            let embed_input = if chunk.context.is_empty() {
                chunk.text.clone()
            } else {
                format!("{}\n\n{}", chunk.context, chunk.text)
            };

            let embedding = match llm.embed(&embed_input).await {
                Ok(v) => {
                    if first_vec_dims.is_none() {
                        first_vec_dims = Some(v.len());
                        if let Err(e) = db::ensure_vec_table(&self.conn, v.len()) {
                            tracing::warn!("ensure_vec_table failed: {e}");
                        }
                    }
                    Some(v)
                }
                Err(e) => {
                    tracing::warn!("embed warning: {e}");
                    None
                }
            };

            db::upsert_chunk(
                &self.conn,
                &id,
                rel,
                chunk.start_line,
                chunk.end_line,
                &self.model,
                &chunk.text,
                embedding.as_deref(),
            )?;

            if let Some(ref v) = embedding {
                db::upsert_vec(&self.conn, &id, v)?;
            }
        }

        Ok(())
    }

    // ── Conversation history ──────────────────────────────────────────────────

    pub fn insert_turn(&self, session_id: &str, turn_index: usize, role: &str, content: &str, response_id: Option<&str>) -> Result<()> {
        db::insert_turn(&self.conn, session_id, turn_index, role, content, response_id)
    }

    pub fn load_recent_turns(&self) -> Result<Vec<(String, String)>> {
        db::load_recent_turns(&self.conn)
    }

    pub fn load_turns_since(&self, since_ts: f64) -> Result<Vec<(String, String, f64)>> {
        db::load_turns_since(&self.conn, since_ts)
    }

    pub fn last_response_id(&self) -> Result<Option<String>> {
        db::last_response_id(&self.conn)
    }

    pub fn sessions_since(&self, since_ts: f64) -> Result<usize> {
        db::count_sessions_since(&self.conn, since_ts)
    }

    // ── Search ────────────────────────────────────────────────────────────────

    pub fn search_keyword(&self, query: &str, limit: usize) -> Result<Vec<SearchRow>> {
        db::search_keyword(&self.conn, query, limit)
    }

    /// True if a vec0 table exists in the DB; caller should compute an embedding
    /// before invoking [`Memory::search_with_vec`] when this is true.
    pub fn vec_available(&self) -> bool {
        db::vec_table_exists(&self.conn)
    }

    /// Synchronous search with a caller-supplied query embedding. Embedding
    /// happens outside Memory so the caller can drop any locks before awaiting.
    pub fn search_with_vec(
        &self,
        query: &str,
        query_vec: Option<&[f32]>,
        limit: usize,
        reranker: Option<&Reranker>,
    ) -> Result<Vec<SearchRow>> {
        let mut results = if let Some(qv) = query_vec {
            db::search_hybrid(
                &self.conn,
                query,
                qv,
                limit,
                self.vector_weight,
                self.text_weight,
            )?
        } else {
            db::search_keyword(&self.conn, query, limit)?
        };

        if let Some(rr) = reranker {
            let docs: Vec<String> = results.iter().map(|r| r.text.clone()).collect();
            let mut scored = rr.rerank(query, &docs)?;
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let old = results;
            results = scored
                .into_iter()
                .map(|(i, score)| {
                    let mut row = old[i].clone();
                    row.score = score;
                    row
                })
                .collect();
        }

        let now_days = now_unix_days();
        for r in &mut results {
            r.raw_score = r.score;
            r.score *= decay_multiplier(&r.path, now_days, self.decay_half_life_days);
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    pub fn search_for_context_with_vec(
        &self,
        query: &str,
        query_vec: Option<&[f32]>,
        reranker: Option<&Reranker>,
        candidates: usize,
        max_results: usize,
        threshold: f32,
    ) -> Result<Vec<db::SearchRow>> {
        let results = self.search_with_vec(query, query_vec, candidates, reranker)?;
        Ok(results
            .into_iter()
            .filter(|r| r.raw_score >= threshold)
            .take(max_results)
            .collect())
    }

}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[derive(Default, Debug)]
pub struct IndexResult {
    pub indexed: usize,
    pub skipped: usize,
    pub deleted: usize,
}

fn now_unix_days() -> f32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f32()
        / 86400.0
}

fn decay_multiplier(path: &str, _now_days: f32, half_life: f32) -> f32 {
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    // Only dated files (YYYY-MM-DD.md) decay; everything else is evergreen
    if !is_dated_filename(filename) {
        return 1.0;
    }
    let date_str = &filename[..10]; // "YYYY-MM-DD"
    let file_days = match chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        Ok(d) => d.num_days_from_ce() as f32,
        Err(_) => return 1.0,
    };
    let today_days = chrono::Local::now().date_naive().num_days_from_ce() as f32;
    let age_days = (today_days - file_days).max(0.0);
    let lambda = std::f32::consts::LN_2 / half_life;
    (-lambda * age_days).exp()
}

fn is_dated_filename(name: &str) -> bool {
    name.len() == 13 // "YYYY-MM-DD.md"
        && name.ends_with(".md")
        && name.as_bytes()[4] == b'-'
        && name.as_bytes()[7] == b'-'
        && name[..4].chars().all(|c| c.is_ascii_digit())
        && name[5..7].chars().all(|c| c.is_ascii_digit())
        && name[8..10].chars().all(|c| c.is_ascii_digit())
}

pub fn load_context_md(data_dir: &Path) -> Option<String> {
    let path = data_dir.join("context.md");
    std::fs::read_to_string(path).ok()
}

fn collect_md_files(dir: &Path) -> Vec<PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().map_or(false, |x| x == "md")
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn rel_path(abs: &Path, data_dir: &Path) -> Result<String> {
    abs.strip_prefix(data_dir)
        .with_context(|| format!("{} not under data_dir", abs.display()))
        .map(|p| p.to_string_lossy().into_owned())
}
