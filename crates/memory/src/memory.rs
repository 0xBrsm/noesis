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
    orphans_swept: bool,
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
            orphans_swept: false,
        })
    }

    pub fn open_from_config(cfg: &crate::Config) -> Result<Self> {
        Self::open(
            &cfg.data_dir,
            cfg.embed_model_name(),
            cfg.decay_half_life_days,
            cfg.semantic_weight,
            cfg.lexical_weight,
        )
    }

    // ── Index ─────────────────────────────────────────────────────────────────

    pub async fn index<L: LLM>(&mut self, llm: &L) -> Result<IndexResult> {
        // Orphans only appear after a crash, so one full-scan sweep per
        // process is enough — re-indexes during the same run stay consistent.
        if !self.orphans_swept {
            let orphans = db::sweep_orphan_vecs(&self.conn)?;
            if orphans > 0 {
                tracing::info!(orphans, "index: swept orphan chunks_vec rows");
            }
            self.orphans_swept = true;
        }

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
            if let Some(stored_hash) = db::get_file_hash(&self.conn, &rel)?
                && stored_hash == hash
            {
                result.skipped += 1;
                continue;
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
        // Embed everything first (network calls), then write in a single
        // transaction — a crash mid-file leaves the DB untouched instead of
        // half-replaced, and avoids orphaned chunks_vec rows.
        let chunks = db::chunk_markdown(content);
        let mut prepared = Vec::with_capacity(chunks.len());
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
                    first_vec_dims = first_vec_dims.or(Some(v.len()));
                    Some(v)
                }
                Err(e) => {
                    tracing::warn!("embed warning: {e}");
                    None
                }
            };
            prepared.push((id, chunk, embedding));
        }

        // DDL outside the transaction — vec0 virtual table creation.
        if let Some(dims) = first_vec_dims
            && let Err(e) = db::ensure_vec_table(&self.conn, dims)
        {
            tracing::warn!("ensure_vec_table failed: {e}");
        }

        let tx = self.conn.transaction()?;
        // File record must exist before chunks (foreign key constraint)
        db::upsert_file(
            &tx,
            &db::FileEntry {
                path: rel.to_string(),
                hash: hash.to_string(),
                mtime,
                size,
            },
        )?;
        db::delete_chunks(&tx, rel)?;

        for (id, chunk, embedding) in &prepared {
            db::upsert_chunk(
                &tx,
                id,
                rel,
                chunk.start_line,
                chunk.end_line,
                &self.model,
                &chunk.text,
                embedding.as_deref(),
            )?;
            if let Some(v) = embedding {
                db::upsert_vec(&tx, id, v)?;
            }
        }
        tx.commit()?;

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

    pub fn user_messages_since(&self, since_ts: f64) -> Result<usize> {
        db::count_user_messages_since(&self.conn, since_ts)
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

/// Render retrieved chunks as the exact `<retrieved_memory>` block the proxy
/// injects into the LLM input. Kept here so `noesis-query` and the live
/// proxy can't drift on framing.
pub fn format_context_block(rows: &[db::SearchRow]) -> String {
    let mut s = String::from(
        "<retrieved_memory>\nThe following notes from the user's persistent memory may be relevant. \
         Use them when they apply; ignore them when they don't.\n\n",
    );
    for row in rows {
        s.push_str(&format!(
            "[{} L{}-{}]\n{}\n\n",
            row.path, row.start_line, row.end_line, row.text
        ));
    }
    s.push_str("</retrieved_memory>");
    s
}

pub fn load_context_md(data_dir: &Path) -> Option<String> {
    let path = data_dir.join("context.md");
    std::fs::read_to_string(path).ok()
}

/// Read a user-overridable prompt from `{data_dir}/{name}.md`. If the file
/// is missing or unreadable, return the compiled default. Lets operators
/// tune framing without rebuilding the binary.
pub fn load_prompt(data_dir: &Path, name: &str, default: &str) -> String {
    let path = data_dir.join(format!("{name}.md"));
    std::fs::read_to_string(path).unwrap_or_else(|_| default.to_string())
}

pub fn load_journal_prompt(data_dir: &Path) -> String {
    load_prompt(data_dir, "journal", crate::journal::JOURNAL_PROMPT)
}

pub fn load_topic_prompt(data_dir: &Path) -> String {
    load_prompt(data_dir, "topic", crate::dream::TOPIC_PROMPT)
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
