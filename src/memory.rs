use anyhow::{Context, Result};
use chrono::{Datelike, Local};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::db::{self, Chunk, SearchRow};
use crate::llm::{LLM, Reranker};

const FLUSH_SYSTEM_PROMPT: &str = "\
You are extracting durable facts from a conversation to store in long-term memory.
Write only facts worth keeping across future sessions: decisions made, preferences \
expressed, important context, things the user wants to be remembered.
Be concise. Use short bullet points. Omit small talk and transient details.
If there is nothing worth storing, reply with exactly: @@SILENT@@";

pub struct Memory {
    conn: Connection,
    workspace: PathBuf,
    model: String,
    chunk_tokens: usize,
    chunk_overlap: usize,
    vector_weight: f32,
    text_weight: f32,
    vec_available: bool,
    decay_half_life_days: f32,
}

impl Memory {
    pub fn open(
        workspace: &Path,
        model: &str,
        chunk_tokens: usize,
        chunk_overlap: usize,
        decay_half_life_days: f32,
        semantic_weight: f32,
        lexical_weight: f32,
    ) -> Result<Self> {
        let db_path = workspace.join("memory.db");
        let vec_available = db::register_vec_extension();
        let conn = db::open(&db_path)?;

        Ok(Self {
            conn,
            workspace: workspace.to_path_buf(),
            model: model.to_string(),
            chunk_tokens,
            chunk_overlap,
            vector_weight: semantic_weight,
            text_weight: lexical_weight,
            vec_available,
            decay_half_life_days,
        })
    }

    // ── Index ─────────────────────────────────────────────────────────────────

    pub async fn index<L: LLM>(&mut self, llm: &L) -> Result<IndexResult> {
        let memory_dir = self.workspace.join("memory");
        if !memory_dir.exists() {
            std::fs::create_dir_all(&memory_dir)?;
            return Ok(IndexResult::default());
        }

        let files = collect_md_files(&memory_dir);
        let stored_paths: std::collections::HashSet<String> =
            db::list_file_paths(&self.conn)?.into_iter().collect();

        let mut result = IndexResult::default();

        for abs_path in &files {
            let rel = rel_path(abs_path, &self.workspace)?;
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
            .map(|p| rel_path(p, &self.workspace))
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

        let chunks = db::chunk_markdown(content, self.chunk_tokens, self.chunk_overlap);
        let mut first_vec_dims: Option<usize> = None;

        for chunk in &chunks {
            let text_hash = db::sha256(&chunk.text);
            let id = db::chunk_id(rel, chunk.start_line, chunk.end_line, &text_hash);

            let embedding = if self.vec_available {
                match llm.embed(&chunk.text).await {
                    Ok(v) => {
                        if first_vec_dims.is_none() {
                            first_vec_dims = Some(v.len());
                            db::ensure_vec_table(&self.conn, v.len())?;
                        }
                        Some(v)
                    }
                    Err(e) => {
                        eprintln!("embed warning: {e}");
                        None
                    }
                }
            } else {
                None
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

    // ── Search ────────────────────────────────────────────────────────────────

    pub fn search_keyword(&self, query: &str, limit: usize) -> Result<Vec<SearchRow>> {
        db::search_keyword(&self.conn, query, limit)
    }

    pub async fn search<L: LLM>(
        &self,
        query: &str,
        limit: usize,
        llm: &L,
        reranker: Option<&Reranker>,
    ) -> Result<Vec<SearchRow>> {
        let mut results = if self.vec_available {
            let query_vec = llm.embed(query).await?;
            db::search_hybrid(
                &self.conn,
                query,
                &query_vec,
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
            results = scored.into_iter().map(|(i, score)| {
                let mut row = old[i].clone();
                row.score = score;
                row
            }).collect();
        }

        let now_days = now_unix_days();
        for r in &mut results {
            r.score *= decay_multiplier(&r.path, now_days, self.decay_half_life_days);
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    pub async fn search_for_context<L: LLM>(
        &self,
        query: &str,
        llm: &L,
        reranker: Option<&Reranker>,
        candidates: usize,
        max_results: usize,
        threshold: f32,
    ) -> Result<Vec<db::SearchRow>> {
        let results = self.search(query, candidates, llm, reranker).await?;
        Ok(results.into_iter().filter(|r| r.score >= threshold).take(max_results).collect())
    }

    // ── Flush ─────────────────────────────────────────────────────────────────

    pub async fn flush<L: LLM>(
        &mut self,
        conversation: &[crate::llm::Message],
        llm: &L,
    ) -> Result<Option<String>> {
        use crate::llm::{Message, Role};

        let mut messages = vec![Message {
            role: Role::System,
            content: FLUSH_SYSTEM_PROMPT.to_string(),
        }];
        messages.extend_from_slice(conversation);

        let response = llm.chat(&messages).await?;
        let extracted = response.trim().to_string();

        if extracted.is_empty() || extracted.contains("@@SILENT@@") {
            return Ok(None);
        }

        let memory_dir = self.workspace.join("memory");
        std::fs::create_dir_all(&memory_dir)?;

        let today = Local::now().format("%Y-%m-%d").to_string();
        let dated_file = memory_dir.join(format!("{today}.md"));

        if dated_file.exists() {
            let existing = std::fs::read_to_string(&dated_file)?;
            let sep = if existing.ends_with("\n\n") { "" } else { "\n\n" };
            std::fs::write(&dated_file, format!("{existing}{sep}{extracted}\n"))?;
        } else {
            std::fs::write(&dated_file, format!("{extracted}\n"))?;
        }

        // Re-index the updated file
        let content = std::fs::read_to_string(&dated_file)?;
        let hash = db::sha256(&content);
        let meta = dated_file.metadata()?;
        let mtime = meta
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        let rel = rel_path(&dated_file, &self.workspace)?;
        self.index_file(&rel, &content, &hash, mtime, meta.len(), llm)
            .await?;

        Ok(Some(extracted))
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

fn decay_multiplier(path: &str, now_days: f32, half_life: f32) -> f32 {
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

pub fn load_context_md(workspace: &Path) -> Option<String> {
    let path = workspace.join("context.md");
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

fn rel_path(abs: &Path, workspace: &Path) -> Result<String> {
    abs.strip_prefix(workspace)
        .with_context(|| format!("{} not under workspace", abs.display()))
        .map(|p| p.to_string_lossy().into_owned())
}
