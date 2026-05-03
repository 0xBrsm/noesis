use anyhow::Result;
use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use std::path::Path;

// ── Schema ────────────────────────────────────────────────────────────────────

pub fn open(db_path: &Path) -> Result<Connection> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(db_path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    ensure_schema(&conn)?;
    Ok(conn)
}

/// Call once at startup, before any Connection is opened.
/// Registers sqlite-vec as an auto-extension for every subsequent connection.
pub fn register_vec_extension() -> bool {
    unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    }
    true
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS files (
            path        TEXT PRIMARY KEY,
            hash        TEXT NOT NULL,
            mtime       REAL NOT NULL,
            size        INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            path        TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            start_line  INTEGER NOT NULL,
            end_line    INTEGER NOT NULL,
            model       TEXT NOT NULL,
            text        TEXT NOT NULL,
            embedding   BLOB
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            id   UNINDEXED,
            path UNINDEXED,
            start_line UNINDEXED,
            end_line   UNINDEXED
        );
    ")?;
    Ok(())
}

pub fn ensure_vec_table(conn: &Connection, dims: usize) -> Result<()> {
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            id   TEXT PRIMARY KEY,
            embedding FLOAT[{dims}]
        );"
    ))?;
    Ok(())
}

// ── Chunker ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Chunk {
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

pub fn chunk_markdown(content: &str, chunk_tokens: usize, overlap_tokens: usize) -> Vec<Chunk> {
    let max_chars = (chunk_tokens * 4).max(32);
    let overlap_chars = overlap_tokens * 4;

    let mut chunks: Vec<Chunk> = Vec::new();
    // buffer: (owned text segment, 1-indexed line number)
    let mut current: Vec<(String, usize)> = Vec::new();
    let mut current_chars: usize = 0;

    let do_flush = |current: &[(String, usize)], chunks: &mut Vec<Chunk>| {
        if current.is_empty() { return; }
        chunks.push(Chunk {
            start_line: current[0].1,
            end_line: current[current.len() - 1].1,
            text: current.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("\n"),
        });
    };

    let do_overlap = |current: &[(String, usize)]| -> (Vec<(String, usize)>, usize) {
        if overlap_chars == 0 || current.is_empty() { return (vec![], 0); }
        let mut acc = 0usize;
        let mut kept: Vec<(String, usize)> = Vec::new();
        for entry in current.iter().rev() {
            acc += entry.0.len() + 1;
            kept.insert(0, entry.clone());
            if acc >= overlap_chars { break; }
        }
        let chars = kept.iter().map(|(t, _)| t.len() + 1).sum();
        (kept, chars)
    };

    for (i, raw_line) in content.split('\n').enumerate() {
        let line_no = i + 1;
        let segments: Vec<String> = if raw_line.is_empty() {
            vec![String::new()]
        } else {
            raw_line
                .as_bytes()
                .chunks(max_chars)
                .map(|c| String::from_utf8_lossy(c).into_owned())
                .collect()
        };

        for segment in segments {
            let line_size = segment.len() + 1;
            if current_chars + line_size > max_chars && !current.is_empty() {
                do_flush(&current, &mut chunks);
                let (kept, kept_chars) = do_overlap(&current);
                current = kept;
                current_chars = kept_chars;
            }
            current_chars += line_size;
            current.push((segment, line_no));
        }
    }
    do_flush(&current, &mut chunks);
    chunks
}

// ── Hashing ───────────────────────────────────────────────────────────────────

pub fn sha256(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    hex::encode(h.finalize())
}

pub fn chunk_id(path: &str, start_line: usize, end_line: usize, text_hash: &str) -> String {
    sha256(&format!("{path}:{start_line}:{end_line}:{text_hash}"))
}

// ── Index ─────────────────────────────────────────────────────────────────────

pub struct FileEntry {
    pub path: String,
    pub hash: String,
    pub mtime: f64,
    pub size: u64,
}

pub fn get_file_hash(conn: &Connection, path: &str) -> Result<Option<String>> {
    let mut stmt = conn.prepare_cached("SELECT hash FROM files WHERE path = ?")?;
    let mut rows = stmt.query(params![path])?;
    Ok(rows.next()?.map(|r| r.get(0).unwrap()))
}

pub fn upsert_file(conn: &Connection, entry: &FileEntry) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO files (path, hash, mtime, size) VALUES (?1, ?2, ?3, ?4)",
        params![entry.path, entry.hash, entry.mtime, entry.size],
    )?;
    Ok(())
}

pub fn delete_chunks(conn: &Connection, path: &str) -> Result<()> {
    // Delete from FTS first (no cascade)
    conn.execute(
        "DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
        params![path],
    )?;
    // Try vec table (may not exist)
    let _ = conn.execute(
        "DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE path = ?)",
        params![path],
    );
    conn.execute("DELETE FROM chunks WHERE path = ?", params![path])?;
    Ok(())
}

pub fn upsert_chunk(
    conn: &Connection,
    id: &str,
    path: &str,
    start_line: usize,
    end_line: usize,
    model: &str,
    text: &str,
    embedding: Option<&[f32]>,
) -> Result<()> {
    let blob: Option<Vec<u8>> = embedding.map(|v| {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    });

    conn.execute(
        "INSERT OR REPLACE INTO chunks (id, path, start_line, end_line, model, text, embedding)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![id, path, start_line, end_line, model, text, blob],
    )?;

    conn.execute(
        "INSERT OR REPLACE INTO chunks_fts (id, path, start_line, end_line, text)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![id, path, start_line, end_line, text],
    )?;

    Ok(())
}

pub fn upsert_vec(conn: &Connection, id: &str, embedding: &[f32]) -> Result<()> {
    let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
    conn.execute(
        "INSERT OR REPLACE INTO chunks_vec (id, embedding) VALUES (?1, ?2)",
        params![id, blob],
    )?;
    Ok(())
}

pub fn list_file_paths(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT path FROM files")?;
    let paths = stmt
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(paths)
}

// ── Search ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SearchRow {
    pub id: String,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub text_score: Option<f32>,
}

pub fn search_keyword(conn: &Connection, query: &str, limit: usize) -> Result<Vec<SearchRow>> {
    let fts_query = build_fts_query(query);
    let Some(fts_query) = fts_query else {
        return Ok(vec![]);
    };

    let mut stmt = conn.prepare(
        "SELECT id, path, start_line, end_line, text, bm25(chunks_fts) AS rank
         FROM chunks_fts
         WHERE chunks_fts MATCH ?1
         ORDER BY rank ASC
         LIMIT ?2",
    )?;

    let rows = stmt
        .query_map(params![fts_query, limit as i64], |r| {
            let rank: f64 = r.get(5)?;
            Ok(SearchRow {
                id: r.get(0)?,
                path: r.get(1)?,
                start_line: r.get::<_, i64>(2)? as usize,
                end_line: r.get::<_, i64>(3)? as usize,
                text: r.get(4)?,
                score: bm25_to_score(rank) as f32,
                vector_score: None,
                text_score: Some(bm25_to_score(rank) as f32),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(rows)
}

pub fn search_vector(
    conn: &Connection,
    query_vec: &[f32],
    limit: usize,
) -> Result<Vec<SearchRow>> {
    let blob: Vec<u8> = query_vec.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut stmt = conn.prepare(
        "SELECT cv.id, c.path, c.start_line, c.end_line, c.text,
                cv.distance
         FROM chunks_vec cv
         JOIN chunks c ON c.id = cv.id
         WHERE cv.embedding MATCH ?1
           AND k = ?2
         ORDER BY cv.distance ASC",
    )?;

    let rows = stmt
        .query_map(params![blob, limit as i64], |r| {
            let dist: f64 = r.get(5)?;
            let score = (1.0 / (1.0 + dist)) as f32;
            Ok(SearchRow {
                id: r.get(0)?,
                path: r.get(1)?,
                start_line: r.get::<_, i64>(2)? as usize,
                end_line: r.get::<_, i64>(3)? as usize,
                text: r.get(4)?,
                score,
                vector_score: Some(score),
                text_score: None,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(rows)
}

pub fn search_hybrid(
    conn: &Connection,
    query: &str,
    query_vec: &[f32],
    limit: usize,
    vector_weight: f32,
    text_weight: f32,
) -> Result<Vec<SearchRow>> {
    let candidate_limit = limit * 3;
    let vec_rows = search_vector(conn, query_vec, candidate_limit)?;
    let kw_rows = search_keyword(conn, query, candidate_limit)?;

    // Merge by id, weighted sum
    let mut scores: std::collections::HashMap<String, (f32, Option<f32>, Option<f32>, SearchRow)> =
        std::collections::HashMap::new();

    for row in vec_rows {
        let vs = row.score * vector_weight;
        scores.insert(row.id.clone(), (vs, Some(row.score), None, row));
    }
    for row in kw_rows {
        let ts = row.score * text_weight;
        if let Some(entry) = scores.get_mut(&row.id) {
            entry.0 += ts;
            entry.2 = Some(row.score);
        } else {
            scores.insert(row.id.clone(), (ts, None, Some(row.score), row));
        }
    }

    let mut merged: Vec<SearchRow> = scores
        .into_values()
        .map(|(combined, vs, ts, mut row)| {
            row.score = combined;
            row.vector_score = vs;
            row.text_score = ts;
            row
        })
        .collect();

    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    merged.truncate(limit);
    Ok(merged)
}

// ── FTS helpers ───────────────────────────────────────────────────────────────

fn build_fts_query(raw: &str) -> Option<String> {
    let tokens: Vec<String> = raw
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" AND "))
    }
}

fn bm25_to_score(rank: f64) -> f64 {
    if !rank.is_finite() {
        return 1.0 / (1.0 + 999.0);
    }
    if rank < 0.0 {
        let r = -rank;
        r / (1.0 + r)
    } else {
        1.0 / (1.0 + rank)
    }
}
