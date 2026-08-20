use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
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
    // `conversations` held individual messages, which collides with the derived
    // unit of the same name. Rename before the CREATE below, or an empty
    // `messages` would exist and the old rows would be stranded.
    let has_old_table = conn
        .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversations'")?
        .exists([])?;
    if has_old_table {
        conn.execute_batch(
            "DROP INDEX IF EXISTS conversations_session;
             DROP INDEX IF EXISTS conversations_role_ts;
             ALTER TABLE conversations RENAME TO messages;",
        )?;
    }

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

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            turn_index  INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            response_id TEXT,
            ts          REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS messages_session
            ON messages (session_id, turn_index);

        CREATE INDEX IF NOT EXISTS messages_role_ts
            ON messages (role, ts);

        -- Banked model output. `digest` is the raw phase-1/phase-2 JSON, kept so
        -- a rendering change can be replayed without paying for the calls again.
        -- `content_hash` covers the session's messages, so an edited or extended
        -- session derives afresh while an untouched one is skipped.
        CREATE TABLE IF NOT EXISTS derivations (
            session_id    TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            prompt_format INTEGER NOT NULL,
            digest        TEXT NOT NULL,
            derived_at    REAL NOT NULL,
            PRIMARY KEY (session_id, content_hash, prompt_format)
        );

        CREATE TABLE IF NOT EXISTS retrievals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           REAL NOT NULL,
            session_id   TEXT NOT NULL,
            turn_index   INTEGER NOT NULL,
            query        TEXT NOT NULL,
            chunk_id     TEXT NOT NULL,
            chunk_path   TEXT NOT NULL,
            rank         INTEGER NOT NULL,
            score        REAL NOT NULL,
            raw_score    REAL NOT NULL,
            vector_score REAL,
            text_score   REAL,
            injected     INTEGER NOT NULL,
            label        REAL
        );

        CREATE INDEX IF NOT EXISTS retrievals_chunk_ts
            ON retrievals (chunk_id, ts);

        CREATE INDEX IF NOT EXISTS retrievals_session_turn
            ON retrievals (session_id, turn_index);
    ")?;

    // Migration for retrievals tables created before the label column existed.
    let has_label = conn
        .prepare("SELECT 1 FROM pragma_table_info('retrievals') WHERE name='label'")?
        .exists([])?;
    if !has_label {
        conn.execute("ALTER TABLE retrievals ADD COLUMN label REAL", [])?;
    }
    Ok(())
}

pub fn vec_table_exists(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
        [],
        |r| r.get::<_, i64>(0),
    ).unwrap_or(0) > 0
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
    /// Header hierarchy path prepended to the embedding input (not stored in DB).
    /// e.g. "Setup > Installation"
    pub context: String,
}

/// Split markdown on header boundaries (#/##/###).
/// Each section (header + body) becomes one chunk.
/// Content before the first header is its own chunk with empty context.
/// `context` is the breadcrumb path of parent headers, used for contextual embedding.
///
/// A leading YAML frontmatter block (`---\n...\n---`) is skipped — it's
/// navigational metadata (summary, aliases) used by the consolidation agent
/// and would otherwise clutter retrieval results.
pub fn chunk_markdown(content: &str) -> Vec<Chunk> {
    let lines: Vec<&str> = content.lines().collect();
    let body_start = frontmatter_end(&lines);
    let mut chunks: Vec<Chunk> = Vec::new();

    // Track header hierarchy [h1, h2, h3]
    let mut headers: [Option<String>; 3] = [None, None, None];

    let mut section_start: usize = body_start;
    let mut section_lines: Vec<&str> = Vec::new();
    let mut section_context = String::new();

    let flush = |lines: &[&str], start: usize, context: &str, chunks: &mut Vec<Chunk>| {
        let text = lines.join("\n").trim().to_string();
        if text.is_empty() { return; }
        chunks.push(Chunk {
            start_line: start + 1,
            end_line: start + lines.len(),
            text,
            context: context.to_string(),
        });
    };

    for (i, &line) in lines.iter().enumerate().skip(body_start) {
        let level = header_level(line);
        if let Some(lvl) = level {
            // Flush current section
            flush(&section_lines, section_start, &section_context, &mut chunks);

            // Update header hierarchy
            let title = line.trim_start_matches('#').trim().to_string();
            headers[lvl] = Some(title);
            // Clear deeper levels
            for h in headers.iter_mut().skip(lvl + 1) { *h = None; }

            section_context = headers.iter()
                .filter_map(|h| h.as_deref())
                .collect::<Vec<_>>()
                .join(" > ");

            section_start = i;
            section_lines = vec![line];
        } else {
            section_lines.push(line);
        }
    }
    flush(&section_lines, section_start, &section_context, &mut chunks);

    chunks
}

/// Index of the first body line after a leading YAML frontmatter block,
/// or 0 if the content doesn't start with one. A frontmatter block is
/// `---` on its own line, followed by metadata, followed by another `---`.
fn frontmatter_end(lines: &[&str]) -> usize {
    if lines.first().map(|l| l.trim()) != Some("---") {
        return 0;
    }
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim() == "---" {
            return i + 1;
        }
    }
    // Unclosed frontmatter — treat the whole file as body.
    0
}

fn header_level(line: &str) -> Option<usize> {
    if !line.starts_with('#') { return None; }
    let level = line.chars().take_while(|&c| c == '#').count();
    if level <= 3 && line.chars().nth(level) == Some(' ') {
        Some(level - 1) // 0-indexed: H1=0, H2=1, H3=2
    } else {
        None
    }
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

/// Drop rows from `chunks_vec` whose `id` no longer exists in `chunks`.
/// `delete_chunks` deletes vec rows via an id-join through `chunks`; if a prior
/// run crashed between writing the two tables, the vec row becomes unreachable
/// and a future `upsert_vec` may collide on it (vec0 doesn't honor INSERT OR
/// REPLACE the same way regular tables do).
pub fn sweep_orphan_vecs(conn: &Connection) -> Result<usize> {
    if !vec_table_exists(conn) {
        return Ok(0);
    }
    let n = conn.execute(
        "DELETE FROM chunks_vec WHERE id NOT IN (SELECT id FROM chunks)",
        [],
    )?;
    Ok(n)
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
    // vec0 doesn't reliably honor INSERT OR REPLACE — explicit DELETE+INSERT
    // is the documented-safe pattern and idempotent against orphan rows.
    conn.execute("DELETE FROM chunks_vec WHERE id = ?", params![id])?;
    conn.execute(
        "INSERT INTO chunks_vec (id, embedding) VALUES (?1, ?2)",
        params![id, blob],
    )?;
    Ok(())
}

/// Chunk embeddings for top-level topic files, as (path, vector) pairs.
/// Nested paths (e.g. hand-dropped doc trees under topics/) are excluded —
/// consolidation only merges flat `topics/<slug>.md` files.
pub fn load_topic_embeddings(conn: &Connection) -> Result<Vec<(String, Vec<f32>)>> {
    let mut stmt = conn.prepare(
        "SELECT path, embedding FROM chunks
         WHERE path LIKE 'topics/%' AND path NOT LIKE 'topics/%/%'
           AND embedding IS NOT NULL",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (path, blob) = row?;
        out.push((path, decode_embedding(&blob)));
    }
    Ok(out)
}

fn decode_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

pub fn load_chunk_embedding(conn: &Connection, id: &str) -> Result<Option<Vec<f32>>> {
    let blob: Option<Vec<u8>> = conn
        .prepare_cached("SELECT embedding FROM chunks WHERE id = ?1")?
        .query_row([id], |r| r.get(0))
        .optional()?;
    Ok(blob.map(|b| decode_embedding(&b)))
}

pub fn list_file_paths(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT path FROM files")?;
    let paths = stmt
        .query_map([], |r| r.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(paths)
}

// ── Conversation history ──────────────────────────────────────────────────────

pub fn insert_turn(
    conn: &Connection,
    session_id: &str,
    turn_index: usize,
    role: &str,
    content: &str,
    response_id: Option<&str>,
) -> Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs_f64();
    conn.execute(
        "INSERT INTO messages (session_id, turn_index, role, content, response_id, ts)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![session_id, turn_index as i64, role, content, response_id, ts],
    )?;
    Ok(())
}

/// Returns the response_id of the most recent assistant turn, for previous_response_id chaining.
pub fn last_response_id(conn: &Connection) -> Result<Option<String>> {
    let mut stmt = conn.prepare(
        "SELECT response_id FROM messages
         WHERE role = 'assistant' AND response_id IS NOT NULL
         ORDER BY ts DESC LIMIT 1",
    )?;
    let mut rows = stmt.query([])?;
    Ok(rows.next()?.and_then(|r| r.get(0).ok()))
}

/// Load all prior turns in chronological order, ready to use as history.
pub fn load_recent_turns(conn: &Connection) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT role, content FROM messages
         ORDER BY ts ASC, turn_index ASC",
    )?;
    let turns = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(turns)
}

/// Load turns with `ts > since_ts` in chronological order — used by the
/// background journal summarizer to grab the delta since its last entry.
pub fn load_turns_since(conn: &Connection, since_ts: f64) -> Result<Vec<(String, String, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT role, content, ts FROM messages
         WHERE ts > ?
         ORDER BY ts ASC, turn_index ASC",
    )?;
    let turns = stmt
        .query_map([since_ts], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f64>(2)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(turns)
}

/// Count user messages stored after `since_ts` (unix seconds as f64). This is
/// the signal for "enough new input to consolidate" — session_id is a poor
/// proxy because it's per-process (long conversations undercount, restart
/// loops overcount).
pub fn count_user_messages_since(conn: &Connection, since_ts: f64) -> Result<usize> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM messages WHERE role = 'user' AND ts > ?",
        [since_ts],
        |r| r.get(0),
    )?;
    Ok(count as usize)
}

// ── Derivation ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    pub ts: f64,
}

/// Session ids in first-message order, so a resumed run processes oldest first.
pub fn list_session_ids(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT session_id FROM messages
         GROUP BY session_id
         ORDER BY MIN(ts) ASC",
    )?;
    let ids = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(ids)
}

pub fn load_session(conn: &Connection, session_id: &str) -> Result<Vec<SessionMessage>> {
    let mut stmt = conn.prepare(
        "SELECT role, content, ts FROM messages
         WHERE session_id = ?
         ORDER BY turn_index ASC, ts ASC",
    )?;
    let rows = stmt
        .query_map([session_id], |r| {
            Ok(SessionMessage {
                role: r.get(0)?,
                content: r.get(1)?,
                ts: r.get(2)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Identifies a session's exact content. Appending a turn changes it, so the
/// session re-derives; nothing else does.
pub fn session_content_hash(messages: &[SessionMessage]) -> String {
    let mut hasher = Sha256::new();
    for m in messages {
        hasher.update(m.role.as_bytes());
        hasher.update([0u8]);
        hasher.update(m.content.as_bytes());
        hasher.update([0u8]);
    }
    format!("{:x}", hasher.finalize())
}

pub fn load_derivation(
    conn: &Connection,
    session_id: &str,
    content_hash: &str,
    prompt_format: u32,
) -> Result<Option<String>> {
    let digest = conn
        .query_row(
            "SELECT digest FROM derivations
             WHERE session_id = ?1 AND content_hash = ?2 AND prompt_format = ?3",
            params![session_id, content_hash, prompt_format],
            |r| r.get::<_, String>(0),
        )
        .optional()?;
    Ok(digest)
}

pub fn save_derivation(
    conn: &Connection,
    session_id: &str,
    content_hash: &str,
    prompt_format: u32,
    digest: &str,
) -> Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs_f64();
    conn.execute(
        "INSERT OR REPLACE INTO derivations
         (session_id, content_hash, prompt_format, digest, derived_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![session_id, content_hash, prompt_format, digest, ts],
    )?;
    Ok(())
}

/// Every banked digest at `prompt_format`, for a re-render with no model calls.
pub fn all_derivations(conn: &Connection, prompt_format: u32) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT session_id, digest FROM derivations
         WHERE prompt_format = ?
         ORDER BY derived_at ASC",
    )?;
    let rows = stmt
        .query_map([prompt_format], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

// ── Search ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SearchRow {
    pub id: String,
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub score: f32,       // decayed score — used for ranking
    pub raw_score: f32,   // pre-decay score — used for threshold filtering
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
            let s = bm25_to_score(rank) as f32;
            Ok(SearchRow {
                id: r.get(0)?,
                path: r.get(1)?,
                start_line: r.get::<_, i64>(2)? as usize,
                end_line: r.get::<_, i64>(3)? as usize,
                text: r.get(4)?,
                score: s,
                raw_score: s,
                vector_score: None,
                text_score: Some(s),
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

    let mut rows: Vec<SearchRow> = stmt
        .query_map(params![blob, limit as i64], |r| {
            let dist: f64 = r.get(5)?;
            // Convert L2 distance to cosine similarity for normalized vectors:
            // cos_sim = 1 - (dist² / 2), clamped to [0, 1]
            let cos_sim = (1.0 - (dist * dist / 2.0)).clamp(0.0, 1.0) as f32;
            Ok(SearchRow {
                id: r.get(0)?,
                path: r.get(1)?,
                start_line: r.get::<_, i64>(2)? as usize,
                end_line: r.get::<_, i64>(3)? as usize,
                text: r.get(4)?,
                score: cos_sim,
                raw_score: cos_sim,
                vector_score: Some(cos_sim),
                text_score: None,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Min-max normalize so top result = 1.0 and scores spread meaningfully
    if rows.len() > 1 {
        let max = rows[0].score;
        let min = rows.last().map(|r| r.score).unwrap_or(0.0);
        let range = max - min;
        if range > 1e-6 {
            for row in &mut rows {
                let norm = (row.score - min) / range;
                row.score = norm;
                row.raw_score = norm;
                row.vector_score = Some(norm);
            }
        }
    }

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
            row.raw_score = combined;
            row.vector_score = vs;
            row.text_score = ts;
            row
        })
        .collect();

    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    merged.truncate(limit);
    Ok(merged)
}

// ── Retrieval log ─────────────────────────────────────────────────────────────

/// One retrieval candidate to record: the row, its post-rerank rank, and
/// whether it survived chain dedup and was actually injected. `label` is a
/// relevance signal filled in by offline replay (None for live traffic).
pub struct RetrievalEvent<'a> {
    pub row: &'a SearchRow,
    pub rank: usize,
    pub injected: bool,
    pub label: Option<f32>,
}

/// Append retrieval candidates for one query to the log. Training data for
/// learned ranking/retention later; cheap to write, never read on the hot path.
pub fn log_retrievals(
    conn: &Connection,
    session_id: &str,
    turn_index: usize,
    query: &str,
    events: &[RetrievalEvent<'_>],
) -> Result<()> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs_f64();
    let mut stmt = conn.prepare_cached(
        "INSERT INTO retrievals
            (ts, session_id, turn_index, query, chunk_id, chunk_path,
             rank, score, raw_score, vector_score, text_score, injected, label)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
    )?;
    for e in events {
        stmt.execute(params![
            ts,
            session_id,
            turn_index as i64,
            query,
            e.row.id,
            e.row.path,
            e.rank as i64,
            e.row.score as f64,
            e.row.raw_score as f64,
            e.row.vector_score.map(|v| v as f64),
            e.row.text_score.map(|v| v as f64),
            e.injected as i64,
            e.label.map(|v| v as f64),
        ])?;
    }
    Ok(())
}

/// A user turn paired with the assistant response that followed it.
#[derive(Debug, Clone)]
pub struct QaPair {
    pub session_id: String,
    pub turn_index: usize,
    pub ts: f64,
    pub question: String,
    pub answer: String,
}

/// Load user/assistant pairs that have no retrieval log rows yet, oldest
/// first — the replay work queue. `limit` of 0 means no limit.
pub fn load_unreplayed_qa_pairs(conn: &Connection, limit: usize) -> Result<Vec<QaPair>> {
    let mut stmt = conn.prepare(
        "SELECT u.session_id, u.turn_index, u.ts, u.content, a.content
         FROM messages u
         JOIN messages a
           ON a.session_id = u.session_id
          AND a.turn_index = u.turn_index + 1
          AND a.role = 'assistant'
         WHERE u.role = 'user'
           AND NOT EXISTS (
               SELECT 1 FROM retrievals r
               WHERE r.session_id = u.session_id AND r.turn_index = u.turn_index
           )
         ORDER BY u.ts, u.turn_index
         LIMIT ?1",
    )?;
    let limit = if limit == 0 { i64::MAX } else { limit as i64 };
    let rows = stmt
        .query_map([limit], |r| {
            Ok(QaPair {
                session_id: r.get(0)?,
                turn_index: r.get::<_, i64>(1)? as usize,
                ts: r.get(2)?,
                question: r.get(3)?,
                answer: r.get(4)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

// ── FTS helpers ───────────────────────────────────────────────────────────────

static STOP_WORDS: std::sync::LazyLock<std::collections::HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        [
            "a", "an", "the", "this", "that", "these", "those",
            "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "can", "may", "might",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "about", "into", "through", "before", "after", "between", "under", "over",
            "and", "or", "but", "if", "then", "because", "as", "while",
            "when", "where", "what", "which", "who", "how", "why",
            "up", "out", "no", "not", "so", "just", "also", "than", "too",
            "get", "go", "use", "make", "set", "show", "find", "tell", "help",
        ]
        .into_iter()
        .collect()
    });

fn build_fts_query(raw: &str) -> Option<String> {
    // Prefix-match each token (`token*`) so a query for "world" hits indexed
    // tokens like "worldbuilding" or "worlds". The default unicode61 tokenizer
    // splits only on non-alphanumeric, so without `*` a phrase query for
    // "world" requires that exact standalone token in the index.
    let tokens: Vec<String> = raw
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .filter(|t| t.len() >= 2)
        .filter(|t| !STOP_WORDS.contains(t.to_lowercase().as_str()))
        .map(|t| format!("{t}*"))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The rename runs against a live memory.db holding real history, so the
    /// rows have to survive it.
    #[test]
    fn migration_carries_conversation_rows_into_messages() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                turn_index  INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                response_id TEXT,
                ts          REAL NOT NULL
             );
             CREATE INDEX conversations_session ON conversations (session_id, turn_index);
             CREATE INDEX conversations_role_ts ON conversations (role, ts);
             INSERT INTO conversations (session_id, turn_index, role, content, response_id, ts)
             VALUES ('s1', 0, 'user', 'hello', NULL, 1.0),
                    ('s1', 1, 'assistant', 'hi', 'resp_1', 2.0);",
        )
        .unwrap();

        ensure_schema(&conn).unwrap();

        let rows = load_session(&conn, "s1").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].content, "hello");
        assert_eq!(rows[1].role, "assistant");
        assert!(!conn
            .prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversations'")
            .unwrap()
            .exists([])
            .unwrap());
    }

    #[test]
    fn ensure_schema_is_idempotent_on_a_fresh_db() {
        let conn = Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        ensure_schema(&conn).unwrap();
        assert!(list_session_ids(&conn).unwrap().is_empty());
    }

    #[test]
    fn content_hash_tracks_appended_turns() {
        let m = |c: &str| SessionMessage { role: "user".into(), content: c.into(), ts: 0.0 };
        let one = vec![m("a")];
        let two = vec![m("a"), m("b")];
        assert_eq!(session_content_hash(&one), session_content_hash(&[m("a")]));
        assert_ne!(session_content_hash(&one), session_content_hash(&two));
        // Field boundaries are delimited, so a split cannot be forged by
        // concatenation.
        let joined = vec![m("ab")];
        assert_ne!(session_content_hash(&joined), session_content_hash(&two));
    }

    #[test]
    fn derivations_round_trip_and_are_scoped_by_format() {
        let conn = Connection::open_in_memory().unwrap();
        ensure_schema(&conn).unwrap();
        save_derivation(&conn, "s1", "hash1", 1, "{\"x\":1}").unwrap();

        assert_eq!(
            load_derivation(&conn, "s1", "hash1", 1).unwrap().as_deref(),
            Some("{\"x\":1}")
        );
        // A prompt-format bump invalidates the bank, which is the whole point.
        assert!(load_derivation(&conn, "s1", "hash1", 2).unwrap().is_none());
        // So does edited session content.
        assert!(load_derivation(&conn, "s1", "hash2", 1).unwrap().is_none());
        assert_eq!(all_derivations(&conn, 1).unwrap().len(), 1);
        assert!(all_derivations(&conn, 2).unwrap().is_empty());
    }
}
