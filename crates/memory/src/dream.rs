// Adapted from claurst/src-rust/crates/query/src/auto_dream.rs.
// Gate logic and state-file semantics follow claurst; session counting uses
// our SQLite conversations table instead of claurst's JSONL transcript scan.
// `DREAM_PROMPT` lifts framing/phrasing from claurst's `consolidation_prompt()`
// (auto_dream.rs:286) and adapts it: noesis returns a JSON plan that the host
// applies, where claurst forks a tool-using subagent that writes files itself.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::fs;

use crate::llm::{Message, RemoteLLM, Role};

pub const DREAM_PROMPT: &str = "\
# Dream: Memory Consolidation

You are performing a dream — a reflective pass over the user's memory files. \
Synthesize what has been learned recently into durable, well-organized memories \
so that future sessions can orient quickly.

The memory store has two kinds of files:
- **Topics** — evergreen, one coherent subject per file (a project, a domain of \
  interest, a stable preference, a long-running thread). These are what you \
  produce.
- **Journal** — dated entries from recent sessions. These are your input signal.

Guiding principles (apply across all phases):
- Merge new signal into existing topic files rather than creating near-duplicates.
- Convert relative dates to absolute dates.
- Drop facts the journal contradicts.
- Write topic content as if it will be retrieved out of context: name the \
  subject, use concrete, searchable phrasing, avoid pronouns referring to the \
  journal.
- Be precise and concise. Skip trivial or transient details.

You will work in three phases. Each phase has a specific output format — follow \
it exactly.";

#[derive(Debug, Clone, Deserialize)]
pub struct TopicUpdate {
    pub name: String,
    pub operation: String, // "create" | "replace" | "append"
    #[serde(default)]
    pub description: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ConsolidationPlan {
    #[serde(default)]
    pub updates: Vec<TopicUpdate>,
    #[serde(default)]
    pub deletes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AutoDreamConfig {
    pub min_hours: f64,
    pub min_sessions: usize,
}

impl Default for AutoDreamConfig {
    fn default() -> Self {
        Self { min_hours: 24.0, min_sessions: 5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsolidationState {
    pub last_consolidated_at: Option<u64>,
}

impl ConsolidationState {
    pub fn last_as_ts(&self) -> f64 {
        self.last_consolidated_at.unwrap_or(0) as f64
    }
}

pub struct AutoDream {
    config: AutoDreamConfig,
    lock_file: PathBuf,
    state_file: PathBuf,
}

impl AutoDream {
    pub fn new(data_dir: &PathBuf) -> Self {
        Self {
            config: AutoDreamConfig::default(),
            lock_file: data_dir.join(".consolidation_lock"),
            state_file: data_dir.join(".consolidation_state.json"),
        }
    }

    pub async fn load_state(&self) -> ConsolidationState {
        match fs::read_to_string(&self.state_file).await {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => ConsolidationState::default(),
        }
    }

    // Gate 1: enough time elapsed since last consolidation
    fn time_gate(&self, state: &ConsolidationState) -> bool {
        match state.last_consolidated_at {
            None => true,
            Some(last) => {
                let hours = now_secs().saturating_sub(last) as f64 / 3600.0;
                hours >= self.config.min_hours
            }
        }
    }

    // Gate 2: caller supplies session count from DB (avoids duplicate DB open)
    fn session_gate(&self, sessions_since: usize) -> bool {
        sessions_since >= self.config.min_sessions
    }

    // Gate 3: no active lock (stale after 1h)
    async fn lock_gate(&self) -> Result<bool> {
        if !self.lock_file.exists() {
            return Ok(true);
        }
        match fs::metadata(&self.lock_file).await {
            Ok(meta) => {
                let age = SystemTime::now()
                    .duration_since(meta.modified().unwrap_or(UNIX_EPOCH))
                    .unwrap_or(Duration::ZERO)
                    .as_secs();
                Ok(age > 3600)
            }
            Err(_) => Ok(true),
        }
    }

    /// Returns true (and acquires lock) if all gates pass.
    /// `sessions_since` = distinct sessions in DB since last_consolidated_at.
    pub async fn should_run(&self, sessions_since: usize) -> Result<bool> {
        let state = self.load_state().await;
        if !self.time_gate(&state) {
            return Ok(false);
        }
        if !self.session_gate(sessions_since) {
            return Ok(false);
        }
        if !self.lock_gate().await? {
            return Ok(false);
        }
        self.acquire_lock().await?;
        Ok(true)
    }

    pub async fn acquire_lock(&self) -> Result<()> {
        if let Some(p) = self.lock_file.parent() {
            fs::create_dir_all(p).await?;
        }
        fs::write(&self.lock_file, now_secs().to_string()).await?;
        Ok(())
    }

    pub async fn finish(&self) -> Result<()> {
        let state = ConsolidationState { last_consolidated_at: Some(now_secs()) };
        let json = serde_json::to_string_pretty(&state)?;
        if let Some(p) = self.state_file.parent() {
            fs::create_dir_all(p).await?;
        }
        fs::write(&self.state_file, json).await?;
        if self.lock_file.exists() {
            fs::remove_file(&self.lock_file).await?;
        }
        Ok(())
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

// ── Consolidation orchestrator ────────────────────────────────────────────────

/// Runs the 3-phase agentic consolidation, chained via `previous_response_id`
/// so phases share the upstream prompt cache. Returns the plan; the caller is
/// responsible for `apply_plan` + re-index.
pub async fn run_consolidation(
    workspace: &Path,
    llm: &RemoteLLM,
    last_consolidated_at: f64,
) -> Result<ConsolidationPlan> {
    let topics_dir = workspace.join("topics");
    let journal_dir = workspace.join("journal");
    fs::create_dir_all(&topics_dir).await?;

    let topic_summaries = list_topic_summaries(&topics_dir).await?;
    let journal_text = read_journal_since(&journal_dir, last_consolidated_at).await?;

    if journal_text.trim().is_empty() {
        return Ok(ConsolidationPlan::default());
    }

    // Phase 1 — Orient. Skim existing topic files so you improve them rather
    // than creating duplicates. (Lifted from claurst consolidation_prompt.)
    let phase1 = vec![
        Message { role: Role::System, content: DREAM_PROMPT.to_string() },
        Message {
            role: Role::User,
            content: format!(
                "## Phase 1 — Orient\n\n\
                 Existing topic files (slug — description):\n\n{}\n\n\
                 Skim them so you improve them rather than creating near-duplicates. \
                 Briefly assess: which areas are well covered, which are sparse, what gaps \
                 stand out? Plain prose, no JSON yet.",
                if topic_summaries.trim().is_empty() {
                    "(none)".to_string()
                } else {
                    topic_summaries
                }
            ),
        },
    ];
    let (_, id1) = llm.respond(&phase1, None, false).await?;

    // Phase 2 — Gather recent signal. Look for new information worth persisting,
    // and existing memories that have drifted — facts the journal now contradicts.
    let phase2 = vec![
        Message { role: Role::System, content: DREAM_PROMPT.to_string() },
        Message {
            role: Role::User,
            content: format!(
                "## Phase 2 — Gather recent signal\n\n\
                 Journal entries since the last consolidation:\n\n{}\n\n\
                 Look for:\n\
                 1. New information worth persisting: recurring topics, decisions, \
                    preferences, ongoing projects, things the user wants remembered.\n\
                 2. Existing memories that have drifted — facts in the topic summaries \
                    above that the journal now contradicts.\n\n\
                 Plain prose, no JSON yet.",
                journal_text
            ),
        },
    ];
    let (_, id2) = llm.respond(&phase2, Some(&id1), false).await?;

    // Phase 3 — Consolidate
    let phase3 = vec![
        Message { role: Role::System, content: DREAM_PROMPT.to_string() },
        Message {
            role: Role::User,
            content:
                "Phase 3 (Consolidate). Output a JSON plan to update the topic files. Schema:\n\
                {\n  \"updates\": [{\"name\": \"<kebab-case-slug>\", \"operation\": \"create|replace|append\", \"description\": \"<one-line summary>\", \"content\": \"<markdown body>\"}],\n  \"deletes\": [\"<slug>\"]\n}\n\
                Rules:\n\
                - Slugs are kebab-case, no .md extension; the slug is the filename.\n\
                - create/replace: provide complete, well-organized markdown content.\n\
                - append: provide just the new content to append.\n\
                - Only delete topics that are obsolete or strictly subsumed elsewhere.\n\
                - If nothing meaningful changed, return {\"updates\": [], \"deletes\": []}.\n\
                Output ONLY the JSON object — no preamble, no code fences.".to_string(),
        },
    ];
    let (plan_text, _) = llm.respond(&phase3, Some(&id2), false).await?;

    parse_plan(&plan_text)
}

pub fn parse_plan(text: &str) -> Result<ConsolidationPlan> {
    let s = text.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s)
        .trim();
    let s = s.strip_suffix("```").unwrap_or(s).trim();
    Ok(serde_json::from_str(s)?)
}

/// Apply a consolidation plan to `<workspace>/topics/`. Returns the number of
/// files written or removed.
pub async fn apply_plan(workspace: &Path, plan: &ConsolidationPlan) -> Result<usize> {
    let topics_dir = workspace.join("topics");
    fs::create_dir_all(&topics_dir).await?;
    let mut changed = 0usize;

    for upd in &plan.updates {
        let slug = sanitize_slug(&upd.name);
        if slug.is_empty() {
            continue;
        }
        let path = topics_dir.join(format!("{slug}.md"));
        let content = match upd.operation.as_str() {
            "create" | "replace" => format_topic(&upd.description, &upd.content),
            "append" => {
                let existing = fs::read_to_string(&path).await.unwrap_or_default();
                let sep = if existing.is_empty() || existing.ends_with("\n\n") {
                    ""
                } else if existing.ends_with('\n') {
                    "\n"
                } else {
                    "\n\n"
                };
                format!("{existing}{sep}{}\n", upd.content.trim())
            }
            other => {
                tracing::warn!("dream: unknown operation '{other}' on {slug}");
                continue;
            }
        };
        fs::write(&path, content).await?;
        changed += 1;
    }

    for name in &plan.deletes {
        let slug = sanitize_slug(name);
        if slug.is_empty() {
            continue;
        }
        let path = topics_dir.join(format!("{slug}.md"));
        if path.exists() {
            fs::remove_file(&path).await?;
            changed += 1;
        }
    }

    Ok(changed)
}

fn sanitize_slug(name: &str) -> String {
    name.trim()
        .trim_end_matches(".md")
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_'))
        .collect()
}

fn format_topic(description: &str, body: &str) -> String {
    let desc = description.trim();
    let body = body.trim();
    if desc.is_empty() {
        format!("{body}\n")
    } else {
        format!("---\ndescription: {desc}\n---\n\n{body}\n")
    }
}

async fn list_topic_summaries(topics_dir: &Path) -> Result<String> {
    if !topics_dir.exists() {
        return Ok(String::new());
    }
    let mut out = String::new();
    let mut rd = fs::read_dir(topics_dir).await?;
    while let Some(entry) = rd.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let content = fs::read_to_string(&path).await.unwrap_or_default();
        let desc = extract_description(&content).unwrap_or_else(|| {
            content
                .lines()
                .find(|l| !l.trim().is_empty() && !l.starts_with("---"))
                .unwrap_or("")
                .trim()
                .to_string()
        });
        out.push_str(&format!("- {name} — {desc}\n"));
    }
    Ok(out)
}

fn extract_description(content: &str) -> Option<String> {
    let s = content.strip_prefix("---\n")?;
    let end = s.find("\n---")?;
    let frontmatter = &s[..end];
    for line in frontmatter.lines() {
        if let Some(v) = line.strip_prefix("description:") {
            return Some(v.trim().to_string());
        }
    }
    None
}

async fn read_journal_since(journal_dir: &Path, since_ts: f64) -> Result<String> {
    if !journal_dir.exists() {
        return Ok(String::new());
    }
    let mut entries: Vec<(f64, PathBuf)> = Vec::new();
    let mut rd = fs::read_dir(journal_dir).await?;
    while let Some(e) = rd.next_entry().await? {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let mtime = e
            .metadata()
            .await?
            .modified()?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        if mtime > since_ts {
            entries.push((mtime, path));
        }
    }
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut out = String::new();
    for (_, path) in entries {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let content = fs::read_to_string(&path).await?;
        out.push_str(&format!("=== {name} ===\n{content}\n\n"));
    }
    Ok(out)
}
