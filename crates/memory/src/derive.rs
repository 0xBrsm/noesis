//! Phased derivation. Turns a session transcript into titled conversations,
//! each with a summary and extracted facts.
//!
//! Two phases, because one call cannot do both jobs. A single call asked for
//! boundaries *and* per-conversation depth spends one output budget on both, so
//! the two compete and the model buys depth by returning fewer conversations —
//! a whole session collapsing into two or three enormous blocks. Splitting the
//! call is the fix; wording is not. Phase 1 emits only ranges and titles, so
//! nothing competes with the count.
//!
//! Phase 2 runs standalone per conversation rather than branching off phase 1
//! with `previous_response_id`. Branching would avoid re-sending the transcript,
//! but only pays off when every branch matches the root's `instructions`,
//! reasoning effort *and* response schema — miss any one and the whole inherited
//! prefix misses cache. Chat sessions are small enough that re-sending one
//! conversation's own messages is cheaper than honouring that constraint.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::db::SessionMessage;
use crate::llm::RemoteLLM;

/// What the model was asked. Bump when a prompt or either schema changes:
/// banked digests answered a different question, so they cannot be replayed and
/// every session needs fresh calls. This is the expensive one.
pub const PROMPT_FORMAT: u32 = 1;

/// What gets written from the model's answer. Bump when the markdown layout
/// changes: banked digests are still valid answers, so `--rerender` replays them
/// for free.
pub const RENDER_FORMAT: u32 = 1;

const MAX_ATTEMPTS: usize = 3;

pub const DERIVE_PROMPT: &str = "\
You convert a transcript of a conversation between a user and an AI assistant \
into faithful retrieval material.

Preserve chronology. Distinguish what was decided from what was merely \
considered, and what was established from what was speculated. Never invent \
detail that is not in the transcript. Prefer concrete, searchable phrasing over \
generic description: name the files, systems, commands, errors, and people that \
actually appear.";

// ── Phase 1: split ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRange {
    pub start_message: usize,
    pub end_message: usize,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SplitResult {
    conversations: Vec<ConversationRange>,
}

/// Strict structured output accepts only a subset of JSON Schema — `minimum`
/// and `minItems` among the keywords it rejects outright — so bounds are absent
/// here on purpose. `validate_split` enforces them instead, which it has to do
/// regardless: no schema can express "these ranges tile the transcript".
fn split_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "conversations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_message": {"type": "integer"},
                        "end_message": {"type": "integer"},
                        "title": {"type": "string"}
                    },
                    "required": ["start_message", "end_message", "title"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["conversations"],
        "additionalProperties": false
    })
}

/// Every message must land in exactly one conversation, and the conversations
/// must run in order with no gap. A model that fails this has not understood the
/// task, and the failure is worth a retry rather than a silent hole in memory.
pub fn validate_split(ranges: &[ConversationRange], message_count: usize) -> Result<()> {
    if message_count == 0 {
        bail!("no messages to split");
    }
    if ranges.is_empty() {
        bail!("no conversations returned");
    }
    let mut expected = 1usize;
    for (i, r) in ranges.iter().enumerate() {
        if r.start_message != expected {
            bail!(
                "conversation {} starts at message {}, expected {}",
                i + 1,
                r.start_message,
                expected
            );
        }
        if r.end_message < r.start_message {
            bail!(
                "conversation {} ends at message {}, before its start at {}",
                i + 1,
                r.end_message,
                r.start_message
            );
        }
        if r.end_message > message_count {
            bail!(
                "conversation {} ends at message {}, past the last message {}",
                i + 1,
                r.end_message,
                message_count
            );
        }
        expected = r.end_message + 1;
    }
    if expected != message_count + 1 {
        bail!(
            "conversations stop at message {}, but the transcript has {}",
            expected - 1,
            message_count
        );
    }
    Ok(())
}

async fn split_session(
    llm: &RemoteLLM,
    instructions: &str,
    messages: &[SessionMessage],
) -> Result<Vec<ConversationRange>> {
    let transcript = numbered_transcript(messages, 1);
    let base = format!(
        "Split this transcript into consecutive conversations. Return ranges that \
         cover messages 1 through {} exactly once, in order, with no gaps.\n\n\
         Start a new conversation when the subject, task, or goal changes. Judge \
         that on content: a conversation ends where the thing being discussed \
         ends, however many messages that took. Do not merge unrelated work \
         because it is adjacent, and do not split a single continuous thread \
         because it is long.\n\n\
         Give each conversation a specific title naming what it was about.\n\n{}",
        messages.len(),
        transcript
    );

    let mut prompt = base;
    let mut last_err = None;
    for attempt in 1..=MAX_ATTEMPTS {
        let (raw, _) = match llm
            .respond_json(instructions, &prompt, "conversation_split", split_schema())
            .await
        {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(e.to_string());
                continue;
            }
        };
        let parsed: SplitResult = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                last_err = Some(format!("decode split: {e}"));
                continue;
            }
        };
        match validate_split(&parsed.conversations, messages.len()) {
            Ok(()) => return Ok(parsed.conversations),
            Err(e) => {
                tracing::warn!(attempt, "split failed validation: {e}");
                prompt = format!(
                    "{prompt}\n\nYour previous response failed validation: {e}. \
                     Regenerate the complete result."
                );
                last_err = Some(e.to_string());
            }
        }
    }
    bail!(
        "split failed after {MAX_ATTEMPTS} attempts: {}",
        last_err.unwrap_or_else(|| "unknown".into())
    )
}

// ── Phase 2: summary and facts ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Detail {
    summary: String,
    facts: Vec<String>,
}

fn detail_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "facts": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["summary", "facts"],
        "additionalProperties": false
    })
}

async fn detail_conversation(
    llm: &RemoteLLM,
    instructions: &str,
    title: &str,
    messages: &[SessionMessage],
    first_index: usize,
) -> Result<Detail> {
    let transcript = numbered_transcript(messages, first_index);
    let prompt = format!(
        "This is one conversation, titled \"{title}\".\n\n\
         Write a summary of it: what was discussed, what was decided, what was \
         left unresolved. Then extract the durable facts worth remembering in \
         later, unrelated sessions — decisions, preferences, constraints, \
         configuration, and the state of ongoing work. One fact per entry, each \
         standing on its own without the summary for context. Omit anything that \
         was transient, superseded within this conversation, or true only while \
         it was being discussed. An empty fact list is a valid answer.\n\n{transcript}"
    );

    let (raw, _) = llm
        .respond_json(instructions, &prompt, "conversation_detail", detail_schema())
        .await?;
    Ok(serde_json::from_str(&raw)?)
}

// ── Digest ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedConversation {
    pub start_message: usize,
    pub end_message: usize,
    pub title: String,
    pub summary: String,
    pub facts: Vec<String>,
}

/// The banked answer for one session. Serialized into `derivations.digest`, so
/// a render change replays it without paying for the calls again.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDigest {
    pub session_id: String,
    pub prompt_format: u32,
    pub conversations: Vec<DerivedConversation>,
}

/// Run both phases over one session. Callers bank the result and render it;
/// this function makes no database or filesystem writes of its own.
pub async fn derive_session(
    llm: &RemoteLLM,
    instructions: &str,
    session_id: &str,
    messages: &[SessionMessage],
) -> Result<SessionDigest> {
    let ranges = split_session(llm, instructions, messages).await?;
    tracing::info!(
        session = session_id,
        messages = messages.len(),
        conversations = ranges.len(),
        "derive: split"
    );

    let mut conversations = Vec::with_capacity(ranges.len());
    for range in ranges {
        let slice = &messages[range.start_message - 1..range.end_message];
        let detail =
            detail_conversation(llm, instructions, &range.title, slice, range.start_message)
                .await?;
        conversations.push(DerivedConversation {
            start_message: range.start_message,
            end_message: range.end_message,
            title: range.title,
            summary: detail.summary,
            facts: detail.facts,
        });
    }

    Ok(SessionDigest {
        session_id: session_id.to_string(),
        prompt_format: PROMPT_FORMAT,
        conversations,
    })
}

// ── Render ────────────────────────────────────────────────────────────────────

/// One markdown file per session, one `##` section per conversation, so the
/// existing header chunker makes each conversation its own retrieval chunk.
pub fn render_digest(digest: &SessionDigest) -> String {
    let mut out = format!("# Session {}\n", digest.session_id);
    for c in &digest.conversations {
        out.push_str(&format!("\n## {}\n\n{}\n", c.title, c.summary.trim()));
        if !c.facts.is_empty() {
            out.push_str("\nFacts:\n");
            for fact in &c.facts {
                out.push_str(&format!("- {}\n", fact.trim()));
            }
        }
    }
    out
}

// ── Transcript formatting ─────────────────────────────────────────────────────

/// `first_index` is the message's number in the whole session, so a phase-2
/// slice carries the same numbers phase 1 assigned it.
fn numbered_transcript(messages: &[SessionMessage], first_index: usize) -> String {
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| format!("[{}] {}: {}", first_index + i, m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(start: usize, end: usize) -> ConversationRange {
        ConversationRange {
            start_message: start,
            end_message: end,
            title: format!("c{start}"),
        }
    }

    #[test]
    fn contiguous_cover_is_valid() {
        assert!(validate_split(&[range(1, 3), range(4, 10)], 10).is_ok());
    }

    #[test]
    fn single_conversation_covering_everything_is_valid() {
        assert!(validate_split(&[range(1, 1)], 1).is_ok());
    }

    #[test]
    fn rejects_gap() {
        let err = validate_split(&[range(1, 3), range(5, 10)], 10).unwrap_err();
        assert!(err.to_string().contains("expected 4"), "{err}");
    }

    #[test]
    fn rejects_overlap() {
        let err = validate_split(&[range(1, 5), range(4, 10)], 10).unwrap_err();
        assert!(err.to_string().contains("expected 6"), "{err}");
    }

    #[test]
    fn rejects_not_starting_at_one() {
        assert!(validate_split(&[range(2, 10)], 10).is_err());
    }

    #[test]
    fn rejects_short_cover() {
        let err = validate_split(&[range(1, 8)], 10).unwrap_err();
        assert!(err.to_string().contains("stop at message 8"), "{err}");
    }

    #[test]
    fn rejects_past_the_end() {
        let err = validate_split(&[range(1, 12)], 10).unwrap_err();
        assert!(err.to_string().contains("past the last message"), "{err}");
    }

    #[test]
    fn rejects_inverted_range() {
        assert!(validate_split(&[range(1, 3), ConversationRange {
            start_message: 4,
            end_message: 2,
            title: "bad".into(),
        }], 10).is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(validate_split(&[], 10).is_err());
        assert!(validate_split(&[range(1, 1)], 0).is_err());
    }

    #[test]
    fn transcript_numbering_is_absolute() {
        let messages = vec![
            SessionMessage { role: "user".into(), content: "a".into(), ts: 0.0 },
            SessionMessage { role: "assistant".into(), content: "b".into(), ts: 1.0 },
        ];
        assert_eq!(numbered_transcript(&messages, 7), "[7] user: a\n\n[8] assistant: b");
    }

    #[test]
    fn render_puts_each_conversation_under_its_own_header() {
        let digest = SessionDigest {
            session_id: "s1".into(),
            prompt_format: PROMPT_FORMAT,
            conversations: vec![
                DerivedConversation {
                    start_message: 1,
                    end_message: 2,
                    title: "Fixing the index".into(),
                    summary: "Added an index.".into(),
                    facts: vec!["Retrievals are indexed by session.".into()],
                },
                DerivedConversation {
                    start_message: 3,
                    end_message: 4,
                    title: "Naming".into(),
                    summary: "Renamed a table.".into(),
                    facts: vec![],
                },
            ],
        };
        let md = render_digest(&digest);
        assert!(md.starts_with("# Session s1\n"));
        assert_eq!(md.matches("\n## ").count(), 2);
        assert!(md.contains("- Retrievals are indexed by session."));
        // A conversation with no facts gets no empty Facts heading.
        assert_eq!(md.matches("Facts:").count(), 1);
    }
}
