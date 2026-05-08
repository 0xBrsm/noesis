use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use std::io::stdout;

use crate::config::Config;
use crate::db::SearchRow;
use crate::dream::AutoDream;
use crate::llm::{LLM, Message, RemoteLLM, Reranker, Role};
use crate::memory::Memory;

// ── Slash command helpers (verbatim from claurst/src-rust/crates/tui/src/input.rs) ──

fn is_slash_command(input: &str) -> bool {
    input.starts_with('/') && !input.starts_with("//")
}

fn parse_slash_command(input: &str) -> (&str, &str) {
    if !is_slash_command(input) {
        return ("", "");
    }
    let s = &input[1..];
    match s.find(' ') {
        Some(i) => (&s[..i], s[i + 1..].trim()),
        None => (s, ""),
    }
}

// ── App state ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ChatMsg {
    role: &'static str, // "you" | "assistant" | "system"
    content: String,
}

struct App {
    messages: Vec<ChatMsg>,
    input: String,
    cursor: usize,
    scroll: u16,
    waiting: bool,
    status: Option<String>, // transient notification
}

impl App {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            waiting: false,
            status: None,
        }
    }

    fn push(&mut self, role: &'static str, content: String) {
        self.messages.push(ChatMsg { role, content });
        self.scroll = 0; // auto-scroll to bottom
    }

    fn insert(&mut self, c: char) {
        let pos = char_to_byte(&self.input, self.cursor);
        self.input.insert(pos, c);
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let pos = char_to_byte(&self.input, self.cursor);
            self.input.remove(pos);
        }
    }

    fn take_input(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.input)
    }
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn run<E: LLM>(
    cfg: &Config,
    embedder: &E,
    reranker: Option<&Reranker>,
    chat_llm: &RemoteLLM,
    mem: &mut Memory,
) -> Result<()> {
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    let result = chat_loop(cfg, embedder, reranker, chat_llm, mem, &mut terminal).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    result
}

async fn chat_loop<E: LLM>(
    cfg: &Config,
    embedder: &E,
    reranker: Option<&Reranker>,
    chat_llm: &RemoteLLM,
    mem: &mut Memory,
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    let mut app = App::new();
    let context_md = crate::memory::load_context_md(&cfg.workspace);
    let dream = AutoDream::new(&cfg.workspace);

    // Load prior history for display
    for (role, content) in mem.load_recent_turns()? {
        let r: &'static str = match role.as_str() {
            "assistant" => "assistant",
            _ => "you",
        };
        app.messages.push(ChatMsg { role: r, content });
    }

    let mut previous_response_id = mem.last_response_id()?;
    let session_id = uuid::Uuid::new_v4().to_string();
    let mut turn_index = 0usize;

    // History vec for LLM (mirrors app.messages excluding current input)
    let mut history: Vec<Message> = mem.load_recent_turns()?
        .into_iter()
        .map(|(role, content)| Message {
            role: match role.as_str() {
                "assistant" => Role::Assistant,
                _ => Role::User,
            },
            content,
        })
        .collect();

    let mut events = EventStream::new();

    loop {
        terminal.draw(|f| render(f, &app, cfg))?;

        let Some(Ok(event)) = events.next().await else { break };

        match event {
            Event::Key(key) => {
                if app.waiting {
                    continue;
                }
                match (key.modifiers, key.code) {
                    (KeyModifiers::CONTROL, KeyCode::Char('c')) => break,
                    (KeyModifiers::CONTROL, KeyCode::Char('d')) if app.input.is_empty() => break,
                    (_, KeyCode::Enter) => {
                        let raw = app.take_input();
                        if raw.is_empty() {
                            continue;
                        }
                        app.status = None;

                        if is_slash_command(&raw) {
                            let msg = handle_slash(
                                &raw, cfg, mem, embedder, reranker, &dream,
                            ).await?;
                            app.push("system", msg);
                            continue;
                        }

                        // Regular chat turn
                        app.push("you", raw.clone());
                        app.waiting = true;
                        terminal.draw(|f| render(f, &app, cfg))?;

                        match do_turn(
                            &raw, &history, cfg, embedder, reranker,
                            chat_llm, mem, context_md.as_deref(),
                            previous_response_id.as_deref(), cfg.debug,
                        ).await {
                            Ok((response, response_id)) => {
                                app.push("assistant", response.clone());
                                mem.insert_turn(&session_id, turn_index,     "user",      &raw,      None)?;
                                mem.insert_turn(&session_id, turn_index + 1, "assistant", &response, Some(&response_id))?;
                                turn_index += 2;
                                previous_response_id = Some(response_id);
                                history.push(Message { role: Role::User,      content: raw });
                                history.push(Message { role: Role::Assistant, content: response.clone() });

                                // Dream trigger check
                                let dream_state = dream.load_state().await;
                                let sessions = mem.sessions_since(dream_state.last_as_ts())?;
                                if dream.should_run(sessions).await? {
                                    app.status = Some("dreaming…".into());
                                    terminal.draw(|f| render(f, &app, cfg))?;
                                    let recent = recent_turns(&history, 40);
                                    mem.extract_to_memory(&recent, chat_llm).await?;
                                    dream.finish().await?;
                                    app.status = None;
                                }
                            }
                            Err(e) => {
                                app.push("system", format!("error: {e}"));
                            }
                        }
                        app.waiting = false;
                    }
                    (_, KeyCode::Backspace) => app.backspace(),
                    (_, KeyCode::Left)  => { if app.cursor > 0 { app.cursor -= 1; } }
                    (_, KeyCode::Right) => {
                        if app.cursor < app.input.chars().count() {
                            app.cursor += 1;
                        }
                    }
                    (_, KeyCode::Home) => app.cursor = 0,
                    (_, KeyCode::End)  => app.cursor = app.input.chars().count(),
                    (_, KeyCode::PageUp)   => app.scroll = app.scroll.saturating_add(5),
                    (_, KeyCode::PageDown) => app.scroll = app.scroll.saturating_sub(5),
                    (_, KeyCode::Char(c))  => app.insert(c),
                    _ => {}
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    Ok(())
}

// ── Turn execution ────────────────────────────────────────────────────────────

async fn do_turn<E: LLM>(
    input: &str,
    history: &[Message],
    cfg: &Config,
    embedder: &E,
    reranker: Option<&Reranker>,
    chat_llm: &RemoteLLM,
    mem: &Memory,
    context_md: Option<&str>,
    previous_response_id: Option<&str>,
    debug: bool,
) -> Result<(String, String)> {
    let chunks = mem.search_for_context(
        input, embedder, reranker,
        cfg.retrieval_candidates,
        cfg.retrieval_limit,
        cfg.retrieval_threshold,
    ).await?;

    let system = build_system_prompt(&chunks, context_md);
    let mut messages = vec![Message { role: Role::System, content: system }];
    messages.extend_from_slice(history);
    messages.push(Message { role: Role::User, content: input.to_string() });

    chat_llm.respond(&messages, previous_response_id, debug).await
}

// ── Slash command handler ─────────────────────────────────────────────────────

async fn handle_slash<E: LLM>(
    raw: &str,
    cfg: &Config,
    mem: &mut Memory,
    embedder: &E,
    reranker: Option<&Reranker>,
    _dream: &AutoDream,
) -> Result<String> {
    let (cmd, args) = parse_slash_command(raw);
    match cmd {
        "help" => Ok(
            "/search <query>  search memory\n\
             /index           re-index memory files\n\
             /dream           run dream extraction now\n\
             /stats           show memory stats\n\
             /clear           clear message display\n\
             /help            show this help"
                .into(),
        ),
        "search" => {
            if args.is_empty() {
                return Ok("usage: /search <query>".into());
            }
            let results = mem.search(args, cfg.search_results, embedder, reranker).await?;
            if results.is_empty() {
                return Ok("no results".into());
            }
            let lines: Vec<String> = results.iter().enumerate().map(|(i, r)| {
                format!("[{}] {:.3}  {}:{}-{}\n  {}",
                    i + 1, r.score, r.path, r.start_line, r.end_line,
                    r.text.trim().chars().take(120).collect::<String>())
            }).collect();
            Ok(lines.join("\n\n"))
        }
        "index" => {
            let r = mem.index(embedder).await?;
            Ok(format!("indexed: {}  skipped: {}  deleted: {}", r.indexed, r.skipped, r.deleted))
        }
        "dream" => {
            // TODO: needs chat_llm — placeholder for now
            Ok("dream requires LLM access — use /dream inside chat".into())
        }
        "stats" => {
            Ok(format!("workspace: {}", cfg.workspace.display()))
        }
        _ => Ok(format!("unknown command: /{cmd}  (type /help)")),
    }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

fn render(f: &mut Frame, app: &App, cfg: &Config) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // status bar
            Constraint::Min(3),      // messages
            Constraint::Length(3),   // input
        ])
        .split(f.area());

    // Status bar
    let status_text = if app.waiting {
        "  thinking…".to_string()
    } else if let Some(ref s) = app.status {
        format!("  {s}")
    } else {
        format!("  noesis  {}  {}", cfg.chat_model, cfg.workspace.display())
    };
    let status = Paragraph::new(status_text)
        .style(Style::default().bg(Color::DarkGray).fg(Color::White));
    f.render_widget(status, chunks[0]);

    // Message area
    let lines = build_lines(&app.messages);
    let total = lines.len() as u16;
    let visible = chunks[1].height;
    let scroll_top = if total > visible {
        (total - visible).saturating_sub(app.scroll)
    } else {
        0
    };
    let messages = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll_top, 0));
    f.render_widget(messages, chunks[1]);

    // Input box
    let input_display = format!("> {}", app.input);
    let input = Paragraph::new(input_display.clone())
        .block(Block::default().borders(Borders::TOP));
    f.render_widget(input, chunks[2]);

    // Cursor
    let cursor_x = chunks[2].x + 2 + app.cursor as u16;
    let cursor_y = chunks[2].y + 1; // +1 for border
    f.set_cursor_position((cursor_x, cursor_y));
}

fn build_lines(messages: &[ChatMsg]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for msg in messages {
        let (label, color) = match msg.role {
            "you"       => ("You", Color::Cyan),
            "assistant" => ("Assistant", Color::Green),
            _           => ("System", Color::Yellow),
        };
        lines.push(Line::from(Span::styled(
            label,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        for line in msg.content.lines() {
            lines.push(Line::from(format!("  {line}")));
        }
        lines.push(Line::from(""));
    }
    lines
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_system_prompt(chunks: &[SearchRow], context_md: Option<&str>) -> String {
    let mut s = match context_md {
        Some(ctx) if !ctx.trim().is_empty() => format!("{}\n", ctx.trim()),
        _ => "You are a helpful assistant with access to the user's memory store.\n".into(),
    };
    if chunks.is_empty() {
        return s;
    }
    s.push_str("\nRelevant memory:\n");
    for c in chunks {
        s.push_str(&format!("\n---\nSource: {} (lines {}-{})\n{}\n",
            c.path, c.start_line, c.end_line, c.text.trim()));
    }
    s.push_str("---\n");
    s
}

fn recent_turns(history: &[Message], n: usize) -> Vec<(String, String)> {
    history.iter().rev().take(n).rev().map(|m| {
        let role = match m.role {
            Role::Assistant => "assistant",
            Role::User      => "user",
            Role::System    => "system",
        };
        (role.to_string(), m.content.clone())
    }).collect()
}
