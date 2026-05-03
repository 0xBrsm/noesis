mod config;
mod db;
mod llm;
mod memory;

use anyhow::Result;
use clap::Parser;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::llm::{LLM, Message, RemoteLLM, Role};
use crate::memory::Memory;

const HELP: &str = "\
Commands:
  /index   re-index memory files
  /flush   flush conversation to memory now
  /clear   clear conversation history
  /help    show this message
  /quit    flush and exit  (Ctrl-D also works)";

#[tokio::main]
async fn main() -> Result<()> {
    let cli = config::Cli::parse();
    let cfg = config::load(&cli)?;

    let llm = RemoteLLM::new(
        &cfg.base_url,
        &cfg.api_key,
        &cfg.chat_model,
        &cfg.embed_model,
    );

    let mut mem = Memory::open(
        &cfg.workspace,
        &cfg.embed_model,
        cfg.chunk_tokens,
        cfg.chunk_overlap,
    )?;

    // Initial index
    print!("indexing memory... ");
    let idx = mem.index(&llm).await?;
    println!("{} indexed, {} skipped", idx.indexed, idx.skipped);

    println!("model: {}  workspace: {}", cfg.chat_model, cfg.workspace.display());
    println!("type /help for commands\n");

    let mut editor = DefaultEditor::new()?;
    let history_path = cfg.workspace.join(".history");
    let _ = editor.load_history(&history_path);

    let mut conversation: Vec<Message> = Vec::new();
    let mut turn: usize = 0;

    loop {
        let line = match editor.readline("you: ") {
            Ok(l) => l,
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
                println!();
                break;
            }
            Err(e) => return Err(e.into()),
        };

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(&input);

        // Slash commands
        match input.as_str() {
            "/quit" | "/exit" => break,
            "/help" => { println!("{HELP}"); continue; }
            "/clear" => {
                conversation.clear();
                turn = 0;
                println!("conversation cleared");
                continue;
            }
            "/index" => {
                print!("indexing... ");
                let r = mem.index(&llm).await?;
                println!("{} indexed, {} skipped, {} deleted", r.indexed, r.skipped, r.deleted);
                continue;
            }
            "/flush" => {
                flush_conversation(&mut mem, &conversation, &llm).await;
                continue;
            }
            _ => {}
        }

        // Search memory for relevant context
        let snippets = mem.search(&input, cfg.search_limit, &llm).await
            .unwrap_or_default();

        // Build system prompt with injected memory
        let system_content = build_system_prompt(&snippets);

        // Assemble messages for this turn
        let mut messages = vec![Message { role: Role::System, content: system_content }];
        messages.extend_from_slice(&conversation);
        messages.push(Message { role: Role::User, content: input.clone() });

        // Stream response
        print!("assistant: ");
        let response = match llm.chat_stream(&messages).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                continue;
            }
        };

        // Update conversation history (without system message)
        conversation.push(Message { role: Role::User,      content: input    });
        conversation.push(Message { role: Role::Assistant, content: response });
        turn += 1;

        // Periodic flush
        if turn % cfg.flush_every == 0 {
            flush_conversation(&mut mem, &conversation, &llm).await;
        }
    }

    // Final flush on exit
    if !conversation.is_empty() {
        flush_conversation(&mut mem, &conversation, &llm).await;
    }

    let _ = editor.save_history(&history_path);
    println!("bye");
    Ok(())
}

fn build_system_prompt(snippets: &[crate::db::SearchRow]) -> String {
    let base = "You are a helpful assistant.";
    if snippets.is_empty() {
        return base.to_string();
    }
    let context = snippets
        .iter()
        .map(|r| {
            let snippet = r.text.trim();
            let short = if snippet.len() > 300 { &snippet[..300] } else { snippet };
            format!("- {short}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("{base}\n\n## Memory\n{context}")
}

async fn flush_conversation(
    mem: &mut Memory,
    conversation: &[Message],
    llm: &RemoteLLM,
) {
    if conversation.is_empty() {
        return;
    }
    print!("flushing memory... ");
    match mem.flush(conversation, llm).await {
        Ok(Some(_)) => println!("saved"),
        Ok(None)    => println!("nothing to save"),
        Err(e)      => eprintln!("flush error: {e}"),
    }
}
