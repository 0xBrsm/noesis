//! noesis-proxy — OpenAI Responses API proxy with RAG memory injection.
//!
//! Skeleton: terminates HTTPS on a local port, forwards to upstream, and (eventually)
//! injects retrieved memory into the request before forwarding. For now it just
//! starts an axum server with a single placeholder route.

use anyhow::Result;
use axum::{Router, routing::post};
use clap::Parser;
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "noesis-proxy", about = "OpenAI Responses API proxy with RAG memory injection")]
struct Cli {
    /// Address to bind the proxy on
    #[arg(long, default_value = "127.0.0.1:8787")]
    bind: SocketAddr,

    /// Upstream Responses API base URL (e.g. https://api.openai.com)
    #[arg(long, env = "NOESIS_UPSTREAM", default_value = "https://api.openai.com")]
    upstream: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let cli = Cli::parse();
    let app = Router::new().route("/v1/responses", post(responses_stub));

    tracing::info!(bind = %cli.bind, upstream = %cli.upstream, "noesis-proxy starting");
    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn responses_stub() -> &'static str {
    "noesis-proxy: /v1/responses not yet implemented\n"
}
