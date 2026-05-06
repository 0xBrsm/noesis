use anyhow::Result;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequestArgs,
        CreateEmbeddingRequestArgs,
    },
};
use fastembed::{
    EmbeddingModel, InitOptionsWithLength, OnnxSource, RerankInitOptionsUserDefined,
    TextEmbedding, TextRerank, TokenizerFiles, UserDefinedRerankingModel,
};
use futures::StreamExt;
use std::future::Future;
use std::io::Write;
use std::sync::Arc;

// ── Message types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl From<&Message> for ChatCompletionRequestMessage {
    fn from(m: &Message) -> Self {
        use async_openai::types::{
            ChatCompletionRequestAssistantMessageArgs,
            ChatCompletionRequestSystemMessageArgs,
            ChatCompletionRequestUserMessageArgs,
        };
        match m.role {
            Role::System => ChatCompletionRequestSystemMessageArgs::default()
                .content(m.content.clone())
                .build()
                .unwrap()
                .into(),
            Role::User => ChatCompletionRequestUserMessageArgs::default()
                .content(m.content.clone())
                .build()
                .unwrap()
                .into(),
            Role::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
                .content(m.content.clone())
                .build()
                .unwrap()
                .into(),
        }
    }
}

// ── LLM trait ─────────────────────────────────────────────────────────────────

pub trait LLM: Send + Sync {
    fn chat(&self, messages: &[Message]) -> impl Future<Output = Result<String>> + Send;
    fn embed(&self, text: &str) -> impl Future<Output = Result<Vec<f32>>> + Send;
}

// ── RemoteLLM ─────────────────────────────────────────────────────────────────

pub struct RemoteLLM {
    client: Client<OpenAIConfig>,
    pub chat_model: String,
    pub embed_model: String,
}

impl RemoteLLM {
    pub fn new(base_url: &str, api_key: &str, chat_model: &str, embed_model: &str) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key(api_key);
        Self {
            client: Client::with_config(config),
            chat_model: chat_model.to_string(),
            embed_model: embed_model.to_string(),
        }
    }

    /// Stream a chat response, printing tokens to stdout as they arrive.
    /// Returns the full assembled response string.
    pub async fn chat_stream(&self, messages: &[Message]) -> Result<String> {
        let msgs: Vec<ChatCompletionRequestMessage> =
            messages.iter().map(|m| m.into()).collect();

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.chat_model)
            .messages(msgs)
            .stream(true)
            .build()?;

        let mut stream = self.client.chat().create_stream(request).await?;
        let mut full = String::new();

        while let Some(chunk) = stream.next().await {
            for choice in chunk?.choices {
                if let Some(content) = choice.delta.content {
                    print!("{content}");
                    std::io::stdout().flush()?;
                    full.push_str(&content);
                }
            }
        }
        println!();
        Ok(full)
    }
}

impl LLM for RemoteLLM {
    async fn chat(&self, messages: &[Message]) -> Result<String> {
        let msgs: Vec<ChatCompletionRequestMessage> =
            messages.iter().map(|m| m.into()).collect();
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.chat_model)
            .messages(msgs)
            .build()?;
        let response = self.client.chat().create(request).await?;
        Ok(response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.embed_model)
            .input(text)
            .build()?;
        let response = self.client.embeddings().create(request).await?;
        Ok(response
            .data
            .into_iter()
            .next()
            .map(|e| e.embedding)
            .unwrap_or_default())
    }
}

// ── LocalLLM ──────────────────────────────────────────────────────────────────

pub struct LocalLLM {
    embedder: Arc<std::sync::Mutex<TextEmbedding>>,
}

impl LocalLLM {
    pub fn new() -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let cache_dir = std::path::PathBuf::from(home).join(".cache").join("fastembed");
        std::fs::create_dir_all(&cache_dir)?;

        let model = TextEmbedding::try_new(
            InitOptionsWithLength::new(EmbeddingModel::AllMiniLML6V2Q)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )?;
        Ok(Self { embedder: Arc::new(std::sync::Mutex::new(model)) })
    }
}

impl LLM for LocalLLM {
    async fn chat(&self, _messages: &[Message]) -> Result<String> {
        anyhow::bail!("LocalLLM chat not yet implemented")
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.to_string();
        let embedder = self.embedder.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = embedder.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
            let mut results = model.embed(vec![text], None)?;
            Ok(results.remove(0))
        })
        .await?
    }
}

// ── Embedder enum (unified dispatch) ──────────────────────────────────────────

/// Dispatches LLM calls to either a remote API or local fastembed/llama.cpp.
/// Use this in main to avoid generics proliferation.
pub enum Embedder {
    Remote(RemoteLLM),
    Local(LocalLLM),
}

impl LLM for Embedder {
    async fn chat(&self, messages: &[Message]) -> Result<String> {
        match self {
            Self::Remote(r) => r.chat(messages).await,
            Self::Local(l) => l.chat(messages).await,
        }
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Remote(r) => r.embed(text).await,
            Self::Local(l) => l.embed(text).await,
        }
    }
}

// ── Reranker ──────────────────────────────────────────────────────────────────

const RERANKER_REPO: &str = "Xenova/ms-marco-MiniLM-L-6-v2";
const RERANKER_ONNX: &str = "onnx/model_int8.onnx";

pub struct Reranker {
    inner: Arc<std::sync::Mutex<TextRerank>>,
}

impl Reranker {
    pub fn new() -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let cache_dir = std::path::PathBuf::from(&home).join(".cache").join("fastembed");
        std::fs::create_dir_all(&cache_dir)?;

        let model_dir = download_reranker(&cache_dir)?;
        let model = load_reranker(&model_dir)?;
        Ok(Self { inner: Arc::new(std::sync::Mutex::new(model)) })
    }

    /// Rerank `docs` against `query`; returns `(original_index, reranker_score)` sorted best-first.
    pub fn rerank(&self, query: &str, docs: &[String]) -> Result<Vec<(usize, f32)>> {
        let query = query.to_string();
        let docs = docs.to_vec();
        let inner = self.inner.clone();
        let results = tokio::task::block_in_place(|| {
            let mut model = inner.lock().map_err(|e| anyhow::anyhow!("lock: {e}"))?;
            model.rerank(query, docs, false, None).map_err(anyhow::Error::from)
        })?;
        Ok(results.into_iter().map(|r| (r.index, r.score)).collect())
    }
}

fn download_reranker(cache_dir: &std::path::Path) -> Result<std::path::PathBuf> {
    use hf_hub::{Cache, api::sync::ApiBuilder};

    let cache = Cache::new(cache_dir.to_path_buf());
    let api = ApiBuilder::from_cache(cache).with_progress(true).build()
        .map_err(|e| anyhow::anyhow!("hf-hub init: {e}"))?;
    let repo = api.model(RERANKER_REPO.to_string());

    // Download required files (hf-hub caches them automatically)
    for file in &[RERANKER_ONNX, "tokenizer.json", "config.json",
                  "special_tokens_map.json", "tokenizer_config.json"] {
        repo.get(file).map_err(|e| anyhow::anyhow!("download {file}: {e}"))?;
    }

    // Return the snapshot dir so we can read files by name
    let onnx_path = repo.get(RERANKER_ONNX)
        .map_err(|e| anyhow::anyhow!("locate onnx: {e}"))?;
    // snapshot dir is two levels up from onnx/model_int8.onnx
    Ok(onnx_path.parent().unwrap().parent().unwrap().to_path_buf())
}

fn load_reranker(model_dir: &std::path::Path) -> Result<TextRerank> {
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(model_dir.join("tokenizer.json"))?,
        config_file: std::fs::read(model_dir.join("config.json"))?,
        special_tokens_map_file: std::fs::read(model_dir.join("special_tokens_map.json"))?,
        tokenizer_config_file: std::fs::read(model_dir.join("tokenizer_config.json"))?,
    };

    let onnx_bytes = std::fs::read(model_dir.join("onnx").join("model_int8.onnx"))?;

    let user_model = UserDefinedRerankingModel::new(
        OnnxSource::Memory(onnx_bytes),
        tokenizer_files,
    );

    TextRerank::try_new_from_user_defined(user_model, RerankInitOptionsUserDefined::default())
        .map_err(anyhow::Error::from)
}
