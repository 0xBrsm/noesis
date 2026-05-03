use anyhow::Result;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequestArgs,
        CreateEmbeddingRequestArgs,
    },
};
use futures::StreamExt;
use std::io::Write;

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

pub trait LLM: Send + Sync {
    fn chat(
        &self,
        messages: &[Message],
    ) -> impl Future<Output = Result<String>> + Send;

    fn embed(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Vec<f32>>> + Send;
}

use std::future::Future;

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
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        Ok(content)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.embed_model)
            .input(text)
            .build()?;

        let response = self.client.embeddings().create(request).await?;
        let vec = response
            .data
            .into_iter()
            .next()
            .map(|e| e.embedding)
            .unwrap_or_default();

        Ok(vec)
    }
}

impl RemoteLLM {
    /// Stream a chat response, printing tokens as they arrive.
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

pub struct LocalLLM;

impl LLM for LocalLLM {
    async fn chat(&self, _messages: &[Message]) -> Result<String> {
        anyhow::bail!("LocalLLM not yet implemented")
    }

    async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        anyhow::bail!("LocalLLM not yet implemented")
    }
}
