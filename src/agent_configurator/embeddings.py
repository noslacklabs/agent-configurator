"""LangChain embeddings backed by ``POST /llm/{config_name}/embeddings``."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from agent_configurator.client import ConfiguratorClient


class ConfiguredEmbeddings(Embeddings):
    def __init__(
        self,
        config_name: str,
        base_url: str | None = None,
        agent_id: str | None = None,
        batch_size: int = 64,
        timeout: float = 120.0,
    ):
        self.config_name = config_name
        self.base_url = base_url
        self.agent_id = agent_id
        self.batch_size = max(1, batch_size)
        self.timeout = timeout

    def _client(self) -> ConfiguratorClient:
        return ConfiguratorClient(self.base_url, timeout=self.timeout)

    def _batches(self, texts: list[str]) -> list[list[str]]:
        return [texts[start:start + self.batch_size] for start in range(0, len(texts), self.batch_size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        vectors: list[list[float]] = []
        for batch in self._batches(list(texts)):
            vectors.extend(client.embeddings(self.config_name, batch, self.agent_id))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        vectors: list[list[float]] = []
        for batch in self._batches(list(texts)):
            vectors.extend(await client.aembeddings(self.config_name, batch, self.agent_id))
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]
