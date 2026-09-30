"""In-memory fakes for Redis, the LLM and the vector store (no network)."""

from __future__ import annotations

import time
from types import SimpleNamespace


class FakeRedis:
    """Async subset of redis.asyncio used by the API (ping, quotas, task map)."""

    def __init__(self, healthy: bool = True):
        self.healthy = healthy
        self.data: dict[str, int | str] = {}
        self.expiry: dict[str, float] = {}

    def _check(self):
        if not self.healthy:
            raise ConnectionError("redis down")

    def _purge(self, key):
        if key in self.expiry and self.expiry[key] <= time.time():
            self.data.pop(key, None)
            self.expiry.pop(key, None)

    async def ping(self):
        self._check()
        return True

    async def incr(self, key):
        return await self.incrby(key, 1)

    async def incrby(self, key, amount):
        self._check()
        self._purge(key)
        self.data[key] = int(self.data.get(key, 0)) + int(amount)
        return self.data[key]

    async def expire(self, key, seconds):
        self._check()
        self.expiry[key] = time.time() + seconds
        return True

    async def ttl(self, key):
        self._check()
        if key not in self.data:
            return -2
        if key not in self.expiry:
            return -1
        return max(0, int(self.expiry[key] - time.time()))

    async def get(self, key):
        self._check()
        self._purge(key)
        value = self.data.get(key)
        return None if value is None else str(value).encode()

    async def set(self, key, value, ex=None):
        self._check()
        self.data[key] = value
        if ex:
            self.expiry[key] = time.time() + ex
        return True

    async def aclose(self):
        return None


class FakeLLM:
    """Streams a scripted answer in fixed-size chunks; final chunk carries usage."""

    def __init__(self, answer: str, chunk: int = 5, tokens: int = 123):
        self.answer = answer
        self.chunk = chunk
        self.tokens = tokens

    async def astream(self, messages, config=None):
        self.last_messages = messages
        for i in range(0, len(self.answer), self.chunk):
            yield SimpleNamespace(content=self.answer[i : i + self.chunk], usage_metadata=None)
        yield SimpleNamespace(content="", usage_metadata={"total_tokens": self.tokens})

    async def ainvoke(self, messages, config=None):
        self.last_messages = messages
        return SimpleNamespace(content=self.answer, usage_metadata={"total_tokens": self.tokens})


class FakeVectorStore:
    def __init__(self, docs=None, fail: bool = False):
        self.docs = docs or []
        self.fail = fail
        self.search_calls: list[dict] = []
        self.deleted: list[tuple[list[str], str]] = []
        self.added: list[tuple[list, list[str], str]] = []

    async def asimilarity_search_with_score(self, query, k=4, namespace=None, filter=None):
        self.search_calls.append({"query": query, "k": k, "namespace": namespace, "filter": filter})
        if self.fail:
            raise ConnectionError("pinecone down")
        return self.docs[:k]

    def delete(self, ids=None, namespace=None):
        self.deleted.append((list(ids or []), namespace))

    def add_documents(self, docs, ids=None, namespace=None):
        self.added.append((docs, list(ids or []), namespace))
