"""Factories for external clients (OpenAI, Pinecone, image store).

Kept in one place so the API and workers build identical, correctly
configured clients (timeouts, retries, namespaces) and so that nothing
connects to external services at import time.
"""

from __future__ import annotations

import base64

from config.settings import Settings

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def build_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )


def build_vectorstore(settings: Settings, embeddings=None):
    """Namespace is passed per call (one namespace per tenant), never fixed here."""
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings or build_embeddings(settings),
        pinecone_api_key=settings.PINECONE_API_KEY.get_secret_value(),
    )


def build_chat_llm(settings: Settings, *, model: str, max_tokens: int, streaming: bool = False):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        streaming=streaming,
        stream_usage=True,  # report token usage on streamed responses (quota accounting)
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        timeout=settings.OPENAI_TIMEOUT_SECONDS,
        max_retries=settings.OPENAI_MAX_RETRIES,
    )


def image_bytes_to_base64(raw: bytes | None) -> str | None:
    """Return base64 PNG for a stored image.

    New records are stored as raw PNG bytes (25% smaller than base64 text).
    Legacy records written by the original pipeline were base64 text; both
    are accepted.
    """
    if not raw:
        return None
    if raw.startswith(PNG_MAGIC):
        return base64.b64encode(raw).decode("ascii")
    return raw.decode("ascii")
