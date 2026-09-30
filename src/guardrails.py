"""Output guardrails.

The original implementation checked the *complete* answer after every token
had already been streamed to the client, and then raised ``HTTPException``
from inside the generator. By that point the 200 status line and the leaked
text were already on the wire, so the guardrail blocked nothing.

``StreamingGuard`` fixes this by inspecting text *before* it is released:
it keeps a hold-back tail so that a forbidden token split across two model
chunks is still caught before any part of it is emitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Pattern, Tuple

MASK_PATTERNS: Tuple[str, ...] = (r"\[REDACTED\]", r"\[MASKED\]", r"\*{4,}")
FORBIDDEN_TOKENS: Tuple[str, ...] = ("SYS_INTERNAL_KEY", "CONFIDENTIAL_PROP_TRADING_ALPHA")


class GuardrailViolation(Exception):
    """Raised when model output breaches an output policy."""

    def __init__(self, rule: str, message: str):
        super().__init__(message)
        self.rule = rule
        self.message = message


@dataclass(frozen=True)
class GuardrailPolicy:
    mask_patterns: Tuple[Pattern[str], ...]
    forbidden_tokens: Tuple[str, ...]
    holdback_chars: int

    @classmethod
    def default(
        cls,
        mask_patterns: Iterable[str] = MASK_PATTERNS,
        forbidden_tokens: Iterable[str] = FORBIDDEN_TOKENS,
    ) -> GuardrailPolicy:
        tokens = tuple(forbidden_tokens)
        # Hold back enough characters to cover the longest literal token and any
        # reasonably sized regex match that could straddle a chunk boundary.
        holdback = max([len(t) for t in tokens] + [32])
        return cls(
            mask_patterns=tuple(re.compile(p) for p in mask_patterns),
            forbidden_tokens=tokens,
            holdback_chars=holdback,
        )

    def check(self, text: str) -> None:
        for pattern in self.mask_patterns:
            if pattern.search(text):
                raise GuardrailViolation(
                    "masked_placeholder",
                    "Guardrail blocked: output contained internal placeholder masks.",
                )
        for token in self.forbidden_tokens:
            if token in text:
                raise GuardrailViolation(
                    "confidential_token",
                    "Guardrail blocked: confidential corporate token leakage intercepted.",
                )


DEFAULT_POLICY = GuardrailPolicy.default()


def enforce_security_guardrails(model_output: str, policy: GuardrailPolicy = DEFAULT_POLICY) -> None:
    """Non-streaming check for a complete answer. Raises GuardrailViolation."""
    policy.check(model_output)


@dataclass
class StreamingGuard:
    """Releases streamed text only after it has passed the policy.

    Usage::

        guard = StreamingGuard()
        for chunk in model_stream:
            safe = guard.feed(chunk)   # may raise GuardrailViolation
            if safe:
                send(safe)
        send(guard.flush())            # may raise GuardrailViolation
    """

    policy: GuardrailPolicy = field(default_factory=lambda: DEFAULT_POLICY)
    _buffer: str = ""
    _emitted: int = 0

    def feed(self, chunk: str | None) -> str:
        if not chunk:
            return ""
        self._buffer += chunk
        # Re-check the whole buffer: regexes such as \*{4,} can grow across chunks
        # and it keeps the logic obviously correct. Answers are small (< ~10 KB).
        self.policy.check(self._buffer)
        safe_until = len(self._buffer) - self.policy.holdback_chars
        if safe_until <= self._emitted:
            return ""
        out = self._buffer[self._emitted:safe_until]
        self._emitted = safe_until
        return out

    def flush(self) -> str:
        self.policy.check(self._buffer)
        out = self._buffer[self._emitted:]
        self._emitted = len(self._buffer)
        return out

    @property
    def text(self) -> str:
        return self._buffer
