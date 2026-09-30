import pytest

from src.guardrails import GuardrailViolation, StreamingGuard, enforce_security_guardrails


def test_clean_output_passes():
    enforce_security_guardrails("The total net profit recorded for the division was $45.2M.")


@pytest.mark.parametrize("text", ["value is [MASKED] here", "see [REDACTED]", "pw: ********"])
def test_masked_placeholders_blocked(text):
    with pytest.raises(GuardrailViolation) as exc:
        enforce_security_guardrails(text)
    assert exc.value.rule == "masked_placeholder"


def test_confidential_token_blocked():
    with pytest.raises(GuardrailViolation) as exc:
        enforce_security_guardrails("weights map to SYS_INTERNAL_KEY indicators")
    assert exc.value.rule == "confidential_token"


def _run_stream(chunks):
    guard = StreamingGuard()
    out = []
    for c in chunks:
        out.append(guard.feed(c))
    out.append(guard.flush())
    return "".join(out)


def test_stream_releases_full_clean_text():
    text = "Revenue grew 12% year over year to $1.24B in FY2024, driven by services. " * 3
    chunks = [text[i : i + 7] for i in range(0, len(text), 7)]
    assert _run_stream(chunks) == text


def test_stream_blocks_token_split_across_chunks_without_leaking_it():
    guard = StreamingGuard()
    released = []
    chunks = ["Some preamble text that is long enough to be released. ", "SYS_INT", "ERNAL_KEY and more"]
    with pytest.raises(GuardrailViolation):
        for c in chunks:
            released.append(guard.feed(c))
        released.append(guard.flush())
    leaked = "".join(released)
    assert "SYS_INT" not in leaked
    assert leaked.startswith("Some preamble")


def test_stream_blocks_growing_asterisk_run():
    guard = StreamingGuard()
    guard.feed("masked value: **")
    with pytest.raises(GuardrailViolation):
        guard.feed("**")
