import pytest
from pydantic import ValidationError

from config.settings import Settings

BASE = {"OPENAI_API_KEY": "sk", "PINECONE_API_KEY": "pc"}


def make(**overrides):
    return Settings(_env_file=None, **{**BASE, **overrides})


def test_weak_jwt_secret_rejected_in_production():
    with pytest.raises(ValidationError):
        make(ENV="production", JWT_SECRET="dev-super-secure-hmac-signing-key")


def test_strong_jwt_secret_accepted_in_production():
    s = make(ENV="production", JWT_SECRET="Zq3" * 16)
    assert s.ENVIRONMENT == "production"


def test_none_algorithm_rejected():
    with pytest.raises(ValidationError):
        make(ENV="test", JWT_SECRET="x" * 40, JWT_ALGORITHM="none")


def test_blank_optional_values_become_none():
    s = make(ENV="test", JWT_SECRET="x" * 40, JWT_ISSUER="", JWT_AUDIENCE="  ", JWT_INGEST_SCOPE="")
    assert s.JWT_ISSUER is None and s.JWT_AUDIENCE is None and s.JWT_INGEST_SCOPE is None


def test_tracing_requires_api_key():
    with pytest.raises(ValidationError):
        make(ENV="test", JWT_SECRET="x" * 40, LANGCHAIN_TRACING_V2="true")
