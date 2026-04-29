from __future__ import annotations

import os

import pytest

from src.llm_client import LLMClient


def test_llm_client_requires_explicit_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(ValueError, match="LLM não configurado"):
        LLMClient()


def test_llm_client_accepts_local_backend_without_real_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "qwen-3.5")

    client = LLMClient()

    assert client.base_url == "http://localhost:8080/v1"
    assert client.api_key == "sk-no-key-required"
    assert client.default_model == "qwen-3.5"


def test_llm_client_keeps_openai_mode_when_key_exists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    client = LLMClient()

    assert client.base_url is None
    assert client.api_key == "sk-test"
    assert client.default_model == "gpt-4o-mini"
