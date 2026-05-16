"""Probe llama.cpp + integração com LLMClient quando OPENAI_API_KEY ausente.

Padrão da voidnxlabs: llama-server em :8081, endpoint /v1/models (OpenAI-compatible).
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src import llm_fallback


def _fake_response(payload: dict):
    fake = MagicMock()
    fake.read.return_value = json.dumps(payload).encode()
    fake.__enter__ = lambda self: self
    fake.__exit__ = lambda *args: None
    return fake


def test_probe_offline_retorna_lista_vazia():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no host")):
        assert llm_fallback.probe("http://localhost:8081") == []


def test_probe_online_extrai_modelos():
    # llama.cpp segue formato OpenAI: {"data": [{"id": "..."}]}
    payload = {"data": [{"id": "qwen2.5-7b-instruct"}, {"id": "llama-3-8b"}]}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        models = llm_fallback.probe()
    assert "qwen2.5-7b-instruct" in models
    assert "llama-3-8b" in models


def test_probe_payload_invalido_retorna_vazio():
    fake = MagicMock()
    fake.read.return_value = b"not json"
    fake.__enter__ = lambda self: self
    fake.__exit__ = lambda *args: None
    with patch("urllib.request.urlopen", return_value=fake):
        assert llm_fallback.probe() == []


def test_pick_model_preferred_quando_disponivel():
    assert llm_fallback.pick_model(
        ["llama-3-8b", "qwen2.5-7b"],
        preferred="qwen2.5-7b",
    ) == "qwen2.5-7b"


def test_pick_model_lista_vazia_retorna_none():
    assert llm_fallback.pick_model([]) is None


def test_pick_model_primeiro_se_preferred_ausente_ou_none():
    # llama-server serve 1 modelo de cada vez → o primeiro é o único certo.
    assert llm_fallback.pick_model(["llama-3-8b", "mistral"]) == "llama-3-8b"
    assert llm_fallback.pick_model(["llama-3-8b"], preferred="qwen-missing") == "llama-3-8b"


def test_maybe_build_client_quando_offline_retorna_none():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("x")):
        assert llm_fallback.maybe_build_client() is None


def test_llm_client_usa_fallback_quando_sem_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLAMACPP_URL", raising=False)
    monkeypatch.delenv("LLAMACPP_MODEL", raising=False)
    # Recarrega o módulo para pegar env limpa
    import importlib
    importlib.reload(llm_fallback)

    payload = {"data": [{"id": "qwen2.5-7b-instruct"}]}
    with patch("urllib.request.urlopen", return_value=_fake_response(payload)):
        from src.llm_client import LLMClient
        client = LLMClient()
    assert client.base_url.startswith("http://localhost:8081")
    assert client.default_model == "qwen2.5-7b-instruct"


def test_llm_client_falha_quando_sem_key_e_sem_llamacpp(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("nope")):
        from src.llm_client import LLMClient
        with pytest.raises(ValueError, match="llama-server|LLM"):
            LLMClient()
