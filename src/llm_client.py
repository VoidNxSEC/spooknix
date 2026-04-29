"""
Módulo integrador para interagir com Large Language Models (LLM).

Usa a biblioteca `openai` com suporte para troca de base_url, permitindo
integrar facilmente com OpenAI, NVIDIA NIM, Groq, Ollama, vLLM, etc.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


class LLMClient:
    """Cliente wrapper para o LLM. Suporta streaming nativo via API OpenAI-compatível."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Inicializa o cliente.
        Tenta buscar credenciais nas variáveis de ambiente padrão se não fornecidas.
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.uses_openai_default_endpoint = self.base_url is None

        if self.uses_openai_default_endpoint and not key:
            raise ValueError(
                "LLM não configurado. Para rodar 100% local, defina LLM_BASE_URL e "
                "LLM_MODEL (ex: http://localhost:8080/v1 + qwen-3.5). "
                "Use OPENAI_API_KEY apenas se quiser OpenAI de verdade."
            )

        # Algumas APIs locais não requerem chave, mas a lib exige uma string
        self.api_key = key if key else "sk-no-key-required"

        default_model = "gpt-4o-mini" if self.uses_openai_default_endpoint else "local-model"
        self.default_model = model or os.getenv("LLM_MODEL") or default_model

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Envia mensagens para o LLM e retorna um gerador assíncrono de tokens (streaming).
        Usado para a interação em tempo real na entrevista.
        """
        target_model = model or self.default_model
        try:
            stream = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            yield f"\n[Erro no LLM: {e}]"

    async def generate(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
    ) -> str:
        """
        Gera a resposta completa de uma vez.
        Útil para o relatório final do Avaliador.
        """
        target_model = model or self.default_model
        try:
            response = await self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"Erro ao gerar relatório com LLM: {e}"


class InterviewSession:
    """
    Gerencia o estado e o histórico de uma simulação de entrevista.
    """

    def __init__(self, system_prompt: str):
        self.messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

    def add_user_message(self, content: str) -> None:
        """Adiciona a resposta (transcrição) do candidato."""
        self.messages.append({"role": "user", "content": content.strip()})

    def add_assistant_message(self, content: str) -> None:
        """Adiciona a pergunta/resposta do entrevistador LLM."""
        self.messages.append({"role": "assistant", "content": content.strip()})

    def get_messages(self) -> list[ChatCompletionMessageParam]:
        """Retorna o histórico completo para a API."""
        return self.messages

    def get_transcript_text(self) -> str:
        """Formata o histórico para envio ao Avaliador."""
        lines = []
        for msg in self.messages:
            if msg["role"] == "system":
                continue

            # Assume que string vazias ou nulas são puladas
            content = msg.get("content", "")
            if not content:
                continue

            role_name = "Candidate" if msg["role"] == "user" else "Interviewer"
            lines.append(f"**{role_name}:** {content}")

        return "\n\n".join(lines)


def load_template(filename: str) -> str:
    """Carrega o prompt da pasta templates/."""
    path = Path(__file__).parent.parent / "templates" / filename
    if not path.exists():
        raise FileNotFoundError(f"Template {filename} não encontrado em {path}")
    return path.read_text(encoding="utf-8").strip()
