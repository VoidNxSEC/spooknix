import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from src.orchestrator import Orchestrator, State, Persona


async def _fake_chat_stream(*_args, **_kwargs):
    for chunk in [
        "Hello, my name is Sarah.",
        " How are you?",
    ]:
        yield chunk

@pytest.mark.asyncio
async def test_orchestrator_state_machine():
    llm = AsyncMock()
    tts = AsyncMock()
    # Mock para evitar chamadas reais de rede
    stt_endpoint = "http://localhost:8000/transcribe"

    orch = Orchestrator(llm=llm, tts=tts, stt_endpoint=stt_endpoint)

    # Valida estado inicial
    assert orch.state == State.LISTENING

    # Simula barge-in
    orch.state = State.SPEAKING
    orch.trigger_barge_in()
    assert orch.state == State.LISTENING

@pytest.mark.asyncio
async def test_tts_orchestration_chunking():
    # Testa se o orquestrador quebra a resposta do LLM em sentenças
    llm = MagicMock()
    tts = AsyncMock()
    llm.chat_stream = _fake_chat_stream

    orch = Orchestrator(llm=llm, tts=tts, stt_endpoint="http://localhost:8000/transcribe")
    session = MagicMock()
    persona = Persona(name="Sarah", system_prompt="Test")

    orch.state = State.SPEAKING
    orch.player.start = MagicMock()
    orch.player.enqueue = MagicMock()
    orch.player.finish = MagicMock()
    orch.player.wait_until_finished = MagicMock()

    # Mock do synthesize para não chamar rede
    tts.synthesize = AsyncMock(return_value=b"RIFF....WAVE....")
    tts.decode_wav = MagicMock(return_value=(np.zeros(100, dtype=np.float32), 24000))

    await orch.stream_llm_to_tts(session, persona)

    # Verifica se o TTS foi chamado para a sentença
    assert tts.synthesize.called
