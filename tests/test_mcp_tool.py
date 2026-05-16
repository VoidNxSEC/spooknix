"""Testes para src/mcp_tool.py — dispatch, doctor inline e summarize subprocess."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_list_tools_inclui_doctor_e_summarize():
    from src import mcp_tool
    tools = await mcp_tool.list_tools()
    names = [t.name for t in tools]
    assert "spooknix_health" in names
    assert "spooknix_transcribe" in names
    assert "spooknix_diarize" in names
    assert "spooknix_doctor" in names
    assert "spooknix_summarize" in names


@pytest.mark.asyncio
async def test_transcribe_enum_inclui_turbo():
    from src import mcp_tool
    tools = await mcp_tool.list_tools()
    t = next(t for t in tools if t.name == "spooknix_transcribe")
    enum = t.inputSchema["properties"]["model_size"]["enum"]
    assert "large-v3-turbo" in enum


@pytest.mark.asyncio
async def test_handle_doctor_renderiza_tabela():
    from src import mcp_tool, doctor as doctor_mod

    fake_checks = [
        doctor_mod.Check("CUDA", "ok", "RTX 3050 (6.0 GB)"),
        doctor_mod.Check("ffmpeg", "ok", "/usr/bin/ffmpeg"),
    ]
    with patch.object(doctor_mod, "run_checks", return_value=fake_checks):
        out = await mcp_tool._handle_doctor()
    text = out[0].text
    assert "CUDA" in text
    assert "ffmpeg" in text


@pytest.mark.asyncio
async def test_handle_summarize_arquivo_inexistente():
    from src import mcp_tool
    out = await mcp_tool._handle_summarize({"file_path": "/no/such/file.mp4"})
    assert "ERROR: File not found" in out[0].text


@pytest.mark.asyncio
async def test_handle_summarize_le_output_do_subprocess(tmp_path: Path):
    from src import mcp_tool

    src = tmp_path / "fake.wav"
    src.write_bytes(b"RIFF" + b"\x00" * 40)  # placeholder; subprocess é mockado

    proc_mock = MagicMock()
    proc_mock.communicate = AsyncMock(return_value=(b"", b""))
    proc_mock.returncode = 0

    def _write_output(*args, **kwargs):
        # Lê o --out do cmd e escreve o resumo
        cmd = args
        # O comando é passado como lista
        if "--out" in cmd:
            out_path = cmd[cmd.index("--out") + 1]
            Path(out_path).write_text("# Fake summary\n[01:23] something\n", encoding="utf-8")
        return proc_mock

    async def _fake_exec(*cmd, **kwargs):
        _write_output(*cmd, **kwargs)
        return proc_mock

    with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
        out = await mcp_tool._handle_summarize({
            "file_path": str(src),
            "template": "lecture",
            "language": "en",
        })

    assert "Fake summary" in out[0].text
    assert "[01:23]" in out[0].text


@pytest.mark.asyncio
async def test_handle_summarize_subprocess_erro_retorna_stderr():
    from src import mcp_tool

    with patch("pathlib.Path.exists", return_value=True):
        proc_mock = MagicMock()
        proc_mock.communicate = AsyncMock(return_value=(b"", b"boom\ntraceback"))
        proc_mock.returncode = 2

        async def _fake_exec(*cmd, **kwargs):
            return proc_mock

        with patch("asyncio.create_subprocess_exec", side_effect=_fake_exec):
            out = await mcp_tool._handle_summarize({"file_path": "/x.mp4"})

    assert "exit 2" in out[0].text
    assert "boom" in out[0].text
