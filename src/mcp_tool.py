# src/mcp_tool.py
"""Spooknix MCP Tool — exposes STT + diarization as MCP tools for Claude.

Runs as a standalone MCP server (stdio transport) that proxies requests to
the Spooknix HTTP server (default: http://localhost:8000).

Usage:
  spooknix-mcp                          # default server at localhost:8000
  spooknix-mcp --server http://host:8000

Register in Claude Code settings (claude_desktop_config.json or .claude/settings.json):
  {
    "mcpServers": {
      "spooknix": {
        "command": "spooknix-mcp",
        "args": []
      }
    }
  }

Tools exposed:
  spooknix_health     — check server/model/GPU status
  spooknix_transcribe — transcribe audio file → text + segments
  spooknix_diarize    — transcribe + speaker diarization
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

import aiohttp
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_SERVER = "http://localhost:8000"


def _server_url() -> str:
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--server" and i < len(sys.argv):
            return sys.argv[i + 1]
    return DEFAULT_SERVER


# ── MCP Server ───────────────────────────────────────────────────────────────

server = Server("spooknix")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="spooknix_health",
            description=(
                "Check Spooknix STT server health. "
                "Returns model name, device (cuda/cpu), VRAM, diarization status."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="spooknix_transcribe",
            description=(
                "Transcribe an audio or video file using Whisper (local, privacy-first). "
                "Accepts file path on the local filesystem. "
                "Returns full transcript text and timestamped segments."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the audio/video file to transcribe.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g. 'pt', 'en', 'es'). Default: 'pt'.",
                        "default": "pt",
                    },
                    "model_size": {
                        "type": "string",
                        "description": "Whisper model size override (tiny/base/small/medium/large-v3).",
                        "enum": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="spooknix_diarize",
            description=(
                "Transcribe an audio file AND identify speakers (diarization). "
                "Requires ENABLE_DIARIZATION=true on server and pyannote-audio installed. "
                "Returns segments tagged with speaker labels (SPEAKER_00, SPEAKER_01, …)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the audio/video file.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code. Default: 'pt'.",
                        "default": "pt",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    base_url = _server_url()

    if name == "spooknix_health":
        return await _handle_health(base_url)
    elif name == "spooknix_transcribe":
        return await _handle_transcribe(base_url, arguments, diarize=False)
    elif name == "spooknix_diarize":
        return await _handle_transcribe(base_url, arguments, diarize=True)
    else:
        raise ValueError(f"Unknown tool: {name}")


# ── Handlers ─────────────────────────────────────────────────────────────────


async def _handle_health(base_url: str) -> list[types.TextContent]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
        return [types.TextContent(type="text", text=json.dumps(data, indent=2, ensure_ascii=False))]
    except Exception as exc:
        return [types.TextContent(
            type="text",
            text=f"ERROR: Cannot reach Spooknix server at {base_url}\n{exc}\n\n"
                 "Make sure the server is running: docker compose up -d",
        )]


async def _handle_transcribe(
    base_url: str,
    arguments: dict,
    diarize: bool,
) -> list[types.TextContent]:
    file_path = Path(arguments["file_path"])
    language = arguments.get("language", "pt")
    model_size = arguments.get("model_size")

    if not file_path.exists():
        return [types.TextContent(
            type="text",
            text=f"ERROR: File not found: {file_path}",
        )]

    try:
        audio_bytes = file_path.read_bytes()
    except OSError as exc:
        return [types.TextContent(type="text", text=f"ERROR reading file: {exc}")]

    form = aiohttp.FormData()
    form.add_field("file", audio_bytes, filename=file_path.name)
    form.add_field("language", language)
    if model_size:
        form.add_field("model_size", model_size)
    if diarize:
        form.add_field("diarize", "true")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/transcribe",
                data=form,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 min for large files
            ) as resp:
                result = await resp.json()

        if "error" in result:
            return [types.TextContent(type="text", text=f"Server error: {result['error']}")]

        # Format output
        lines: list[str] = []
        lines.append(f"# Transcript — {file_path.name}")
        lines.append(f"Model: {result.get('model', '?')} | Language: {language}")
        if result.get("diarized"):
            lines.append("Diarization: enabled")
        lines.append("")
        lines.append("## Full Text")
        lines.append(result.get("text", "").strip())
        lines.append("")
        lines.append("## Segments")

        for seg in result.get("segments", []):
            speaker = f"[{seg['speaker']}] " if "speaker" in seg else ""
            t_start = f"{seg['start']:.1f}s"
            t_end = f"{seg['end']:.1f}s"
            lines.append(f"{speaker}{t_start}→{t_end}: {seg['text'].strip()}")

        return [types.TextContent(type="text", text="\n".join(lines))]

    except aiohttp.ClientConnectorError:
        return [types.TextContent(
            type="text",
            text=f"ERROR: Cannot reach Spooknix server at {base_url}\n"
                 "Make sure the server is running: docker compose up -d",
        )]
    except Exception as exc:
        return [types.TextContent(type="text", text=f"ERROR: {exc}")]


# ── Entry point ───────────────────────────────────────────────────────────────


async def _run() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
