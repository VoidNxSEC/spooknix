"""Testes para src/media.py — extract_audio via ffmpeg.

Geramos um clipe de teste com ffmpeg in-process pra evitar baixar arquivos.
Se ffmpeg não estiver no PATH, os testes são pulados.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from src.media import MediaError, extract_audio, is_video_or_compressed


@pytest.fixture
def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        pytest.skip("ffmpeg não disponível")
    return path


@pytest.fixture
def sample_mp4(tmp_path: Path, ffmpeg: str) -> Path:
    """Gera um MP4 de 1 segundo com tom de 440 Hz."""
    out = tmp_path / "tone.mp4"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:a", "aac", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def test_extract_audio_mp4_para_wav_16k(sample_mp4: Path):
    out = extract_audio(sample_mp4)
    try:
        assert out.exists()
        with wave.open(str(out)) as wf:
            assert wf.getframerate() == 16_000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() >= 16_000 * 0.9  # ~1s
    finally:
        out.unlink(missing_ok=True)


def test_extract_audio_input_inexistente():
    with pytest.raises(FileNotFoundError):
        extract_audio("/tmp/spooknix-does-not-exist.mp4")


def test_extract_audio_arquivo_invalido(tmp_path: Path, ffmpeg: str):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    with pytest.raises(MediaError):
        extract_audio(bad)


def test_is_video_or_compressed_extensoes():
    assert is_video_or_compressed("video.mp4")
    assert is_video_or_compressed("audio.mp3")
    assert is_video_or_compressed("meeting.m4a")
    assert is_video_or_compressed("podcast.opus")
    assert not is_video_or_compressed("clean.wav")
    assert not is_video_or_compressed("README.md")
