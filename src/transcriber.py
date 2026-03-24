# src/transcriber.py
"""
Módulo de transcrição - Privacy-first STT Engine
Nenhum dado sai da máquina local.
"""

from pathlib import Path
from typing import Generator
from faster_whisper import WhisperModel
import numpy as np
import time


SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def _compute_type(size: str, device: str) -> str:
    """Seleciona compute_type ideal para o modelo e dispositivo."""
    if device == "cpu":
        return "int8"
    if size in ("large-v2", "large-v3"):
        return "int8_float16"  # ~3GB VRAM — cabe folgado em 6GB
    return "float16"           # tiny/base/small/medium


def get_model(size: str = "large-v3", device: str = "cuda", compute_type: str | None = None):
    """
    Carrega modelo Whisper.

    Tamanhos disponíveis para 6GB VRAM (RTX 3050):
    - tiny / base : ~1GB   (mais rápido, menos preciso)
    - small       : ~2GB   (balanceado)
    - medium      : ~5GB   (alta qualidade, pouca margem)
    - large-v2/v3 : ~3GB   (int8_float16) ← Recomendado — qualidade máxima
    """
    ct = compute_type or _compute_type(size, device)
    print(f"📥 Carregando modelo '{size}' no dispositivo '{device}' ({ct})...")
    start = time.time()

    model = WhisperModel(
        size,
        device=device,
        compute_type=ct,
    )
    
    elapsed = time.time() - start
    print(f"✅ Modelo carregado em {elapsed:.1f}s")
    
    return model


def transcribe_file(model, audio_path: str, language: str = "pt", on_progress=None):
    """
    Transcreve um arquivo de áudio.
    
    Retorna:
        - text: Texto completo
        - segments: Lista de segmentos com timestamps
    """
    print(f"🎯 Transcrevendo: {audio_path}")
    start = time.time()
    
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        best_of=5,          # 5 candidatos — escolhe o melhor (↑ precisão conversacional)
        temperature=0.0,    # determinístico — melhor para STT
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            threshold=0.4,  # mais sensível que o padrão (0.5)
        ),
        word_timestamps=True,
    )
    
    if on_progress:
        on_progress(0.0, info.duration)
        
    # Processar segmentos
    results = []
    full_text = []
    
    for segment in segments:
        words = []
        if getattr(segment, "words", None):
            for w in segment.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "probability": w.probability
                })
                
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words
        })
        full_text.append(segment.text.strip())
        
        if on_progress:
            on_progress(segment.end, info.duration)
        else:
            # Print em tempo real
            print(f"  [{segment.start:.1f}s → {segment.end:.1f}s] {segment.text.strip()}")
    
    elapsed = time.time() - start
    print(f"\n✅ Transcrição completa em {elapsed:.1f}s")
    print(f"📊 Idioma detectado: {info.language} ({info.language_probability:.0%})")
    
    return {
        "text": " ".join(full_text),
        "segments": results,
        "language": info.language,
        "duration": info.duration
    }


def transcribe_stream(
    model: WhisperModel,
    audio: np.ndarray,
    language: str = "pt",
    chunk_length_s: int = 3,
) -> Generator[dict, None, None]:
    """Transcreve um array numpy diretamente, gerando segmentos conforme decodifica.

    Usa o generator nativo do faster-whisper — sem arquivo temporário por janela.
    condition_on_previous_text=True mantém contexto conversacional entre chunks.

    Args:
        model: Instância WhisperModel já carregada.
        audio: Array float32 normalizado em [-1, 1] a 16 kHz.
        language: Código do idioma.
        chunk_length_s: Tamanho do chunk interno do faster-whisper (segundos).

    Yields:
        dict com {start, end, text, words, avg_confidence}
    """
    if audio.size == 0:
        return

    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        temperature=0.0,
        vad_filter=True,
        word_timestamps=True,
        chunk_length=chunk_length_s,
        condition_on_previous_text=True,
        batch_size=8,
    )

    for segment in segments:
        words = []
        probs: list[float] = []
        if getattr(segment, "words", None):
            for w in segment.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "probability": w.probability,
                })
                probs.append(w.probability)

        avg_confidence = float(sum(probs) / len(probs)) if probs else 0.0

        yield {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words,
            "avg_confidence": avg_confidence,
        }


def generate_srt(segments: list, output_path: str):
    """Gera arquivo de legenda .srt"""
    
    def format_timestamp(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")
    
    print(f"💾 SRT salvo: {output_path}")


# === TESTE DIRETO ===
if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("🧪 TESTE DE VALIDAÇÃO - STT Pipeline")
    print("=" * 50)
    print()
    
    # 1. Verificar CUDA
    import torch
    print(f"🔍 PyTorch CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔍 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # 2. Carregar modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("large-v3", device)
    print()
    
    # 3. Teste com arquivo (se fornecido)
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        result = transcribe_file(model, audio_file, language="pt")
        
        # Salvar outputs
        output_base = Path("outputs")
        txt_path = output_base / "transcripts" / f"{Path(audio_file).stem}.txt"
        srt_path = output_base / "subtitles" / f"{Path(audio_file).stem}.srt"
        
        # Salvar texto
        txt_path.write_text(result["text"], encoding="utf-8")
        print(f"💾 TXT salvo: {txt_path}")
        
        # Salvar SRT
        generate_srt(result["segments"], str(srt_path))
    else:
        print("💡 Para testar transcrição:")
        print("   python src/transcriber.py seu_audio.mp3")
        print()
        print("✅ Validação básica concluída!")