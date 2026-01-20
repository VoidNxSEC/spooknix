# src/transcriber.py
"""
Módulo de transcrição - Privacy-first STT Engine
Nenhum dado sai da máquina local.
"""

from pathlib import Path
from faster_whisper import WhisperModel
import time


def get_model(size: str = "small", device: str = "cuda"):
    """
    Carrega modelo Whisper.
    
    Tamanhos disponíveis para 6GB VRAM:
    - tiny:  ~1GB  (mais rápido, menos preciso)
    - base:  ~1GB  (bom para real-time)
    - small: ~2GB  (balanceado) ← Recomendado para real-time
    - medium: ~5GB (alta qualidade) ← Recomendado para batch
    """
    print(f"📥 Carregando modelo '{size}' no dispositivo '{device}'...")
    start = time.time()
    
    model = WhisperModel(
        size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )
    
    elapsed = time.time() - start
    print(f"✅ Modelo carregado em {elapsed:.1f}s")
    
    return model


def transcribe_file(model, audio_path: str, language: str = "pt"):
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
        vad_filter=True,  # Remove silêncios
        vad_parameters=dict(
            min_silence_duration_ms=500,
        )
    )
    
    # Processar segmentos
    results = []
    full_text = []
    
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        full_text.append(segment.text.strip())
        
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
    model = get_model("small", device)
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