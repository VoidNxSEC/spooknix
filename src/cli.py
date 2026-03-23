# src/cli.py
"""CLI do Spooknix — Privacy-first STT Engine."""

import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TimeRemainingColumn

console = Console()


@click.group()
def cli():
    """Spooknix — Privacy-first Speech-to-Text Engine."""
    pass


@cli.command()
def info():
    """Mostra status do sistema: GPU, VRAM e modelos disponíveis."""
    import torch

    table = Table(title="Spooknix — System Info", show_header=False, min_width=52)
    table.add_column("Campo", style="bold cyan")
    table.add_column("Valor")

    cuda = torch.cuda.is_available()
    table.add_row("CUDA", "✅ disponível" if cuda else "❌ não disponível")
    if cuda:
        table.add_row("GPU", torch.cuda.get_device_name(0))
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        table.add_row("VRAM", f"{vram_gb:.1f} GB")

    table.add_row("Modelos", "tiny (~1GB) · base (~1GB) · small (~2GB) · medium (~5GB)")
    table.add_row("Idioma padrão", "pt (Português)")

    console.print(table)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="pt", show_default=True,
              help="Código do idioma (pt, en, es, …)")
@click.option("--model", "-m",
              type=click.Choice(["tiny", "base", "small", "medium"]),
              default="small", show_default=True,
              help="Tamanho do modelo Whisper")
@click.option("--output-dir", "-o", default="outputs", show_default=True,
              type=click.Path(),
              help="Diretório raiz para os arquivos de saída")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["txt", "srt", "json", "all"]),
              default="all", show_default=True,
              help="Formato(s) de saída")
def file(audio_path, language, model, output_dir, fmt):
    """Transcreve um arquivo de áudio ou vídeo."""
    from .transcriber import get_model, transcribe_file, generate_srt
    import torch

    stem = Path(audio_path).stem
    output_base = Path(output_dir)

    # Carrega o modelo
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Carregando modelo '{model}'…", total=None)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = get_model(model, device)

    console.print(
        f"[bold cyan]►[/bold cyan] Modelo [bold]{model}[/bold] "
        f"no dispositivo [bold]{device}[/bold]\n"
    )

    # Transcreve
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Transcrevendo...", total=100.0)
        
        def on_progress_cb(current, total):
            if total > 0:
                progress.update(task_id, completed=(current / total) * 100.0)
                
        result = transcribe_file(m, audio_path, language=language, on_progress=on_progress_cb)

    # Persiste outputs
    saved = []

    if fmt in ("txt", "all"):
        out = output_base / "transcripts" / f"{stem}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result["text"], encoding="utf-8")
        saved.append(str(out))

    if fmt in ("srt", "all"):
        out = output_base / "subtitles" / f"{stem}.srt"
        out.parent.mkdir(parents=True, exist_ok=True)
        generate_srt(result["segments"], str(out))
        saved.append(str(out))

    if fmt in ("json", "all"):
        out = output_base / "transcripts" / f"{stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(str(out))

    # Resumo final
    files_list = "\n".join(f"  {p}" for p in saved)
    console.print(
        Panel(
            f"[green]Idioma:[/green]    {result['language']}\n"
            f"[green]Duração:[/green]   {result['duration']:.1f}s\n"
            f"[green]Segmentos:[/green] {len(result['segments'])}\n"
            f"[green]Arquivos:[/green]\n{files_list}",
            title="✅ Transcrição concluída",
            border_style="green",
        )
    )


SERVER_URL = os.getenv("SPOOKNIX_URL", "http://localhost:8000")


@cli.command()
@click.option("--language", "-l", default="pt", show_default=True,
              help="Código do idioma (pt, en, es, …)")
@click.option("--silence", "-s", default=2.0, type=float, show_default=True,
              help="Segundos de silêncio para parar a gravação")
@click.option("--threshold", "-t", default=0.01, type=float, show_default=True,
              help="Limiar de RMS para detecção de silêncio")
@click.option("--clip/--no-clip", default=False,
              help="Copiar resultado para clipboard via wl-copy (Wayland)")
@click.option("--max-duration", default=120.0, type=float, show_default=True,
              help="Duração máxima da gravação em segundos")
@click.option("--server", default=None, show_default=True,
              help=f"URL do servidor (padrão: $SPOOKNIX_URL ou {SERVER_URL})")
@click.option("--stop-word", "-w", default="stop", show_default=True,
              help="Palavra-chave falada para parar a gravação (ex: 'stop', 'para')")
def record(language, silence, threshold, clip, max_duration, server, stop_word):
    """Grava do microfone e transcreve via servidor HTTP."""
    import os
    import subprocess
    import urllib.request
    import urllib.error
    from .recorder import record_until_silence, RecordingError

    base_url = server or SERVER_URL

    # Verificar servidor antes de gravar
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
            import json
            info = json.loads(resp.read())
            console.print(
                f"[bold cyan]►[/bold cyan] Servidor [bold]{base_url}[/bold] "
                f"| modelo [bold]{info.get('model','?')}[/bold] "
                f"| device [bold]{info.get('device','?')}[/bold]"
                f"{' | CUDA ✓' if info.get('cuda') else ''}\n"
            )
    except (urllib.error.URLError, Exception) as exc:
        console.print(f"[red]✗ Servidor não disponível em {base_url}: {exc}[/red]")
        console.print("[dim]  Inicie com: docker compose up -d[/dim]")
        return

    # Função de stop por palavra-chave — chama o servidor com os últimos segundos
    def _make_stop_check(word: str):
        boundary = "spooknix-boundary-kw"

        def check(wav_bytes: bytes) -> bool:
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="kw.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + wav_bytes + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="language"\r\n\r\n'
                f"{language}\r\n"
                f"--{boundary}--\r\n"
            ).encode()
            req = urllib.request.Request(
                f"{base_url}/transcribe",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    import json as _json
                    text = _json.loads(resp.read()).get("text", "").lower()
                    return word.lower() in text
            except Exception:
                return False

        return check

    # Gravar
    tmp_path: str | None = None
    try:
        with console.status(
            f"[red bold]● Gravando… (Ctrl+C ou diga '{stop_word}' para parar)[/red bold]"
        ):
            try:
                tmp_path = record_until_silence(
                    silence_duration=silence,
                    silence_threshold=threshold,
                    max_duration=max_duration,
                    stop_check_fn=_make_stop_check(stop_word),
                    stop_check_interval=2.0,
                )
            except KeyboardInterrupt:
                console.print("\n[yellow]Gravação interrompida.[/yellow]")
                if tmp_path is None:
                    return
            except RecordingError as exc:
                console.print(f"[red]Erro de gravação: {exc}[/red]")
                return

        if tmp_path is None:
            console.print("[red]Nenhum áudio capturado.[/red]")
            return

        console.print("[green]✓ Gravação concluída.[/green]")

        # Enviar para o servidor via multipart/form-data
        with console.status("[cyan]Transcrevendo…[/cyan]"):
            import urllib.parse
            import email.generator
            import io

            boundary = "spooknix-boundary-42"
            wav_data = Path(tmp_path).read_bytes()

            body_parts = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="recording.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + wav_data + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="language"\r\n\r\n'
                f"{language}\r\n"
                f"--{boundary}--\r\n"
            ).encode()

            req = urllib.request.Request(
                f"{base_url}/transcribe",
                data=body_parts,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    import json
                    result = json.loads(resp.read())
            except urllib.error.URLError as exc:
                console.print(f"[red]Erro na transcrição: {exc}[/red]")
                return

        text = result.get("text", "").strip()
        lang_detected = result.get("language", language)
        duration = result.get("duration", 0.0)

        console.print(
            Panel(
                text or "(sem texto detectado)",
                title=f"✅ Transcrição [{lang_detected}] — {duration:.1f}s",
                border_style="green",
            )
        )

        # Clipboard
        if clip and text:
            try:
                subprocess.run(["wl-copy", text], check=True, timeout=5)
                console.print("[dim]📋 Copiado para o clipboard.[/dim]")
            except FileNotFoundError:
                console.print("[yellow]⚠ wl-copy não encontrado — clipboard ignorado.[/yellow]")
            except subprocess.CalledProcessError as exc:
                console.print(f"[yellow]⚠ Erro ao copiar: {exc}[/yellow]")

    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    cli()
