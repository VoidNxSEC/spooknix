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

console = Console(stderr=True)
out_console = Console()


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

    table.add_row(
        "Modelos",
        "tiny (~1GB) · base (~1GB) · small (~2GB) · medium (~5GB) · large-v3 (~3GB int8_float16) ← recomendado",
    )
    table.add_row("Idioma padrão", "pt (Português)")

    out_console.print(table)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="pt", show_default=True,
              help="Código do idioma (pt, en, es, …)")
@click.option("--model", "-m",
              type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
              default="large-v3", show_default=True,
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
@click.option("--max-duration", default=300.0, type=float, show_default=True,
              help="Duração máxima da gravação em segundos")
@click.option("--server", default=None, show_default=True,
              help=f"URL do servidor (padrão: $SPOOKNIX_URL ou {SERVER_URL})")
@click.option("--stop-word", "-w", default="stop", show_default=True,
              help="Palavra-chave falada para parar a gravação (ex: 'stop', 'para')")
@click.option("--diarize/--no-diarize", default=False,
              help="Ativar diarização de speakers via pyannote-audio (requer HF_TOKEN)")
@click.option("--out", default=None, type=click.Path(dir_okay=False, writable=True),
              help="Salvar a transcrição final em um arquivo de texto/markdown")
def record(language, silence, threshold, clip, max_duration, server, stop_word, diarize, out):
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

            diarize_value = "true" if diarize else "false"
            body_parts = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="recording.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode() + wav_data + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="language"\r\n\r\n'
                f"{language}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="diarize"\r\n\r\n'
                f"{diarize_value}\r\n"
                f"--{boundary}--\r\n"
            ).encode()

            req = urllib.request.Request(
                f"{base_url}/transcribe",
                data=body_parts,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=max_duration + 120) as resp:
                    import json
                    result = json.loads(resp.read())
            except urllib.error.URLError as exc:
                console.print(f"[red]Erro na transcrição: {exc}[/red]")
                return

        text = result.get("text", "").strip()
        lang_detected = result.get("language", language)
        duration = result.get("duration", 0.0)
        diarized = result.get("diarized", False)
        model_used = result.get("model", "?")

        if diarized:
            # Exibir segmentos com speaker labels
            speaker_lines = []
            for seg in result.get("segments", []):
                spk = seg.get("speaker", "?")
                speaker_lines.append(f"{spk}: {seg['text']}")
            body = "\n".join(speaker_lines) or "(sem texto detectado)"
        else:
            body = text or "(sem texto detectado)"

        title = f"✅ Transcrição [{lang_detected}] — {duration:.1f}s — modelo {model_used}"
        if diarized:
            title += " — diarizado"

        console.print(Panel(body, title=title, border_style="green"))
        print(body)

        if out:
            try:
                Path(out).write_text(body, encoding="utf-8")
                console.print(f"[dim]💾 Salvo em: {out}[/dim]")
            except Exception as exc:
                console.print(f"[red]Erro ao salvar arquivo: {exc}[/red]")

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


SERVER_WS_URL = os.getenv("SPOOKNIX_WS_URL", "ws://localhost:8000")


@cli.command()
@click.option("--language", "-l", default="pt", show_default=True,
              help="Código do idioma (pt, en, es, …)")
@click.option("--window", default=3.0, type=float, show_default=True,
              help="Janela de flush em segundos")
@click.option("--clip/--no-clip", default=False,
              help="Copiar resultado final para clipboard via wl-copy (Wayland)")
@click.option("--stop-word", "-w", default=None,
              help="Palavra-chave no texto parcial acumulado para encerrar automaticamente")
@click.option("--server", default=None,
              help=f"URL base do servidor WebSocket (padrão: $SPOOKNIX_WS_URL ou {SERVER_WS_URL})")
@click.option("--max-duration", default=300.0, type=float, show_default=True,
              help="Duração máxima da sessão em segundos")
@click.option("--out", default=None, type=click.Path(dir_okay=False, writable=True),
              help="Salvar a transcrição final em um arquivo de texto/markdown")
def stream(language, window, clip, stop_word, server, max_duration, out):
    """Stream do microfone com transcrição parcial em tempo real via WebSocket."""
    import asyncio
    asyncio.run(_stream_async(language, window, clip, stop_word, server, max_duration, out))


async def _stream_async(
    language: str,
    window: float,
    clip: bool,
    stop_word: str | None,
    server: str | None,
    max_duration: float,
    out: str | None,
):
    import asyncio
    import json as _json
    import subprocess

    import numpy as np
    import sounddevice as sd
    import websockets  # type: ignore
    from rich.live import Live
    from rich.text import Text

    from .recorder import BLOCKSIZE, SAMPLE_RATE

    ws_base = (server or SERVER_WS_URL).rstrip("/")
    url = f"{ws_base}/ws/stream?language={language}&window={window}"

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def sd_callback(indata: np.ndarray, frames: int, t, status) -> None:
        data = indata[:, 0].copy().astype(np.float32)
        loop.call_soon_threadsafe(queue.put_nowait, data.tobytes())

    confirmed: list[str] = []
    partial_buf = ""
    # Mutable container — acessível de closures aninhadas sem nonlocal
    state = {"stop": False}

    def _render() -> Text:
        txt = Text()
        if confirmed:
            txt.append(" ".join(confirmed), style="dim")
            txt.append(" ")
        if partial_buf:
            txt.append(partial_buf.lstrip(), style="bold cyan")
        return txt

    try:
        async with websockets.connect(url, max_size=2**23) as ws:
            # session_start
            raw = await ws.recv()
            info = _json.loads(raw)
            console.print(
                f"[dim]WS conectado | modelo [bold]{info.get('model')}[/bold] "
                f"| device {info.get('device')} | janela {info.get('window_s')}s[/dim]\n"
            )

            deadline = loop.time() + max_duration

            async def _send_loop() -> None:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=BLOCKSIZE,
                    callback=sd_callback,
                ):
                    while loop.time() < deadline and not state["stop"]:
                        try:
                            chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        if chunk is None:
                            break
                        await ws.send(chunk)
                # Sinaliza encerramento
                await queue.put(None)

            send_task = asyncio.create_task(_send_loop())

            with Live(console=console, refresh_per_second=10) as live:
                try:
                    async for raw_msg in ws:
                        if not isinstance(raw_msg, str):
                            continue
                        data = _json.loads(raw_msg)
                        t = data.get("type")

                        if t == "partial":
                            partial_buf += data.get("text", "")
                            if stop_word and stop_word.lower() in partial_buf.lower():
                                state["stop"] = True
                                await ws.send(_json.dumps({"cmd": "stop"}))
                                break
                            live.update(_render())

                        elif t == "final":
                            seg_text = data.get("text", "").strip()
                            if seg_text:
                                confirmed.append(seg_text)
                            partial_buf = ""
                            live.update(_render())

                except websockets.exceptions.ConnectionClosed:
                    pass

            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

    except Exception as exc:
        console.print(f"[red]Erro WebSocket: {exc}[/red]")

    full_text = " ".join(confirmed).strip()
    if full_text:
        console.print(Panel(full_text, title="✅ Transcrição final", border_style="green"))
        print(full_text)
        if out:
            try:
                from pathlib import Path
                Path(out).write_text(full_text, encoding="utf-8")
                console.print(f"[dim]💾 Salvo em: {out}[/dim]")
            except Exception as exc:
                console.print(f"[red]Erro ao salvar arquivo: {exc}[/red]")

        if clip:
            try:
                import subprocess as _sp
                _sp.run(["wl-copy", full_text], check=True, timeout=5)
                console.print("[dim]Copiado para o clipboard.[/dim]")
            except FileNotFoundError:
                console.print("[yellow]wl-copy não encontrado.[/yellow]")
            except Exception:
                pass
    else:
        console.print("[yellow]Nenhum texto transcrito.[/yellow]")


@cli.command()
@click.option("--language", "-l", default="en", show_default=True,
              help="Código do idioma (padrão: en para simulação de inglês)")
@click.option("--silence", "-s", default=2.5, type=float, show_default=True,
              help="Segundos de silêncio para detectar fim do turno")
@click.option("--threshold", "-t", default=0.01, type=float, show_default=True,
              help="Nível RMS mínimo para considerar como voz")
@click.option("--server", default=None,
              help="URL base do servidor HTTP STT (padrão: http://localhost:8000)")
@click.option("--model", default=None,
              help="Modelo do LLM a ser utilizado (ex: gpt-4o, llama-3)")
@click.option("--out", default="outputs/interviews/session.md", type=click.Path(dir_okay=False, writable=True),
              help="Salvar o relatório final em Markdown")
def interview(language, silence, threshold, server, model, out):
    """Simulador interativo de entrevistas profissionais com feedback via LLM (Full-Duplex TTS)."""
    import asyncio
    try:
        asyncio.run(_interview_async(language, silence, threshold, server, model, out))
    except KeyboardInterrupt:
        pass

async def _interview_async(language, silence, threshold, server_url, model, out_path):
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.console import Console
    import os
    from pathlib import Path

    from .llm_client import LLMClient, InterviewSession, load_template
    from .tts_client import LocalTTSClient
    from .orchestrator import Orchestrator, Persona, Scenario, build_system_prompt

    console = Console()
    console.print(Panel.fit("[bold green]Iniciando Orchestrator Full-Duplex...[/]\n[dim]Pressione Ctrl+C para encerrar e gerar o relatório.[/]"))

    # Configura Endpoints e Dependências
    SERVER_HTTP_URL = "http://localhost:8000"
    base_url = (server_url or SERVER_HTTP_URL).rstrip("/")
    if base_url.startswith("ws://"):
        base_url = "http://" + base_url[5:]

    stt_endpoint = f"{base_url}/transcribe"

    try:
        llm = LLMClient(model=model)
        tts = LocalTTSClient() # Worker 3 local, apontando nativamente para $TTS_BASE_URL (http://localhost:8001)
    except Exception as e:
        console.print(f"[bold red]Erro ao inicializar Clients (LLM/TTS):[/] {e}")
        return

    # Injetando Camadas 1 e 2 dinâmicas (Dados da Sessão)
    # Por ora, chumbaremos a Sarah, mas poderá vir via argumento da CLI depois.
    persona_voice = os.getenv("SPOOKNIX_PERSONA_VOICE")
    if persona_voice and any(sep in persona_voice for sep in ("/", ".")):
        if not Path(persona_voice).exists():
            console.print(
                f"[yellow]Voice ref não encontrada em {persona_voice}. "
                "Usando a voz padrão do worker TTS.[/]"
            )
            persona_voice = None

    sarah = Persona(
        name="Sarah",
        system_prompt="You are a Senior Technical Recruiter. You ask precise behavioral and technical questions, challenging candidates gently but firmly. Do not be a sycophant.",
        voice_ref_audio=persona_voice,
        voice_ref_text="Hi, I am Sarah, your technical interviewer."
    )

    scenario = Scenario(
        interview_type="System Design",
        target_role="Senior Software Engineer",
        difficulty="Standard",
        duration_mins=15
    )

    # Constrói a sessão de Entrevista com base nas Camadas 1 e 2
    prompt = build_system_prompt(sarah, scenario)
    session = InterviewSession(prompt)

    # Executa a Camada 3 (Full-Duplex Engine)
    orchestrator = Orchestrator(llm=llm, tts=tts, stt_endpoint=stt_endpoint, language=language)

    await orchestrator.run_session(
        session=session,
        persona=sarah,
        silence_s=silence,
        threshold=threshold,
        model=model
    )

    # --- Relatório Final (Camada 5: Reflexão) ---
    transcript = session.get_transcript_text()
    if len(transcript.split()) < 10:
        console.print("[dim]Conversa muito curta. Relatório não gerado.[/]")
        return

    console.print("\n[dim]Gerando relatório de feedback da entrevista (Camada de Reflexão). Por favor, aguarde...[/]")
    try:
        evaluator_prompt = load_template("evaluator.md")
    except FileNotFoundError:
        # Se template falhar fallback inline
        evaluator_prompt = "You are an expert technical interview evaluator. Provide constructive feedback on the candidate's answers, grammar, and fluency based on the transcript."

    evaluator_session = InterviewSession(evaluator_prompt)
    evaluator_session.add_user_message(transcript)

    report = await llm.generate(evaluator_session.get_messages(), model)

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(report, encoding="utf-8")

    console.print(Panel(Markdown(report), title="Feedback da Entrevista", expand=False))
    console.print(f"\n[bold green]Relatório salvo em:[/] {out_path}")


if __name__ == "__main__":
    cli()
