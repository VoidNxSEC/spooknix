from __future__ import annotations

"""
Camada 3 — Orquestrador do Spooknix Conversational Suite.
Gerencia a máquina de estados, chunking de TTS e playback de áudio.
"""

import asyncio
import queue
import re
import threading
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import sounddevice as sd

from .llm_client import LLMClient, InterviewSession
from .tts_client import LocalTTSClient


# ── CAMADAS 1 & 2: Personas e Cenários ─────────────────────────────────────

@dataclass
class Persona:
    name: str
    system_prompt: str
    voice_ref_audio: str | None = None
    voice_ref_text: str | None = None

@dataclass
class Scenario:
    interview_type: str
    target_role: str
    difficulty: str
    duration_mins: int

def build_system_prompt(persona: Persona, scenario: Scenario) -> str:
    """Mescla Persona e Cenário no prompt de sistema (Camadas 1 e 2)."""
    return f"""
You are {persona.name}. {persona.system_prompt}

Context:
- Type: {scenario.interview_type}
- Target Role: {scenario.target_role}
- Difficulty: {scenario.difficulty}
- Duration Target: {scenario.duration_mins} minutes.

Guidelines:
1. Speak exclusively in English.
2. Ask ONE question at a time. Wait for the response.
3. Be conversational and natural. Keep responses concise.
4. Use short, speech-friendly sentences.
5. Use commas and ellipses sparingly to create natural pauses for TTS.
"""


# ── CAMADA 3: Playback de Áudio Assíncrono com Barge-in ───────────────────

class AsyncAudioPlayer:
    """
    Toca chunks de áudio gerados pelo TTS em uma thread separada para não
    bloquear o event loop e permitir interrupção imediata (barge-in).
    """
    def __init__(self, samplerate: int = 24000):
        self.samplerate = samplerate
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.stream: sd.OutputStream | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._finished_event = threading.Event()

    def _playback_worker(self):
        """Thread que consome a fila e escreve no OutputStream do PipeWire."""
        try:
            with sd.OutputStream(samplerate=self.samplerate, channels=1, dtype='float32') as stream:
                self.stream = stream
                while not self._stop_event.is_set():
                    try:
                        # Timeout permite checar o _stop_event periodicamente
                        chunk = self.queue.get(timeout=0.1)
                        if chunk is None: # Sentinel para finalizar suavemente
                            break
                        if not self._stop_event.is_set() and chunk.size:
                            stream.write(chunk)
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"\n[Playback Error] {e}")
        finally:
            self.stream = None
            self._finished_event.set()

    def start(self):
        self._stop_event.clear()
        self._finished_event.clear()
        # Esvazia fila antiga
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        self._thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._thread.start()

    def enqueue(self, audio_chunk: np.ndarray):
        """Adiciona PCM float32 na fila de reprodução."""
        if not self._stop_event.is_set():
            self.queue.put(audio_chunk)

    def finish(self):
        """Sinaliza que não haverá mais áudio e deixa a thread drenar a fila."""
        if not self._stop_event.is_set():
            self.queue.put(None)

    def wait_until_finished(self, timeout: float = 10.0):
        """Bloqueia até o playback encerrar ou o timeout expirar."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def stop_instantly(self):
        """Barge-in: Cala a boca do TTS imediatamente."""
        self._stop_event.set()
        # Esvazia fila
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.stream:
            self.stream.abort() # Para o hardware imediatamente (PipeWire)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ── CAMADA 3: Orquestrador State Machine ───────────────────────────────────

class State(Enum):
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()

class Orchestrator:
    def __init__(self, llm: LLMClient, tts: LocalTTSClient, stt_endpoint: str, language: str = "en"):
        self.llm = llm
        self.tts = tts
        self.player = AsyncAudioPlayer()
        self.state = State.LISTENING
        self.tts_tasks: set[asyncio.Task] = set()

        self.stt_endpoint = stt_endpoint
        self.language = language
        self.queue: asyncio.Queue[tuple[np.ndarray, bool]] = asyncio.Queue()
        self.loop = asyncio.get_running_loop()

    async def _tts_synthesize_and_play(self, text: str, persona: Persona):
        """Worker: Bate no TTS e empurra pra fila do Player."""
        if not text.strip():
            return

        wav_bytes = await self.tts.synthesize(
            text=text,
            voice=persona.voice_ref_audio or "alloy" # Mapeia o campo voice_ref_audio para voice da OpenAI
        )
        audio_float32, sr = self.tts.decode_wav(wav_bytes)


        # Ajusta samplerate do player sob demanda (normalmente F5 usa 24kHz)
        if self.player.samplerate != sr:
            self.player.samplerate = sr

        self.player.enqueue(audio_float32)

    async def stream_llm_to_tts(self, session: InterviewSession, persona: Persona, model: str | None = None):
        """Consome o LLM token a token, quebra em frases e manda pro TTS."""
        sentence_buffer = ""
        # Regex para identificar fim de sentença (+ lookahead pra não cortar abreviações comuns)
        split_pattern = re.compile(r'(?<=[.!?])\s+')

        self.player.start()
        full_reply = ""
        pending_tts_tasks: list[asyncio.Task] = []

        try:
            async for chunk in self.llm.chat_stream(session.get_messages(), model):
                if self.state != State.SPEAKING:
                    break # Barge-in aconteceu

                print(chunk, end="", flush=True)
                full_reply += chunk
                sentence_buffer += chunk

                # Se a buffer contém fim de frase, envia pro TTS
                splits = split_pattern.split(sentence_buffer)
                if len(splits) > 1:
                    complete_sentence = splits[0].strip()
                    if complete_sentence:
                        # Cria task pra requisição HTTP em background
                        task = asyncio.create_task(self._tts_synthesize_and_play(complete_sentence, persona))
                        pending_tts_tasks.append(task)
                        self.tts_tasks.add(task)
                        task.add_done_callback(self.tts_tasks.discard)

                    sentence_buffer = " ".join(splits[1:])

            # Envia o que sobrou no buffer
            if sentence_buffer.strip() and self.state == State.SPEAKING:
                task = asyncio.create_task(self._tts_synthesize_and_play(sentence_buffer.strip(), persona))
                pending_tts_tasks.append(task)
                self.tts_tasks.add(task)
                task.add_done_callback(self.tts_tasks.discard)

            if pending_tts_tasks:
                await asyncio.gather(*pending_tts_tasks, return_exceptions=True)

            # Aguarda o player terminar de tocar as ultimas frases antes de escutar de novo
            if self.state == State.SPEAKING:
                self.player.finish()
                await asyncio.to_thread(self.player.wait_until_finished)

        except asyncio.CancelledError:
            raise
        finally:
            if full_reply:
                session.add_assistant_message(full_reply.strip())

    def trigger_barge_in(self):
        """Acionado pela thread do microfone quando detecta voz no estado SPEAKING."""
        self.state = State.LISTENING
        # 1. Cala o hardware de som
        self.player.stop_instantly()
        # 2. Cancela requests TTS pendentes (aiohttp)
        for task in self.tts_tasks:
            task.cancel()
        self.tts_tasks.clear()

    # ── CAMADA 4: Worker STT ────────────────────────────────────────────────

    async def transcribe_audio(self, chunks: list[np.ndarray], sample_rate: int) -> str:
        """Envia o buffer de áudio (numpy chunks) para o servidor STT."""
        import aiohttp
        from .recorder import _chunks_to_wav_bytes

        wav_bytes = _chunks_to_wav_bytes(chunks, sample_rate)
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="turn.wav")
        form.add_field("language", self.language)
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(self.stt_endpoint, data=form, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("text", "").strip()
        except Exception as e:
            print(f"\n[STT Error] {e}")
        return ""

    # ── LOOP PRINCIPAL DO DIRETOR (Full-Duplex) ─────────────────────────────

    async def process_turn(self, session: InterviewSession, persona: Persona, chunks: list[np.ndarray], sample_rate: int, model: str | None = None):
        """Pipeline do Orquestrador: Transcreve (STT) -> Raciocina (LLM) -> Fala (TTS)"""
        from rich.console import Console
        console = Console()

        # 1. STT (Transcreve o turno completo)
        console.print("\n[dim italic]Transcrevendo...[/]", end="\r")
        text = await self.transcribe_audio(chunks, sample_rate)
        if not text:
            console.print("\x1b[2K[dim]Nenhuma fala detectada.[/]")
            self.state = State.LISTENING
            return

        console.print(f"\x1b[2K[green]Você:[/] {text}")
        session.add_user_message(text)

        # 2. LLM e TTS
        self.state = State.SPEAKING
        console.print(f"\n[bold blue]{persona.name}:[/] ", end="")

        try:
            await self.stream_llm_to_tts(session, persona, model)
        except asyncio.CancelledError:
            console.print(" [dim italic red](Cancelado)[/]")
            raise
        finally:
            print("\n")
            if self.state == State.SPEAKING:
                self.state = State.LISTENING
                console.print("[dim cyan](Ouvindo)[/]")

    async def run_session(self, session: InterviewSession, persona: Persona, silence_s: float = 2.5, threshold: float = 0.01, model: str | None = None):
        """Inicia a sessão, escutando do microfone e roteando para as pipelines."""
        from rich.console import Console
        console = Console()

        from .recorder import BLOCKSIZE, SAMPLE_RATE

        silent_chunks_needed = int(silence_s * SAMPLE_RATE / BLOCKSIZE)
        silent_count = 0
        has_spoken = False
        audio_buffer: list[np.ndarray] = []
        processing_task: asyncio.Task | None = None

        def audio_callback(indata: np.ndarray, frames: int, time_info, status):
            """Callback síncrono da PipeWire. Ouve o tempo todo."""
            chunk = indata[:, 0].copy().astype(np.float32)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_speech = rms >= threshold

            self.loop.call_soon_threadsafe(self.queue.put_nowait, (chunk, is_speech))

        # --- Início da Sessão (Apresentação Inicial) ---
        console.print(f"\n[bold blue]{persona.name}:[/] ", end="")
        session.add_user_message(f"Hi, I am ready for the mock interview.")
        self.state = State.SPEAKING
        try:
            await self.stream_llm_to_tts(session, persona, model)
        except Exception as e:
            console.print(f"\n[bold red]Erro na comunicação inicial:[/] {e}")
            return
        print("\n")
        self.state = State.LISTENING
        console.print("[dim cyan](Ouvindo)[/]")

        # --- Loop de Eventos do Áudio ---
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCKSIZE, callback=audio_callback):
                while True:
                    chunk, is_speech = await self.queue.get()

                    if self.state == State.LISTENING:
                        if is_speech:
                            audio_buffer.append(chunk)
                            has_spoken = True
                            silent_count = 0
                        elif has_spoken:
                            audio_buffer.append(chunk)
                            silent_count += 1

                            if silent_count > silent_chunks_needed:
                                self.state = State.PROCESSING
                                # Cria task assíncrona para não bloquear o loop de captação
                                processing_task = asyncio.create_task(
                                    self.process_turn(session, persona, audio_buffer, SAMPLE_RATE, model)
                                )
                                # Reseta buffer pro próximo turno
                                audio_buffer = []
                                has_spoken = False
                                silent_count = 0

                    elif self.state == State.SPEAKING:
                        # Barge-in: Usuário começou a falar enquanto a IA falava/tocava
                        if is_speech:
                            self.trigger_barge_in() # Corta o TTS e a fila de playback

                            if processing_task and not processing_task.done():
                                processing_task.cancel() # Interrompe LLM

                            console.print(" [dim italic red](Interrompido)[/]\n")

                            audio_buffer = [chunk]
                            has_spoken = True
                            silent_count = 0
                            self.state = State.LISTENING # Volta imediatamente pro loop de Listening

        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n\n[bold yellow]Sessão encerrada pelo usuário.[/]")
            if processing_task and not processing_task.done():
                processing_task.cancel()
            self.trigger_barge_in()
