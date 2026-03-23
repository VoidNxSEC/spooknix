"""Spooknix GUI — systray leve para STT/TTS sem abrir terminal.

Requer: PyQt6
Variáveis de ambiente:
  SPOOKNIX_URL : URL base do servidor (padrão: http://localhost:8000)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import (
    QByteArray,
    QPropertyAnimation,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtNetwork import QHttpMultiPart, QHttpPart, QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

SERVER_URL = os.getenv("SPOOKNIX_URL", "http://localhost:8000")

# ── Ícone SVG inline (microfone) ──────────────────────────────────────────────

_MIC_SVG_ACTIVE = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#89b4fa">
  <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
  <path d="M19 10a7 7 0 0 1-14 0H3a9 9 0 0 0 8 8.94V21H9v2h6v-2h-2v-2.06A9 9 0 0 0 21 10h-2z"/>
</svg>"""

_MIC_SVG_INACTIVE = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#585b70">
  <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4z"/>
  <path d="M19 10a7 7 0 0 1-14 0H3a9 9 0 0 0 8 8.94V21H9v2h6v-2h-2v-2.06A9 9 0 0 0 21 10h-2z"/>
</svg>"""


def _svg_to_icon(svg_bytes: bytes, size: int = 64) -> QIcon:
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    painter = QPainter(px)
    from PyQt6.QtSvg import QSvgRenderer
    renderer = QSvgRenderer(QByteArray(svg_bytes))
    renderer.render(painter)
    painter.end()
    return QIcon(px)


# ── Thread de gravação ────────────────────────────────────────────────────────

class RecordThread(QThread):
    """Grava do microfone em background; nunca toca em widgets Qt diretamente."""

    recording_finished = pyqtSignal(str)   # emite caminho do WAV temporário
    recording_failed = pyqtSignal(str)     # emite mensagem de erro

    def run(self) -> None:
        try:
            from .recorder import record_until_silence
            wav_path = record_until_silence()
            self.recording_finished.emit(wav_path)
        except Exception as exc:  # noqa: BLE001
            self.recording_failed.emit(str(exc))


# ── Janela principal ──────────────────────────────────────────────────────────

class SpooknixWindow(QMainWindow):
    """Janela compacta 380×480, frameless, com fade-in/out."""

    def __init__(self, tray: "SpooknixTray") -> None:
        super().__init__()
        self._tray = tray
        self._nam = QNetworkAccessManager(self)
        self._active_reply: QNetworkReply | None = None
        self._pending_file: Path | None = None
        self._record_thread: RecordThread | None = None
        self._pending_recording_path: str | None = None

        self._record_blink_timer = QTimer(self)
        self._record_blink_timer.setInterval(500)
        self._record_blink_timer.timeout.connect(self._blink_record_button)
        self._record_blink_state = False

        self.setWindowTitle("Spooknix")
        self.setFixedSize(380, 480)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(0.0)

        self._build_ui()

        # Animação fade in/out
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(200)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root.setStyleSheet("""
            QWidget#root {
                background: rgba(30, 30, 46, 0.96);
                border-radius: 12px;
                border: 1px solid rgba(137, 180, 250, 0.3);
            }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QPushButton {
                background: rgba(137, 180, 250, 0.25);
                color: #89b4fa;
                border: 1px solid rgba(137, 180, 250, 0.4);
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(137, 180, 250, 0.4); }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
            QTextEdit {
                background: rgba(17, 17, 27, 0.9);
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 6px;
                font-size: 12px;
                padding: 6px;
            }
            QProgressBar {
                background: #313244;
                border-radius: 4px;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
        """)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Cabeçalho
        header = QLabel("🎙 Spooknix STT")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(header)

        # Status do servidor
        self._status_label = QLabel("● Verificando servidor…")
        self._status_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        layout.addWidget(self._status_label)

        # Drop zone / seleção de arquivo
        self._drop_label = QLabel("Clique para selecionar ou arraste um arquivo")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_label.setMinimumHeight(80)
        self._drop_label.setStyleSheet("""
            QLabel {
                color: #6c7086;
                border: 2px dashed #45475a;
                border-radius: 8px;
                font-size: 12px;
            }
        """)
        self._drop_label.setAcceptDrops(True)
        self._drop_label.mousePressEvent = lambda _: self._pick_file()
        layout.addWidget(self._drop_label)

        # Botão transcrever
        self._btn_transcribe = QPushButton("Transcrever")
        self._btn_transcribe.setEnabled(False)
        self._btn_transcribe.clicked.connect(self._do_transcribe)
        layout.addWidget(self._btn_transcribe)

        # Botão gravar
        self._btn_record = QPushButton("🎙 Gravar")
        self._btn_record.setStyleSheet("""
            QPushButton {
                background: rgba(166, 227, 161, 0.2);
                color: #a6e3a1;
                border: 1px solid rgba(166, 227, 161, 0.4);
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(166, 227, 161, 0.35); }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
        """)
        self._btn_record.clicked.connect(self._do_record)
        layout.addWidget(self._btn_record)

        # Barra de progresso (oculta por padrão)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        layout.addWidget(self._progress)

        # Resultado
        result_label = QLabel("Resultado:")
        layout.addWidget(result_label)

        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setPlaceholderText("O texto transcrito aparecerá aqui…")
        layout.addWidget(self._result_text)

        # Botão fechar
        btn_close = QPushButton("Fechar")
        btn_close.setStyleSheet("""
            QPushButton {
                background: rgba(243, 139, 168, 0.15);
                color: #f38ba8;
                border: 1px solid rgba(243, 139, 168, 0.3);
            }
            QPushButton:hover { background: rgba(243, 139, 168, 0.3); }
        """)
        btn_close.clicked.connect(self.toggle_visibility)
        layout.addWidget(btn_close)

        self.setCentralWidget(root)

    # ── Visibilidade com fade ─────────────────────────────────────────────────

    def toggle_visibility(self) -> None:
        if self.isVisible() and self.windowOpacity() > 0.5:
            self._fade_out()
        else:
            self._fade_in()

    def _fade_in(self) -> None:
        self.show()
        self._center_on_screen()
        self._anim.stop()
        self._anim.setStartValue(self.windowOpacity())
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _fade_out(self) -> None:
        self._anim.stop()
        self._anim.setStartValue(self.windowOpacity())
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._on_fade_out_done)
        self._anim.start()

    @pyqtSlot()
    def _on_fade_out_done(self) -> None:
        self._anim.finished.disconnect(self._on_fade_out_done)
        self.hide()

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - self.width()) // 2,
                geo.y() + (geo.height() - self.height()) // 2,
            )

    # ── Interação arquivo ─────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar arquivo de áudio/vídeo",
            str(Path.home()),
            "Áudio/Vídeo (*.wav *.mp3 *.mp4 *.ogg *.flac *.m4a *.webm);;Todos (*)",
        )
        if path:
            self._set_file(Path(path))

    def _set_file(self, path: Path) -> None:
        self._pending_file = path
        name = path.name
        if len(name) > 40:
            name = name[:37] + "…"
        self._drop_label.setText(f"📄 {name}")
        self._drop_label.setStyleSheet("""
            QLabel {
                color: #89b4fa;
                border: 2px dashed #89b4fa;
                border-radius: 8px;
                font-size: 12px;
            }
        """)
        self._btn_transcribe.setEnabled(True)

    # ── Transcrição ───────────────────────────────────────────────────────────

    @pyqtSlot()
    def _do_transcribe(self) -> None:
        if self._pending_file is None:
            return

        self._btn_transcribe.setEnabled(False)
        self._progress.show()
        self._result_text.setPlaceholderText("Transcrevendo…")

        url = QUrl(f"{SERVER_URL}/transcribe")
        request = QNetworkRequest(url)

        multipart = QHttpMultiPart(QHttpMultiPart.ContentType.FormDataType)

        file_part = QHttpPart()
        file_part.setHeader(
            QNetworkRequest.KnownHeaders.ContentDispositionHeader,
            f'form-data; name="file"; filename="{self._pending_file.name}"',
        )
        data = self._pending_file.read_bytes()
        file_part.setBody(data)
        multipart.append(file_part)

        lang_part = QHttpPart()
        lang_part.setHeader(
            QNetworkRequest.KnownHeaders.ContentDispositionHeader,
            'form-data; name="language"',
        )
        lang_part.setBody(b"pt")
        multipart.append(lang_part)

        reply = self._nam.post(request, multipart)
        multipart.setParent(reply)
        self._active_reply = reply
        reply.finished.connect(lambda: self._on_transcribe_done(reply))

    def _on_transcribe_done(self, reply: QNetworkReply) -> None:
        self._progress.hide()
        self._btn_transcribe.setEnabled(True)
        self._btn_record.setEnabled(True)
        self._active_reply = None

        # Limpar WAV temporário da gravação, se houver
        if self._pending_recording_path:
            Path(self._pending_recording_path).unlink(missing_ok=True)
            self._pending_recording_path = None

        if reply.error() != QNetworkReply.NetworkError.NoError:
            self._result_text.setPlainText(f"Erro: {reply.errorString()}")
            reply.deleteLater()
            return

        import json
        raw = bytes(reply.readAll())
        try:
            data = json.loads(raw)
            text = data.get("text", "").strip()
            self._result_text.setPlainText(text or "(sem texto transcrito)")
        except Exception as exc:
            self._result_text.setPlainText(f"Erro ao parsear resposta: {exc}")
        reply.deleteLater()

    # ── Gravação ──────────────────────────────────────────────────────────────

    @pyqtSlot()
    def _do_record(self) -> None:
        self._btn_transcribe.setEnabled(False)
        self._btn_record.setEnabled(False)
        self._record_blink_state = False
        self._record_blink_timer.start()

        self._record_thread = RecordThread(self)
        self._record_thread.recording_finished.connect(self._on_recording_done)
        self._record_thread.recording_failed.connect(self._on_recording_error)
        self._record_thread.start()

    @pyqtSlot()
    def _blink_record_button(self) -> None:
        self._record_blink_state = not self._record_blink_state
        if self._record_blink_state:
            self._btn_record.setStyleSheet("""
                QPushButton {
                    background: rgba(243, 139, 168, 0.6);
                    color: #ffffff;
                    border: 1px solid rgba(243, 139, 168, 0.8);
                    border-radius: 6px;
                    padding: 6px 14px;
                    font-size: 13px;
                }
            """)
        else:
            self._btn_record.setStyleSheet("""
                QPushButton {
                    background: rgba(243, 139, 168, 0.25);
                    color: #f38ba8;
                    border: 1px solid rgba(243, 139, 168, 0.4);
                    border-radius: 6px;
                    padding: 6px 14px;
                    font-size: 13px;
                }
            """)

    @pyqtSlot(str)
    def _on_recording_done(self, wav_path: str) -> None:
        self._record_blink_timer.stop()
        self._btn_record.setText("🎙 Gravar")
        self._btn_record.setStyleSheet("""
            QPushButton {
                background: rgba(166, 227, 161, 0.2);
                color: #a6e3a1;
                border: 1px solid rgba(166, 227, 161, 0.4);
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(166, 227, 161, 0.35); }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
        """)
        self._pending_recording_path = wav_path
        self._pending_file = Path(wav_path)
        self._do_transcribe()

    @pyqtSlot(str)
    def _on_recording_error(self, msg: str) -> None:
        self._record_blink_timer.stop()
        self._btn_record.setText("🎙 Gravar")
        self._btn_record.setStyleSheet("""
            QPushButton {
                background: rgba(166, 227, 161, 0.2);
                color: #a6e3a1;
                border: 1px solid rgba(166, 227, 161, 0.4);
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
            }
            QPushButton:hover { background: rgba(166, 227, 161, 0.35); }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
        """)
        self._result_text.setPlainText(f"Erro de gravação: {msg}")
        self._btn_transcribe.setEnabled(self._pending_file is not None)
        self._btn_record.setEnabled(True)

    # ── Atualização de status ─────────────────────────────────────────────────

    def update_server_status(self, online: bool) -> None:
        if online:
            self._status_label.setText("● Servidor online")
            self._status_label.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        else:
            self._status_label.setText("● Servidor offline")
            self._status_label.setStyleSheet("color: #f38ba8; font-size: 11px;")


# ── Systray ───────────────────────────────────────────────────────────────────

class SpooknixTray(QSystemTrayIcon):
    """Systray icon com health check periódico."""

    def __init__(self, app: QApplication) -> None:
        self._icon_active = _svg_to_icon(_MIC_SVG_ACTIVE)
        self._icon_inactive = _svg_to_icon(_MIC_SVG_INACTIVE)
        super().__init__(self._icon_inactive)

        self._app = app
        self._window = SpooknixWindow(self)
        self._nam = QNetworkAccessManager()
        self._server_online = False

        self._build_menu()
        self.activated.connect(self._on_tray_activated)

        # Health check a cada 10s
        self._timer = QTimer()
        self._timer.timeout.connect(self._check_health)
        self._timer.start(10_000)
        self._check_health()

        self.setToolTip("Spooknix STT")
        self.show()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background: rgba(30, 30, 46, 0.97);
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item { padding: 6px 20px; border-radius: 4px; }
            QMenu::item:selected { background: rgba(137, 180, 250, 0.2); }
            QMenu::separator { height: 1px; background: #45475a; margin: 4px 0; }
        """)

        act_open = menu.addAction("🎙  Abrir Spooknix")
        act_open.triggered.connect(self._window.toggle_visibility)

        self._act_status = menu.addAction("○  Verificando…")
        self._act_status.setEnabled(False)

        menu.addSeparator()

        act_quit = menu.addAction("✕  Sair")
        act_quit.triggered.connect(self._app.quit)

        self.setContextMenu(menu)

    # ── Eventos ───────────────────────────────────────────────────────────────

    @pyqtSlot(QSystemTrayIcon.ActivationReason)
    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._window.toggle_visibility()

    # ── Health check ──────────────────────────────────────────────────────────

    @pyqtSlot()
    def _check_health(self) -> None:
        request = QNetworkRequest(QUrl(f"{SERVER_URL}/health"))
        reply = self._nam.get(request)
        reply.finished.connect(lambda: self._on_health_done(reply))

    def _on_health_done(self, reply: QNetworkReply) -> None:
        online = reply.error() == QNetworkReply.NetworkError.NoError
        reply.deleteLater()

        if online == self._server_online:
            return  # sem mudança

        self._server_online = online
        self.setIcon(self._icon_active if online else self._icon_inactive)
        self._window.update_server_status(online)

        if online:
            self._act_status.setText("●  Servidor online")
            self.showMessage("Spooknix", "Servidor STT online", self._icon_active, 2000)
        else:
            self._act_status.setText("○  Servidor offline")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Spooknix")
    app.setQuitOnLastWindowClosed(False)  # manter rodando no systray

    if not QSystemTrayIcon.isSystemTrayAvailable():
        print("Systray não disponível neste sistema.", file=sys.stderr)
        sys.exit(1)

    _tray = SpooknixTray(app)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
