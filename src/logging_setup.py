"""Central logging configuration for Spooknix.

Provides a single entry point `configure_logging(level)` used by CLI commands
and long-running services (server, orchestrator). All loggers under the `src.*`
namespace inherit this configuration.

Format: `HH:MM:SS.mmm [LEVEL] logger.name | message`
Output: stderr (Rich's console is also on stderr — they coexist cleanly).
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FMT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """Configure the root logger for Spooknix.

    Idempotent: subsequent calls are no-ops unless `force=True`.

    Args:
        level: logging level (int or string like "DEBUG", "INFO").
        force: reapply configuration even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    env_override = os.getenv("SPOOKNIX_LOG_LEVEL")
    if env_override:
        level = logging.getLevelName(env_override.upper())

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(handler)

    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name. Convenience wrapper."""
    return logging.getLogger(name)
