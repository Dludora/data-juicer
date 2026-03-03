"""
File Transport - Emits lineage events to JSONL files.

Default transport for Data-Juicer. Zero external dependencies.
Supports local files and remote filesystems via fsspec (optional).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from data_juicer.core.lineage.transports.base import BaseTransport, TransportError

if TYPE_CHECKING:
    from data_juicer.core.lineage.events import LineageEvent


class FileTransport(BaseTransport):
    """Emits lineage events to JSONL files.

    By default, events are appended to a single JSONL file.
    If append=False, each event creates a timestamped file.

    Config:
        - log_file_path: Path to the JSONL file.
        - append: Whether to append (True) or create timestamped files (False).
    """

    def __init__(self, log_file_path: str, append: bool = True):
        self._log_file_path = log_file_path
        self._append = append
        self._lock = threading.Lock()
        self._file = None

        # Ensure directory exists
        parent = Path(log_file_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: "LineageEvent") -> None:
        """Write event as a JSON line to file."""
        try:
            event_dict = event.to_dict()
            event_json = json.dumps(event_dict, ensure_ascii=False, default=str)

            with self._lock:
                if self._append:
                    with open(self._log_file_path, "a", encoding="utf-8") as f:
                        f.write(event_json + "\n")
                        f.flush()
                else:
                    # Create timestamped file
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                    base = Path(self._log_file_path)
                    ts_path = base.parent / f"{base.stem}_{ts}{base.suffix}"
                    with open(ts_path, "w", encoding="utf-8") as f:
                        f.write(event_json + "\n")

        except Exception as e:
            raise TransportError(
                f"Failed to write event to {self._log_file_path}: {e}",
                transport_type="file",
                cause=e,
            )

    def close(self) -> None:
        """No-op for file transport (files are opened/closed per write)."""
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FileTransport":
        """Create FileTransport from config dict.

        Args:
            config: Dict with keys:
                - log_file_path (str): File path. Required.
                - append (bool): Append mode. Default True.
        """
        log_file_path = config.get("log_file_path")
        if not log_file_path:
            raise ValueError("FileTransport requires 'log_file_path' in config")
        append = config.get("append", True)
        return cls(log_file_path=log_file_path, append=append)
