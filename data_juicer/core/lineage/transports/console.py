"""
Console Transport - Emits lineage events to the logger.

Simple transport for debugging and development.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

from data_juicer.core.lineage.transports.base import BaseTransport

if TYPE_CHECKING:
    from data_juicer.core.lineage.events import LineageEvent


class ConsoleTransport(BaseTransport):
    """Emits lineage events to loguru logger.

    Useful for debugging and development. No external dependencies.
    """

    def emit(self, event: "LineageEvent") -> None:
        """Log event as JSON to loguru."""
        event_dict = event.to_dict()
        event_json = json.dumps(event_dict, ensure_ascii=False, default=str)
        logger.info(f"LINEAGE_EVENT | {event_json}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConsoleTransport":
        return cls()
