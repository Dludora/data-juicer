"""
Base Transport - Abstract interface for all lineage transports.

All transports must implement the `emit(event)` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from data_juicer.core.lineage.events import LineageEvent


class BaseTransport(ABC):
    """Abstract base class for all lineage event transports."""

    @abstractmethod
    def emit(self, event: "LineageEvent") -> None:
        """Emit a lineage event to the transport backend.

        Args:
            event: The LineageEvent to emit.

        Raises:
            TransportError: If event emission fails.
        """

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseTransport":
        """Factory method to create transport from config dict.

        Subclasses should override this to handle their specific config keys.

        Args:
            config: Transport configuration dictionary.

        Returns:
            Configured transport instance.
        """
        return cls()


class TransportError(Exception):
    """Exception raised when transport emission fails."""

    def __init__(self, message: str, transport_type: str = "unknown", cause: Optional[Exception] = None):
        self.transport_type = transport_type
        self.cause = cause
        super().__init__(f"[{transport_type}] {message}")
