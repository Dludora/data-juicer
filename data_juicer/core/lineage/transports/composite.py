"""
Composite Transport - Combines multiple transports for simultaneous emission.

Inspired by OpenLineage CompositeTransport. Events are delivered sequentially
to each sub-transport with configurable failure/success behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from loguru import logger

from data_juicer.core.lineage.transports.base import BaseTransport, TransportError

if TYPE_CHECKING:
    from data_juicer.core.lineage.events import LineageEvent


class CompositeTransport(BaseTransport):
    """Combines multiple transports for simultaneous event emission.

    Config:
        - transports: Dict of named transport configs.
        - continue_on_failure: Continue if a sub-transport fails. Default: True.
        - continue_on_success: Continue after a sub-transport succeeds. Default: True.
        - sort_transports: Sort by priority before emission. Default: False.
    """

    def __init__(
        self,
        transports: List[BaseTransport],
        continue_on_failure: bool = True,
        continue_on_success: bool = True,
    ):
        self._transports = transports
        self._continue_on_failure = continue_on_failure
        self._continue_on_success = continue_on_success

    def emit(self, event: "LineageEvent") -> None:
        """Emit event to all sub-transports sequentially."""
        for transport in self._transports:
            try:
                transport.emit(event)
                if not self._continue_on_success:
                    break
            except Exception as e:
                logger.warning(f"Composite sub-transport {type(transport).__name__} failed: {e}")
                if not self._continue_on_failure:
                    raise TransportError(
                        f"Sub-transport {type(transport).__name__} failed: {e}",
                        transport_type="composite",
                        cause=e,
                    )

    def close(self) -> None:
        """Close all sub-transports."""
        for transport in self._transports:
            try:
                transport.close()
            except Exception as e:
                logger.warning(f"Error closing sub-transport {type(transport).__name__}: {e}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompositeTransport":
        """Create CompositeTransport from config dict.

        Args:
            config: Dict with keys:
                - transports: Dict[str, dict] of named sub-transport configs.
                - continue_on_failure: bool. Default True.
                - continue_on_success: bool. Default True.
                - sort_transports: bool. Default False.
        """
        from data_juicer.core.lineage.transports.factory import create_transport

        transports_config = config.get("transports", {})
        if not transports_config:
            raise ValueError("CompositeTransport requires 'transports' config with at least one sub-transport")

        sort_by_priority = config.get("sort_transports", False)

        # Build sub-transports
        sub_configs = []
        for name, sub_config in transports_config.items():
            priority = sub_config.get("priority", 0)
            sub_configs.append((name, priority, sub_config))

        if sort_by_priority:
            sub_configs.sort(key=lambda x: x[1], reverse=True)

        sub_transports = []
        for name, priority, sub_config in sub_configs:
            try:
                transport = create_transport(sub_config)
                sub_transports.append(transport)
                logger.debug(f"Composite: initialized sub-transport '{name}' (priority={priority})")
            except Exception as e:
                logger.warning(f"Composite: failed to initialize sub-transport '{name}': {e}")

        return cls(
            transports=sub_transports,
            continue_on_failure=config.get("continue_on_failure", True),
            continue_on_success=config.get("continue_on_success", True),
        )
