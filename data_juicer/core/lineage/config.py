"""
Lineage Configuration.

Defines configuration dataclasses for the lineage module,
controlling granularity levels, transport settings, and filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GranularityConfig:
    """Controls which lineage granularity levels are enabled."""

    pipeline: bool = True
    """Level 0: Pipeline-level lineage (near-zero overhead)."""

    operator: bool = True
    """Level 1: Operator-level lineage (low overhead)."""

    column: bool = True
    """Level 2: Column-level lineage (low overhead, schema diff)."""

    sample: bool = False
    """Level 3: Sample-level lineage (has overhead, reuses Tracer)."""


@dataclass
class TransportConfig:
    """Configuration for a single lineage transport."""

    type: str = "file"
    """Transport type: 'console', 'file', 'http', 'composite'."""

    # File transport options
    log_file_path: Optional[str] = None
    """File path for file transport. Default: {work_dir}/lineage/events.jsonl."""

    append: bool = True
    """Whether to append to existing file (file transport)."""

    # HTTP transport options
    url: Optional[str] = None
    """Base URL for HTTP transport."""

    endpoint: str = "api/v1/lineage"
    """API endpoint path appended to url."""

    timeout: float = 5.0
    """HTTP request timeout in seconds."""

    verify: bool = True
    """Whether to verify TLS certificates."""

    auth: Optional[Dict[str, str]] = None
    """Authentication config: {'type': 'api_key', 'apiKey': '...'}."""

    compression: Optional[str] = None
    """HTTP compression: 'gzip' or None."""

    retry: Optional[Dict[str, Any]] = None
    """HTTP retry config: {'total': 5, 'backoff_factor': 0.3, ...}."""

    # Composite transport options
    transports: Optional[Dict[str, Any]] = None
    """Sub-transport configs for composite transport."""

    continue_on_failure: bool = True
    """Composite: continue if a sub-transport fails."""

    continue_on_success: bool = True
    """Composite: continue after a sub-transport succeeds."""

    sort_transports: bool = False
    """Composite: sort sub-transports by priority before emission."""

    # Priority (for composite sub-transports)
    priority: int = 0
    """Transport priority for composite ordering (higher = first)."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                result[k] = v
        return result


@dataclass
class LineageConfig:
    """Top-level lineage configuration."""

    enabled: bool = False
    """Master switch for lineage tracking."""

    granularity: GranularityConfig = field(default_factory=GranularityConfig)
    """Granularity level controls."""

    transport: TransportConfig = field(default_factory=TransportConfig)
    """Transport configuration for event emission."""

    namespace: str = "data-juicer"
    """Default namespace for jobs and datasets."""

    filters: List[Dict[str, str]] = field(default_factory=list)
    """Event filters: [{'type': 'exact', 'match': 'some_op'}, ...]."""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "LineageConfig":
        """Create LineageConfig from a dictionary (e.g., parsed from YAML).

        Args:
            data: Configuration dictionary. If None, returns default config.
        """
        if data is None:
            return cls()

        # Parse granularity
        gran_data = data.get("granularity", {})
        granularity = GranularityConfig(
            pipeline=gran_data.get("pipeline", True),
            operator=gran_data.get("operator", True),
            column=gran_data.get("column", True),
            sample=gran_data.get("sample", False),
        )

        # Parse transport
        transport_data = data.get("transport", {})
        transport = TransportConfig(
            type=transport_data.get("type", "file"),
            log_file_path=transport_data.get("log_file_path"),
            append=transport_data.get("append", True),
            url=transport_data.get("url"),
            endpoint=transport_data.get("endpoint", "api/v1/lineage"),
            timeout=transport_data.get("timeout", 5.0),
            verify=transport_data.get("verify", True),
            auth=transport_data.get("auth"),
            compression=transport_data.get("compression"),
            retry=transport_data.get("retry"),
            transports=transport_data.get("transports"),
            continue_on_failure=transport_data.get("continue_on_failure", True),
            continue_on_success=transport_data.get("continue_on_success", True),
            sort_transports=transport_data.get("sort_transports", False),
            priority=transport_data.get("priority", 0),
        )

        return cls(
            enabled=data.get("enabled", False),
            granularity=granularity,
            transport=transport,
            namespace=data.get("namespace", "data-juicer"),
            filters=data.get("filters", []),
        )
