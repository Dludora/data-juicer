"""
Lineage Facets - Extensible metadata descriptors for data lineage.

Inspired by OpenLineage Facet design. Each Facet is an independent,
JSON-serializable metadata block that can be attached to LineageEvents.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseFacet:
    """Base class for all lineage facets.

    All facets must be JSON-serializable via `to_dict()`.
    """

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/v1", init=False, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize facet to a JSON-compatible dictionary."""
        d = asdict(self)
        d.pop("_schema_url", None)
        d["_schemaURL"] = self._schema_url
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseFacet":
        """Deserialize from dict. Subclasses can override for custom logic."""
        data = {k: v for k, v in data.items() if k != "_schemaURL"}
        return cls(**data)


@dataclass
class SchemaFacet(BaseFacet):
    """Describes the schema (columns and types) of a dataset.

    Used for both input and output dataset descriptions.
    """

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/schema/v1", init=False, repr=False)

    columns: List[Dict[str, str]] = field(default_factory=list)
    """List of column descriptors, each with 'name' and 'type' keys."""

    @classmethod
    def from_column_set(cls, columns: set, type_map: Optional[Dict[str, str]] = None) -> "SchemaFacet":
        """Create SchemaFacet from a column set and optional type mapping.

        Args:
            columns: Set of column names.
            type_map: Optional mapping of column name -> type string.
                      If not provided, type defaults to 'unknown'.
        """
        type_map = type_map or {}
        col_list = [{"name": col, "type": type_map.get(col, "unknown")} for col in sorted(columns)]
        return cls(columns=col_list)

    @property
    def column_names(self) -> set:
        """Get the set of column names."""
        return {col["name"] for col in self.columns}


@dataclass
class ColumnLineageFacet(BaseFacet):
    """Tracks column-level changes performed by an operator.

    Records which columns were read, written (created/modified),
    and deleted by an operator execution.
    """

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/column_lineage/v1", init=False, repr=False)

    columns_read: List[str] = field(default_factory=list)
    """Columns read/accessed by the operator."""

    columns_written: List[str] = field(default_factory=list)
    """Columns created or modified by the operator."""

    columns_deleted: List[str] = field(default_factory=list)
    """Columns removed by the operator."""

    schema_before: Optional[List[Dict[str, str]]] = None
    """Full schema snapshot before operator execution."""

    schema_after: Optional[List[Dict[str, str]]] = None
    """Full schema snapshot after operator execution."""

    @classmethod
    def from_column_diff(
        cls,
        columns_before: set,
        columns_after: set,
        columns_read: Optional[set] = None,
    ) -> "ColumnLineageFacet":
        """Create ColumnLineageFacet from before/after column sets.

        Args:
            columns_before: Column set before operator execution.
            columns_after: Column set after operator execution.
            columns_read: Optional explicit set of columns read.
                         If None, assumed to be columns_before.
        """
        added = columns_after - columns_before
        deleted = columns_before - columns_after
        read = sorted(columns_read) if columns_read else sorted(columns_before)

        return cls(
            columns_read=read,
            columns_written=sorted(added),
            columns_deleted=sorted(deleted),
            schema_before=[{"name": c, "type": "unknown"} for c in sorted(columns_before)],
            schema_after=[{"name": c, "type": "unknown"} for c in sorted(columns_after)],
        )


@dataclass
class DataQualityFacet(BaseFacet):
    """Records data quality metrics for a dataset snapshot.

    Attached to input/output dataset descriptors.
    """

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/data_quality/v1", init=False, repr=False)

    row_count: Optional[int] = None
    """Number of rows in the dataset."""

    column_count: Optional[int] = None
    """Number of columns in the dataset."""

    bytes_size: Optional[int] = None
    """Approximate dataset size in bytes."""


@dataclass
class PerformanceFacet(BaseFacet):
    """Records performance metrics for an operator or pipeline execution."""

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/performance/v1", init=False, repr=False)

    duration_seconds: Optional[float] = None
    """Wall-clock execution time in seconds."""

    throughput_rows_per_second: Optional[float] = None
    """Processing throughput."""

    input_rows: Optional[int] = None
    """Number of input rows."""

    output_rows: Optional[int] = None
    """Number of output rows."""

    reduction_ratio: Optional[float] = None
    """Ratio of rows removed: (input - output) / input. 0.0 means no change."""

    @classmethod
    def from_row_counts(
        cls,
        input_rows: int,
        output_rows: int,
        duration: float,
    ) -> "PerformanceFacet":
        """Create PerformanceFacet from row counts and duration."""
        throughput = input_rows / duration if duration > 0 else None
        reduction = (input_rows - output_rows) / input_rows if input_rows > 0 else 0.0
        return cls(
            duration_seconds=duration,
            throughput_rows_per_second=throughput,
            input_rows=input_rows,
            output_rows=output_rows,
            reduction_ratio=reduction,
        )


@dataclass
class ErrorFacet(BaseFacet):
    """Records error information when an operator or pipeline fails."""

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/error/v1", init=False, repr=False)

    error_type: Optional[str] = None
    """Exception class name."""

    error_message: Optional[str] = None
    """Error description."""

    stack_trace: Optional[str] = None
    """Full stack trace."""

    @classmethod
    def from_exception(cls, exc: BaseException) -> "ErrorFacet":
        """Create ErrorFacet from an exception."""
        import traceback as tb

        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            stack_trace=tb.format_exc(),
        )


@dataclass
class OpConfigFacet(BaseFacet):
    """Records operator configuration parameters."""

    _schema_url: str = field(default="https://datajuicer.io/lineage/facets/op_config/v1", init=False, repr=False)

    op_name: Optional[str] = None
    """Operator registry name (e.g., 'language_id_score_filter')."""

    op_type: Optional[str] = None
    """Operator base type (e.g., 'Mapper', 'Filter', 'Deduplicator')."""

    op_params: Dict[str, Any] = field(default_factory=dict)
    """Operator configuration parameters."""

    batch_size: Optional[int] = None
    """Processing batch size."""

    num_proc: Optional[int] = None
    """Number of parallel processes/workers."""

    num_gpus: Optional[float] = None
    """Number of GPUs used."""

    @classmethod
    def from_op(cls, op) -> "OpConfigFacet":
        """Create OpConfigFacet from an operator instance.

        Args:
            op: An OP instance from data_juicer.ops.base_op.
        """
        # Determine the op_type from class hierarchy
        from data_juicer.ops.base_op import (
            Deduplicator,
            Filter,
            Mapper,
            Selector,
        )

        if isinstance(op, Mapper):
            op_type = "Mapper"
        elif isinstance(op, Filter):
            op_type = "Filter"
        elif isinstance(op, Deduplicator):
            op_type = "Deduplicator"
        elif isinstance(op, Selector):
            op_type = "Selector"
        else:
            op_type = type(op).__name__

        # Extract safe-to-serialize params
        params = {}
        if hasattr(op, "_op_cfg") and isinstance(op._op_cfg, dict):
            params = {k: _safe_serialize(v) for k, v in op._op_cfg.items()}

        return cls(
            op_name=getattr(op, "_name", None),
            op_type=op_type,
            op_params=params,
            batch_size=getattr(op, "batch_size", None),
            num_proc=getattr(op, "num_proc", None),
            num_gpus=getattr(op, "num_gpus", None),
        )


def _safe_serialize(value: Any) -> Any:
    """Convert a value to a JSON-safe representation."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    elif isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    elif isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    else:
        return str(value)
