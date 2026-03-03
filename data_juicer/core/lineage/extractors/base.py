"""
Base Lineage Extractor - Abstract interface for operator-specific lineage extraction.

Each operator type can have a dedicated extractor that knows how to
extract meaningful lineage metadata from that operator's execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from data_juicer.core.lineage.facets import (
    BaseFacet,
    ColumnLineageFacet,
    DataQualityFacet,
    OpConfigFacet,
    PerformanceFacet,
    SchemaFacet,
)


@dataclass
class OperatorLineage:
    """Container for lineage metadata extracted from an operator execution.

    Analogous to Airflow's OperatorLineage(inputs, outputs, run_facets, job_facets).
    """

    input_facets: Dict[str, BaseFacet] = field(default_factory=dict)
    """Facets describing the input dataset (schema, quality, etc.)."""

    output_facets: Dict[str, BaseFacet] = field(default_factory=dict)
    """Facets describing the output dataset."""

    run_facets: Dict[str, BaseFacet] = field(default_factory=dict)
    """Facets describing the run (performance, error, column lineage, etc.)."""

    op_facets: Dict[str, BaseFacet] = field(default_factory=dict)
    """Facets describing the operator itself (config, etc.)."""


class BaseLineageExtractor(ABC):
    """Abstract base class for operator-specific lineage extraction.

    Subclasses implement extraction logic for specific operator types
    (Mapper, Filter, Deduplicator, etc.).
    """

    def extract_on_start(
        self,
        op: Any,
        columns_before: Set[str],
        row_count_before: Optional[int] = None,
    ) -> OperatorLineage:
        """Extract lineage metadata before operator execution.

        Called immediately before the operator processes data.

        Args:
            op: The operator instance.
            columns_before: Column names before execution.
            row_count_before: Row count before execution (may be None if expensive to compute).

        Returns:
            OperatorLineage with input_facets and op_facets populated.
        """
        lineage = OperatorLineage()

        # Schema facet
        lineage.input_facets["schema"] = SchemaFacet.from_column_set(columns_before)

        # Data quality facet
        if row_count_before is not None:
            lineage.input_facets["dataQuality"] = DataQualityFacet(
                row_count=row_count_before,
                column_count=len(columns_before),
            )

        # Op config facet
        lineage.op_facets["config"] = OpConfigFacet.from_op(op)

        return lineage

    def extract_on_complete(
        self,
        op: Any,
        columns_before: Set[str],
        columns_after: Set[str],
        row_count_before: Optional[int] = None,
        row_count_after: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> OperatorLineage:
        """Extract lineage metadata after successful operator execution.

        Called immediately after the operator finishes processing.

        Args:
            op: The operator instance.
            columns_before: Column names before execution.
            columns_after: Column names after execution.
            row_count_before: Row count before execution.
            row_count_after: Row count after execution.
            duration: Execution duration in seconds.

        Returns:
            OperatorLineage with output_facets and run_facets populated.
        """
        lineage = OperatorLineage()

        # Output schema facet
        lineage.output_facets["schema"] = SchemaFacet.from_column_set(columns_after)

        # Output data quality
        if row_count_after is not None:
            lineage.output_facets["dataQuality"] = DataQualityFacet(
                row_count=row_count_after,
                column_count=len(columns_after),
            )

        # Column lineage facet
        lineage.run_facets["columnLineage"] = ColumnLineageFacet.from_column_diff(
            columns_before=columns_before,
            columns_after=columns_after,
        )

        # Performance facet
        if row_count_before is not None and row_count_after is not None and duration is not None:
            lineage.run_facets["performance"] = PerformanceFacet.from_row_counts(
                input_rows=row_count_before,
                output_rows=row_count_after,
                duration=duration,
            )

        return lineage

    def extract_on_fail(
        self,
        op: Any,
        error: BaseException,
    ) -> OperatorLineage:
        """Extract lineage metadata when an operator fails.

        Args:
            op: The operator instance.
            error: The exception that caused the failure.

        Returns:
            OperatorLineage with error facets populated.
        """
        from data_juicer.core.lineage.facets import ErrorFacet

        lineage = OperatorLineage()
        lineage.run_facets["error"] = ErrorFacet.from_exception(error)
        lineage.op_facets["config"] = OpConfigFacet.from_op(op)
        return lineage
