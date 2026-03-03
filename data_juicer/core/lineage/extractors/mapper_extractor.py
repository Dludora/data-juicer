"""
Mapper Lineage Extractor - Enhanced extractor for Mapper operators.

Captures additional information specific to Mapper operations:
- Primary keys (text_key, image_key, etc.) as read columns
- Detailed column-level read/write tracking
"""

from __future__ import annotations

from typing import Any, Optional, Set

from data_juicer.core.lineage.extractors.base import (
    BaseLineageExtractor,
    OperatorLineage,
)
from data_juicer.core.lineage.facets import (
    ColumnLineageFacet,
    OpConfigFacet,
    PerformanceFacet,
    SchemaFacet,
    DataQualityFacet,
)


class MapperLineageExtractor(BaseLineageExtractor):
    """Extractor for Mapper operators.

    Enhancements over default:
    - On start: adds primary key columns to input facets
    - On complete: builds column lineage with read columns from op keys
    """

    @staticmethod
    def _get_op_key_columns(op: Any) -> Set[str]:
        """Extract the key columns that a mapper reads from."""
        key_cols = set()
        for attr in ('text_key', 'image_key', 'audio_key', 'video_key'):
            val = getattr(op, attr, None)
            if val:
                key_cols.add(val)
        return key_cols

    def extract_on_start(
        self,
        op: Any,
        columns_before: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
    ) -> OperatorLineage:
        """Extract lineage at mapper start.

        Adds key columns info to the op facets.
        """
        lineage = super().extract_on_start(op, columns_before, row_count_before)

        # Record which columns this mapper reads
        key_cols = self._get_op_key_columns(op)
        if key_cols:
            lineage.op_facets['mapper_read_keys'] = SchemaFacet(
                columns=[{'name': c, 'type': 'key'} for c in sorted(key_cols)]
            )

        return lineage

    def extract_on_complete(
        self,
        op: Any,
        columns_before: Optional[Set[str]] = None,
        columns_after: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
        row_count_after: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> OperatorLineage:
        """Extract lineage at mapper completion.

        Builds column lineage with explicit read columns from key fields.
        """
        lineage = super().extract_on_complete(
            op, columns_before, columns_after,
            row_count_before, row_count_after, duration,
        )

        # Enhance column lineage with read keys
        if columns_before is not None and columns_after is not None:
            key_cols = self._get_op_key_columns(op)
            added = columns_after - columns_before
            deleted = columns_before - columns_after

            # For mappers, the read columns are: key columns + deleted columns
            # (they read from source columns to produce new ones)
            columns_read = key_cols | deleted
            # Write columns are the newly added ones
            columns_written = added

            lineage.run_facets['columnLineage'] = ColumnLineageFacet(
                columns_read=sorted(columns_read),
                columns_written=sorted(columns_written),
                columns_deleted=sorted(deleted),
                schema_before=SchemaFacet.from_column_set(columns_before),
                schema_after=SchemaFacet.from_column_set(columns_after),
            )

        return lineage
