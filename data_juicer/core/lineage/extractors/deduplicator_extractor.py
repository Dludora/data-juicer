"""
Deduplicator Lineage Extractor - Enhanced extractor for Deduplicator operators.

Captures additional information specific to deduplication operations:
- Dedup rate (fraction of duplicates removed)
- Dedup method information
"""

from __future__ import annotations

from typing import Any, Optional, Set

from data_juicer.core.lineage.extractors.base import (
    BaseLineageExtractor,
    OperatorLineage,
)
from data_juicer.core.lineage.facets import BaseFacet


class DedupStatsFacet(BaseFacet):
    """Facet capturing deduplication-specific statistics."""

    _schemaURL = 'https://datajuicer.com/schemas/dedup-stats-facet.json'

    def __init__(
        self,
        rows_before: int = 0,
        rows_after: int = 0,
        duplicates_removed: int = 0,
        dedup_rate: float = 0.0,
        dedup_method: Optional[str] = None,
    ):
        super().__init__()
        self.rows_before = rows_before
        self.rows_after = rows_after
        self.duplicates_removed = duplicates_removed
        self.dedup_rate = dedup_rate
        self.dedup_method = dedup_method

    @classmethod
    def from_row_counts(
        cls,
        rows_before: int,
        rows_after: int,
        dedup_method: Optional[str] = None,
    ) -> 'DedupStatsFacet':
        duplicates_removed = rows_before - rows_after
        dedup_rate = duplicates_removed / rows_before if rows_before > 0 else 0.0
        return cls(
            rows_before=rows_before,
            rows_after=rows_after,
            duplicates_removed=duplicates_removed,
            dedup_rate=round(dedup_rate, 6),
            dedup_method=dedup_method,
        )


class DeduplicatorLineageExtractor(BaseLineageExtractor):
    """Extractor for Deduplicator operators.

    Enhancements over default:
    - On complete: adds DedupStatsFacet with dedup rate and method
    """

    @staticmethod
    def _get_dedup_method(op: Any) -> Optional[str]:
        """Infer dedup method from operator class name."""
        name = type(op).__name__.lower()
        if 'minhash' in name:
            return 'minhash_lsh'
        elif 'simhash' in name:
            return 'simhash'
        elif 'exact' in name or 'document' in name:
            return 'exact_hash'
        elif 'video' in name:
            return 'video_dedup'
        elif 'image' in name:
            return 'image_dedup'
        elif 'ray' in name:
            return 'ray_distributed'
        return 'unknown'

    def extract_on_complete(
        self,
        op: Any,
        columns_before: Optional[Set[str]] = None,
        columns_after: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
        row_count_after: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> OperatorLineage:
        """Extract lineage at dedup completion.

        Adds dedup-specific stats like dedup rate and method.
        """
        lineage = super().extract_on_complete(
            op, columns_before, columns_after,
            row_count_before, row_count_after, duration,
        )

        # Add dedup stats facet
        if row_count_before is not None and row_count_after is not None:
            method = self._get_dedup_method(op)
            lineage.run_facets['dedupStats'] = DedupStatsFacet.from_row_counts(
                rows_before=row_count_before,
                rows_after=row_count_after,
                dedup_method=method,
            )

        return lineage
