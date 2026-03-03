"""
Default Lineage Extractor - Base implementation for all operator types.

Uses the base class default logic: captures schema, data quality,
column lineage, performance, and op config facets.
"""

from __future__ import annotations

from data_juicer.core.lineage.extractors.base import BaseLineageExtractor


class DefaultLineageExtractor(BaseLineageExtractor):
    """Default extractor that works with any operator type.

    Inherits all default behavior from BaseLineageExtractor:
    - extract_on_start: SchemaFacet + DataQualityFacet + OpConfigFacet
    - extract_on_complete: SchemaFacet + DataQualityFacet + ColumnLineageFacet + PerformanceFacet
    - extract_on_fail: ErrorFacet + OpConfigFacet
    """
    pass
