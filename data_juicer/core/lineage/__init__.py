"""
Data-Juicer Lineage Module

OpenLineage-inspired multi-granularity data lineage tracking for Data-Juicer.

Supports 4 granularity levels:
- Level 0: Pipeline Lineage (pipeline-level input/output tracking)
- Level 1: Operator Lineage (per-op metrics and schema)
- Level 2: Column Lineage (column-level read/write/delete tracking)
- Level 3: Sample Lineage (sample-level changes, via existing Tracer)

Currently implemented: Level 0, 1, 2.
"""

from data_juicer.core.lineage.config import LineageConfig
from data_juicer.core.lineage.events import (
    DatasetDescriptor,
    JobDescriptor,
    LineageEvent,
    LineageEventType,
    OperatorDescriptor,
)
from data_juicer.core.lineage.facets import (
    BaseFacet,
    ColumnLineageFacet,
    DataQualityFacet,
    ErrorFacet,
    OpConfigFacet,
    PerformanceFacet,
    SchemaFacet,
)
from data_juicer.core.lineage.adapter import LineageAdapter
from data_juicer.core.lineage.mixin import LineageLoggingMixin

__all__ = [
    "LineageConfig",
    "LineageEvent",
    "LineageEventType",
    "DatasetDescriptor",
    "OperatorDescriptor",
    "JobDescriptor",
    "BaseFacet",
    "SchemaFacet",
    "ColumnLineageFacet",
    "DataQualityFacet",
    "PerformanceFacet",
    "ErrorFacet",
    "OpConfigFacet",
    "LineageAdapter",
    "LineageLoggingMixin",
]
