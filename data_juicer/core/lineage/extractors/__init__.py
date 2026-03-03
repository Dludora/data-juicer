"""
Lineage Extractors - Per-operator-type lineage metadata extraction.

Inspired by Airflow's ExtractorManager / BaseExtractor pattern.
"""

from data_juicer.core.lineage.extractors.base import BaseLineageExtractor, OperatorLineage
from data_juicer.core.lineage.extractors.registry import (
    LineageExtractorRegistry,
    get_extractor_registry,
)

__all__ = [
    "BaseLineageExtractor",
    "OperatorLineage",
    "LineageExtractorRegistry",
    "get_extractor_registry",
]
