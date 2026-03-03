"""
Lineage Extractor Registry - Maps operator types to their extractors.

Inspired by Airflow's ExtractorManager. Supports automatic lookup
by operator class hierarchy with fallback to DefaultExtractor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from loguru import logger

from data_juicer.core.lineage.extractors.base import BaseLineageExtractor


class LineageExtractorRegistry:
    """Registry mapping operator classes to their lineage extractors.

    Supports:
    - Exact class matching
    - Inheritance-based fallback (checks parent classes)
    - Default extractor as final fallback
    """

    def __init__(self):
        self._registry: Dict[Type, Type[BaseLineageExtractor]] = {}
        self._instances: Dict[Type, BaseLineageExtractor] = {}
        self._default_extractor: Optional[BaseLineageExtractor] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-register built-in extractors on first use."""
        if self._initialized:
            return
        self._initialized = True

        try:
            from data_juicer.ops.base_op import (
                Deduplicator,
                Filter,
                Mapper,
            )

            from data_juicer.core.lineage.extractors.default_extractor import DefaultLineageExtractor
            from data_juicer.core.lineage.extractors.deduplicator_extractor import DeduplicatorLineageExtractor
            from data_juicer.core.lineage.extractors.filter_extractor import FilterLineageExtractor
            from data_juicer.core.lineage.extractors.mapper_extractor import MapperLineageExtractor

            self.register(Mapper, MapperLineageExtractor)
            self.register(Filter, FilterLineageExtractor)
            self.register(Deduplicator, DeduplicatorLineageExtractor)

            self._default_extractor = DefaultLineageExtractor()

        except ImportError as e:
            logger.debug(f"Could not initialize built-in lineage extractors: {e}")
            # Provide a minimal default
            from data_juicer.core.lineage.extractors.default_extractor import DefaultLineageExtractor
            self._default_extractor = DefaultLineageExtractor()

    def register(self, op_class: Type, extractor_class: Type[BaseLineageExtractor]) -> None:
        """Register an extractor for an operator class.

        Args:
            op_class: The operator class (e.g., Mapper, Filter).
            extractor_class: The extractor class to use for this operator type.
        """
        self._registry[op_class] = extractor_class
        logger.debug(f"Registered lineage extractor: {op_class.__name__} -> {extractor_class.__name__}")

    def get_extractor(self, op: Any) -> BaseLineageExtractor:
        """Get the appropriate extractor for an operator instance.

        Lookup order:
        1. Exact class match
        2. Parent class match (MRO)
        3. Default extractor

        Args:
            op: An operator instance.

        Returns:
            A BaseLineageExtractor instance.
        """
        self._ensure_initialized()

        op_type = type(op)

        # Check instance cache first
        if op_type in self._instances:
            return self._instances[op_type]

        # Check exact match
        if op_type in self._registry:
            extractor = self._registry[op_type]()
            self._instances[op_type] = extractor
            return extractor

        # Check parent classes via MRO
        for parent in op_type.__mro__[1:]:
            if parent in self._registry:
                extractor = self._registry[parent]()
                # Cache for this specific type too
                self._instances[op_type] = extractor
                return extractor

        # Fallback to default
        return self._default_extractor


# Global singleton registry
_global_registry: Optional[LineageExtractorRegistry] = None


def get_extractor_registry() -> LineageExtractorRegistry:
    """Get the global extractor registry singleton."""
    global _global_registry
    if _global_registry is None:
        _global_registry = LineageExtractorRegistry()
    return _global_registry
