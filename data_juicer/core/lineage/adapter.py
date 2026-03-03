"""
Lineage Adapter - Orchestrates extractors, assembles events, and emits via transport.

The LineageAdapter is the central coordinator:
1. Receives lifecycle hooks (pipeline/operator start/complete/fail)
2. Uses the ExtractorRegistry to get the right extractor for each operator
3. Assembles LineageEvent objects from extracted facets
4. Emits events through the configured Transport
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from loguru import logger

from data_juicer.core.lineage.config import LineageConfig
from data_juicer.core.lineage.events import (
    DatasetDescriptor,
    JobDescriptor,
    LineageEvent,
    LineageEventType,
    OperatorDescriptor,
)
from data_juicer.core.lineage.extractors.registry import get_extractor_registry
from data_juicer.core.lineage.facets import (
    DataQualityFacet,
    ErrorFacet,
    PerformanceFacet,
    SchemaFacet,
)
from data_juicer.core.lineage.transports.base import BaseTransport
from data_juicer.core.lineage.transports.factory import create_transport


class LineageAdapter:
    """Orchestrates lineage extraction and event emission.

    Usage::

        adapter = LineageAdapter(config)

        # Pipeline level
        adapter.on_pipeline_start(job_name, dataset_path)
        # ... for each operator ...
        adapter.on_op_start(op, op_index, columns_before, row_count_before)
        adapter.on_op_complete(op, op_index, columns_before, columns_after,
                               row_count_before, row_count_after, duration)
        # ... end of pipeline ...
        adapter.on_pipeline_complete(dataset_path)

        adapter.close()
    """

    def __init__(self, config: LineageConfig):
        self._config = config
        self._run_id = str(uuid4())
        self._job: Optional[JobDescriptor] = None
        self._transport: Optional[BaseTransport] = None
        self._pipeline_start_time: Optional[float] = None

        # Initialize transport
        if config.enabled:
            try:
                self._transport = create_transport(config.transport)
                logger.info(
                    f"Lineage tracking enabled. Transport: {config.transport.type}, "
                    f"Run ID: {self._run_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize lineage transport: {e}")
                self._transport = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._transport is not None

    # ------------------------------------------------------------------ #
    #  Pipeline-Level Events (Level 0)
    # ------------------------------------------------------------------ #

    def on_pipeline_start(
        self,
        job_name: str,
        input_dataset: Optional[str] = None,
        num_operators: Optional[int] = None,
    ) -> None:
        """Emit a pipeline START event."""
        if not self.enabled or not self._config.granularity.pipeline:
            return

        self._pipeline_start_time = time.time()
        self._job = JobDescriptor(
            name=job_name,
            namespace=self._config.namespace,
        )

        inputs = []
        facets: Dict[str, Any] = {}

        if input_dataset:
            inputs.append(DatasetDescriptor(
                name=input_dataset,
                namespace=self._config.namespace,
            ))

        if num_operators is not None:
            facets['pipelineInfo'] = {
                'numOperators': num_operators,
            }

        event = LineageEvent.create_pipeline_start(
            run_id=self._run_id,
            job=self._job,
            inputs=inputs,
            facets=facets,
        )
        self._emit(event)

    def on_pipeline_complete(
        self,
        output_dataset: Optional[str] = None,
        row_count: Optional[int] = None,
    ) -> None:
        """Emit a pipeline COMPLETE event."""
        if not self.enabled or not self._config.granularity.pipeline:
            return

        if self._job is None:
            return

        outputs = []
        facets: Dict[str, Any] = {}

        if output_dataset:
            ds_facets = {}
            if row_count is not None:
                ds_facets['dataQuality'] = DataQualityFacet(row_count=row_count)
            outputs.append(DatasetDescriptor(
                name=output_dataset,
                namespace=self._config.namespace,
                facets=ds_facets,
            ))

        if self._pipeline_start_time is not None:
            duration = time.time() - self._pipeline_start_time
            facets['performance'] = PerformanceFacet(duration_seconds=duration)

        event = LineageEvent.create_pipeline_complete(
            run_id=self._run_id,
            job=self._job,
            outputs=outputs,
            facets=facets,
        )
        self._emit(event)

    def on_pipeline_fail(self, error: Exception) -> None:
        """Emit a pipeline FAIL event."""
        if not self.enabled or not self._config.granularity.pipeline:
            return

        if self._job is None:
            return

        facets: Dict[str, Any] = {
            'error': ErrorFacet.from_exception(error),
        }

        if self._pipeline_start_time is not None:
            duration = time.time() - self._pipeline_start_time
            facets['performance'] = PerformanceFacet(duration_seconds=duration)

        event = LineageEvent.create_pipeline_fail(
            run_id=self._run_id,
            job=self._job,
            facets=facets,
        )
        self._emit(event)

    # ------------------------------------------------------------------ #
    #  Operator-Level Events (Level 1 + Level 2)
    # ------------------------------------------------------------------ #

    def on_op_start(
        self,
        op: Any,
        op_index: int,
        columns_before: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
    ) -> None:
        """Emit an operator START event."""
        if not self.enabled or not self._config.granularity.operator:
            return

        if self._job is None:
            return

        # Use extractor registry
        registry = get_extractor_registry()
        extractor = registry.get_extractor(op)

        cols = columns_before if self._config.granularity.column else None
        lineage = extractor.extract_on_start(op, cols, row_count_before)

        op_descriptor = OperatorDescriptor(
            name=getattr(op, '_name', type(op).__name__),
            type=type(op).__bases__[0].__name__ if type(op).__bases__ else 'OP',
            index=op_index,
            facets=lineage.op_facets,
        )

        inputs = []
        if lineage.input_facets:
            inputs.append(DatasetDescriptor(
                name='intermediate',
                namespace=self._config.namespace,
                facets=lineage.input_facets,
            ))

        event = LineageEvent.create_op_start(
            run_id=self._run_id,
            job=self._job,
            operator=op_descriptor,
            inputs=inputs,
            facets=lineage.run_facets,
        )
        self._emit(event)

    def on_op_complete(
        self,
        op: Any,
        op_index: int,
        columns_before: Optional[Set[str]] = None,
        columns_after: Optional[Set[str]] = None,
        row_count_before: Optional[int] = None,
        row_count_after: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Emit an operator COMPLETE event."""
        if not self.enabled or not self._config.granularity.operator:
            return

        if self._job is None:
            return

        registry = get_extractor_registry()
        extractor = registry.get_extractor(op)

        cols_before = columns_before if self._config.granularity.column else None
        cols_after = columns_after if self._config.granularity.column else None

        lineage = extractor.extract_on_complete(
            op, cols_before, cols_after,
            row_count_before, row_count_after, duration,
        )

        op_descriptor = OperatorDescriptor(
            name=getattr(op, '_name', type(op).__name__),
            type=type(op).__bases__[0].__name__ if type(op).__bases__ else 'OP',
            index=op_index,
            facets=lineage.op_facets,
        )

        outputs = []
        if lineage.output_facets:
            outputs.append(DatasetDescriptor(
                name='intermediate',
                namespace=self._config.namespace,
                facets=lineage.output_facets,
            ))

        event = LineageEvent.create_op_complete(
            run_id=self._run_id,
            job=self._job,
            operator=op_descriptor,
            outputs=outputs,
            facets=lineage.run_facets,
        )
        self._emit(event)

    def on_op_fail(
        self,
        op: Any,
        op_index: int,
        error: Exception,
    ) -> None:
        """Emit an operator FAIL event."""
        if not self.enabled or not self._config.granularity.operator:
            return

        if self._job is None:
            return

        registry = get_extractor_registry()
        extractor = registry.get_extractor(op)
        lineage = extractor.extract_on_fail(op, error)

        op_descriptor = OperatorDescriptor(
            name=getattr(op, '_name', type(op).__name__),
            type=type(op).__bases__[0].__name__ if type(op).__bases__ else 'OP',
            index=op_index,
            facets=lineage.op_facets,
        )

        event = LineageEvent.create_op_fail(
            run_id=self._run_id,
            job=self._job,
            operator=op_descriptor,
            facets=lineage.run_facets,
        )
        self._emit(event)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _emit(self, event: LineageEvent) -> None:
        """Emit an event via the configured transport."""
        if self._transport is None:
            return

        try:
            event_dict = event.to_dict()
            self._transport.emit(event_dict)
        except Exception as e:
            logger.warning(f"Failed to emit lineage event: {e}")

    def close(self) -> None:
        """Close the transport and release resources."""
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception as e:
                logger.warning(f"Failed to close lineage transport: {e}")
            finally:
                self._transport = None
