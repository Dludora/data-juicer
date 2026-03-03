"""
Lineage Events - Core event data structures for data lineage tracking.

Inspired by OpenLineage RunEvent model. Each LineageEvent captures a
lifecycle moment (START/COMPLETE/FAIL) at either pipeline or operator level,
with attached input/output dataset descriptors and extensible facets.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from data_juicer.core.lineage.facets import BaseFacet


class LineageEventType(Enum):
    """Lifecycle event types, following OpenLineage convention."""

    START = "START"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


@dataclass
class JobDescriptor:
    """Describes a Data-Juicer pipeline (analogous to OpenLineage Job)."""

    name: str
    """Pipeline name, typically derived from config file name."""

    namespace: str = "data-juicer"
    """Namespace for the job. Defaults to 'data-juicer'."""

    facets: Dict[str, Any] = field(default_factory=dict)
    """Extensible job-level facets."""

    def to_dict(self) -> Dict[str, Any]:
        facets_dict = {}
        for k, v in self.facets.items():
            if isinstance(v, BaseFacet):
                facets_dict[k] = v.to_dict()
            else:
                facets_dict[k] = v
        return {
            "name": self.name,
            "namespace": self.namespace,
            "facets": facets_dict,
        }


@dataclass
class DatasetDescriptor:
    """Describes an input or output dataset (analogous to OpenLineage Dataset)."""

    name: str
    """Dataset identifier (path, URI, or logical name)."""

    namespace: str = "data-juicer"
    """Dataset namespace."""

    facets: Dict[str, Any] = field(default_factory=dict)
    """Extensible dataset facets (schema, quality, storage, etc.)."""

    def to_dict(self) -> Dict[str, Any]:
        facets_dict = {}
        for k, v in self.facets.items():
            if isinstance(v, BaseFacet):
                facets_dict[k] = v.to_dict()
            else:
                facets_dict[k] = v
        return {
            "name": self.name,
            "namespace": self.namespace,
            "facets": facets_dict,
        }


@dataclass
class OperatorDescriptor:
    """Describes a Data-Juicer operator (analogous to an OpenLineage Task)."""

    name: str
    """Operator registry name (e.g., 'language_id_score_filter')."""

    type: str
    """Operator base type (e.g., 'Mapper', 'Filter')."""

    index: int
    """Position in the pipeline (0-based)."""

    facets: Dict[str, Any] = field(default_factory=dict)
    """Extensible operator facets (config, etc.)."""

    def to_dict(self) -> Dict[str, Any]:
        facets_dict = {}
        for k, v in self.facets.items():
            if isinstance(v, BaseFacet):
                facets_dict[k] = v.to_dict()
            else:
                facets_dict[k] = v
        return {
            "name": self.name,
            "type": self.type,
            "index": self.index,
            "facets": facets_dict,
        }


@dataclass
class LineageEvent:
    """A lineage event capturing a lifecycle moment.

    Follows the OpenLineage RunEvent model:
    - event_type: START / COMPLETE / FAIL
    - run_id: unique identifier for this pipeline execution
    - job: pipeline descriptor
    - operator: optional operator descriptor (for op-level events)
    - inputs / outputs: dataset descriptors
    - facets: run-level facets (performance, error, column lineage, etc.)
    """

    event_type: LineageEventType
    """Event lifecycle type."""

    event_time: float
    """Unix timestamp of the event."""

    run_id: str
    """Unique run identifier (UUID) for the pipeline execution."""

    job: JobDescriptor
    """Pipeline descriptor."""

    operator: Optional[OperatorDescriptor] = None
    """Operator descriptor (None for pipeline-level events)."""

    inputs: List[DatasetDescriptor] = field(default_factory=list)
    """Input dataset descriptors."""

    outputs: List[DatasetDescriptor] = field(default_factory=list)
    """Output dataset descriptors."""

    facets: Dict[str, Any] = field(default_factory=dict)
    """Run-level facets (performance, error, column lineage, etc.)."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        facets_dict = {}
        for k, v in self.facets.items():
            if isinstance(v, BaseFacet):
                facets_dict[k] = v.to_dict()
            else:
                facets_dict[k] = v

        result = {
            "eventType": self.event_type.value,
            "eventTime": self.event_time,
            "run": {"runId": self.run_id},
            "job": self.job.to_dict(),
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
            "facets": facets_dict,
        }

        if self.operator is not None:
            result["operator"] = self.operator.to_dict()

        return result

    @classmethod
    def create_pipeline_start(
        cls,
        run_id: str,
        job: JobDescriptor,
        inputs: Optional[List[DatasetDescriptor]] = None,
        facets: Optional[Dict[str, Any]] = None,
    ) -> "LineageEvent":
        """Factory: create a pipeline START event."""
        return cls(
            event_type=LineageEventType.START,
            event_time=time.time(),
            run_id=run_id,
            job=job,
            inputs=inputs or [],
            facets=facets or {},
        )

    @classmethod
    def create_pipeline_complete(
        cls,
        run_id: str,
        job: JobDescriptor,
        outputs: Optional[List[DatasetDescriptor]] = None,
        facets: Optional[Dict[str, Any]] = None,
    ) -> "LineageEvent":
        """Factory: create a pipeline COMPLETE event."""
        return cls(
            event_type=LineageEventType.COMPLETE,
            event_time=time.time(),
            run_id=run_id,
            job=job,
            outputs=outputs or [],
            facets=facets or {},
        )

    @classmethod
    def create_pipeline_fail(
        cls,
        run_id: str,
        job: JobDescriptor,
        facets: Optional[Dict[str, Any]] = None,
    ) -> "LineageEvent":
        """Factory: create a pipeline FAIL event."""
        return cls(
            event_type=LineageEventType.FAIL,
            event_time=time.time(),
            run_id=run_id,
            job=job,
            facets=facets or {},
        )

    @classmethod
    def create_op_start(
        cls,
        run_id: str,
        job: JobDescriptor,
        operator: OperatorDescriptor,
        inputs: Optional[List[DatasetDescriptor]] = None,
        facets: Optional[Dict[str, Any]] = None,
    ) -> "LineageEvent":
        """Factory: create an operator START event."""
        return cls(
            event_type=LineageEventType.START,
            event_time=time.time(),
            run_id=run_id,
            job=job,
            operator=operator,
            inputs=inputs or [],
            facets=facets or {},
        )

    @classmethod
    def create_op_complete(
        cls,
        run_id: str,
        job: JobDescriptor,
        operator: OperatorDescriptor,
        outputs: Optional[List[DatasetDescriptor]] = None,
        facets: Optional[Dict[str, Any]] = None,
    ) -> "LineageEvent":
        """Factory: create an operator COMPLETE event."""
        return cls(
            event_type=LineageEventType.COMPLETE,
            event_time=time.time(),
            run_id=run_id,
            job=job,
            operator=operator,
            outputs=outputs or [],
            facets=facets or {},
        )

    @classmethod
    def create_op_fail(
        cls,
        run_id: str,
        job: JobDescriptor,
        operator: OperatorDescriptor,
        facets: Optional[Dict[str, Any]] = None,
    ) -> "LineageEvent":
        """Factory: create an operator FAIL event."""
        return cls(
            event_type=LineageEventType.FAIL,
            event_time=time.time(),
            run_id=run_id,
            job=job,
            operator=operator,
            facets=facets or {},
        )
