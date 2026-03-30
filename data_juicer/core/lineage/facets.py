from typing import Any, Dict, Optional

import attrs
from openlineage.client.facet_v2 import JobFacet, RunFacet


@attrs.define
class DataJuicerRunFacet(RunFacet):
    """Run-level facet for data-juicer pipeline execution metadata."""

    jobId: Optional[str] = None
    executorType: Optional[str] = None
    status: Optional[str] = None
    durationSeconds: Optional[float] = None
    errorMessage: Optional[str] = None
    workDir: Optional[str] = None
    custom: Dict[str, Any] = attrs.field(factory=dict)

    @staticmethod
    def _get_schema() -> str:
        return (
            "https://raw.githubusercontent.com/modelscope/data-juicer/main/"
            "data_juicer/core/lineage/schemas/DataJuicerRunFacet.json"
        )


@attrs.define
class DataJuicerJobFacet(JobFacet):
    """Job-level facet for data-juicer pipeline definition metadata."""

    projectName: Optional[str] = None
    executorType: Optional[str] = None
    configPath: Optional[str] = None
    numOperators: Optional[int] = None
    processHash: Optional[str] = None
    process: Any = None
    custom: Dict[str, Any] = attrs.field(factory=dict)

    @staticmethod
    def _get_schema() -> str:
        return (
            "https://raw.githubusercontent.com/modelscope/data-juicer/main/"
            "data_juicer/core/lineage/schemas/DataJuicerJobFacet.json"
        )
