from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetRef:
    namespace: str
    name: str
    uri: str
    role: str  # input | output
    platform: str = "filesystem"
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecipeInfo:
    project_name: str
    executor_type: str
    config_path: Optional[str]
    process: List[Dict[str, Any]]
    process_hash: str
    num_operators: int


@dataclass
class PipelineRunContext:
    event_type: str  # START | COMPLETE | FAIL
    event_time: str
    run_id: str
    job_namespace: str
    job_name: str
    producer: str
    job_id: Optional[str]
    status: Optional[str]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    inputs: List[DatasetRef] = field(default_factory=list)
    outputs: List[DatasetRef] = field(default_factory=list)
    recipe: Optional[RecipeInfo] = None
    extra_run: Dict[str, Any] = field(default_factory=dict)
    extra_job: Dict[str, Any] = field(default_factory=dict)
