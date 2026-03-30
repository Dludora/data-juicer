import time
import uuid
from typing import Any, Dict, Optional

from loguru import logger

from data_juicer.core.lineage.service import OpenLineageService


class LineageLoggingMixin:
    """Executor mixin that emits pipeline-level OpenLineage events."""

    def __init__(self, cfg: Any, executor_type: str):
        self._lineage_enabled = False
        self._lineage_service: Optional[OpenLineageService] = None
        self._lineage_pipeline_start_time: Optional[float] = None
        self._lineage_run_id = self._build_run_id(cfg, executor_type)

        try:
            self._lineage_service = OpenLineageService(cfg, executor_type)
            self._lineage_enabled = bool(self._lineage_service.transport.enabled)
        except Exception as e:
            logger.warning(f"Failed to initialize lineage service, lineage disabled: {e}")
            self._lineage_enabled = False

    def _build_run_id(self, cfg: Any, executor_type: str) -> str:
        project_name = getattr(cfg, "project_name", "data_juicer")
        job_id = getattr(cfg, "job_id", None)

        if job_id:
            seed = f"data-juicer:{project_name}:{executor_type}:{job_id}"
            return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

        return str(uuid.uuid4())

    def emit_pipeline_start(self, extra_run: Optional[Dict[str, Any]] = None) -> None:
        if not self._lineage_enabled or not self._lineage_service:
            return
        self._lineage_pipeline_start_time = time.time()
        self._lineage_service.emit_pipeline_start(run_id=self._lineage_run_id, extra_run=extra_run)

    def emit_pipeline_complete(
        self,
        duration_seconds: Optional[float] = None,
        output_path: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._lineage_enabled or not self._lineage_service:
            return

        duration = duration_seconds
        if duration is None and self._lineage_pipeline_start_time is not None:
            duration = time.time() - self._lineage_pipeline_start_time

        self._lineage_service.emit_pipeline_complete(
            run_id=self._lineage_run_id,
            duration_seconds=duration,
            output_path=output_path,
            extra_run=extra_run,
        )

    def emit_pipeline_fail(
        self,
        error: Exception,
        duration_seconds: Optional[float] = None,
        output_path: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._lineage_enabled or not self._lineage_service:
            return

        duration = duration_seconds
        if duration is None and self._lineage_pipeline_start_time is not None:
            duration = time.time() - self._lineage_pipeline_start_time

        self._lineage_service.emit_pipeline_fail(
            run_id=self._lineage_run_id,
            error_message=str(error),
            duration_seconds=duration,
            output_path=output_path,
            extra_run=extra_run,
        )
