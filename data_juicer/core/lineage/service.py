from typing import Any, Dict, Optional

from loguru import logger

from data_juicer.core.lineage.builder import build_pipeline_context
from data_juicer.core.lineage.transport import OpenLineageTransport


class OpenLineageService:
    """Facade that emits pipeline-level OpenLineage events."""

    def __init__(self, cfg: Any, executor_type: str):
        self.cfg = cfg
        self.executor_type = executor_type
        self.transport = OpenLineageTransport(cfg)

    def _build_event(
        self,
        event_type: str,
        run_id: str,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
        extra_job: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        if not self.transport.enabled:
            return None

        try:
            from openlineage.client.event_v2 import (
                InputDataset,
                Job,
                OutputDataset,
                Run,
                RunEvent,
                RunState,
            )

            from data_juicer.core.lineage.facets import (
                DataJuicerJobFacet,
                DataJuicerRunFacet,
            )
        except Exception as e:
            logger.warning(f"OpenLineage SDK not available when building event: {e}")
            return None

        context = build_pipeline_context(
            cfg=self.cfg,
            executor_type=self.executor_type,
            event_type=event_type,
            run_id=run_id,
            status=status,
            duration_seconds=duration_seconds,
            error_message=error_message,
            output_path=output_path,
            extra_run=extra_run,
            extra_job=extra_job,
        )

        run_state_map = {
            "START": RunState.START,
            "COMPLETE": RunState.COMPLETE,
            "FAIL": RunState.FAIL,
        }

        run_facet = DataJuicerRunFacet(
            jobId=context.job_id,
            executorType=self.executor_type,
            status=context.status,
            durationSeconds=context.duration_seconds,
            errorMessage=context.error_message,
            workDir=getattr(self.cfg, "work_dir", None),
            custom=context.extra_run,
        )

        recipe = context.recipe
        job_facet = DataJuicerJobFacet(
            projectName=recipe.project_name if recipe else None,
            executorType=recipe.executor_type if recipe else self.executor_type,
            configPath=recipe.config_path if recipe else None,
            numOperators=recipe.num_operators if recipe else None,
            processHash=recipe.process_hash if recipe else None,
            process=recipe.process if recipe else None,
            custom=context.extra_job,
        )

        inputs = [InputDataset(namespace=d.namespace, name=d.name) for d in context.inputs]
        outputs = [OutputDataset(namespace=d.namespace, name=d.name) for d in context.outputs]

        return RunEvent(
            eventTime=context.event_time,
            eventType=run_state_map[event_type],
            producer=context.producer,
            run=Run(runId=context.run_id, facets={"datajuicer": run_facet}),
            job=Job(namespace=context.job_namespace, name=context.job_name, facets={"datajuicer": job_facet}),
            inputs=inputs,
            outputs=outputs,
        )

    def emit_pipeline_start(self, run_id: str, extra_run: Optional[Dict[str, Any]] = None) -> None:
        self._emit(
            event_type="START",
            run_id=run_id,
            status="started",
            extra_run=extra_run,
        )

    def emit_pipeline_complete(
        self,
        run_id: str,
        duration_seconds: Optional[float] = None,
        output_path: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            event_type="COMPLETE",
            run_id=run_id,
            status="completed",
            duration_seconds=duration_seconds,
            output_path=output_path,
            extra_run=extra_run,
        )

    def emit_pipeline_fail(
        self,
        run_id: str,
        error_message: str,
        duration_seconds: Optional[float] = None,
        output_path: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._emit(
            event_type="FAIL",
            run_id=run_id,
            status="failed",
            duration_seconds=duration_seconds,
            error_message=error_message,
            output_path=output_path,
            extra_run=extra_run,
        )

    def _emit(
        self,
        event_type: str,
        run_id: str,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        error_message: Optional[str] = None,
        output_path: Optional[str] = None,
        extra_run: Optional[Dict[str, Any]] = None,
        extra_job: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.transport.enabled:
            return

        event = self._build_event(
            event_type=event_type,
            run_id=run_id,
            status=status,
            duration_seconds=duration_seconds,
            error_message=error_message,
            output_path=output_path,
            extra_run=extra_run,
            extra_job=extra_job,
        )
        if event is None:
            return

        self.transport.emit(event)
