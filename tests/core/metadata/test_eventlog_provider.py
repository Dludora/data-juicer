import unittest

from data_juicer.core.metadata.models import Ctx, DatasetSnapshot, OpCtx
from data_juicer.core.metadata.provider import EventLogProvider
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class EventLogExecutorStub:
    def __init__(self):
        self.event_logger = object()
        self.calls = []

    def log_job_start(self, config, total_partitions):
        self.calls.append(("job_start", config, total_partitions))

    def log_job_complete(self, duration, output_path=None):
        self.calls.append(("job_complete", duration, output_path))

    def log_job_failed(self, error_message, duration):
        self.calls.append(("job_failed", error_message, duration))

    def log_op_start(self, partition_id, operation_name, operation_idx, op_args, **kwargs):
        self.calls.append(("op_start", partition_id, operation_name, operation_idx, op_args, kwargs))

    def log_op_complete(
        self,
        partition_id,
        operation_name,
        operation_idx,
        duration,
        checkpoint_path,
        input_rows,
        output_rows,
        **kwargs,
    ):
        self.calls.append(
            (
                "op_complete",
                partition_id,
                operation_name,
                operation_idx,
                duration,
                checkpoint_path,
                input_rows,
                output_rows,
                kwargs,
            )
        )

    def log_op_failed(self, partition_id, operation_name, operation_idx, error_message, retry_count, **kwargs):
        self.calls.append(("op_failed", partition_id, operation_name, operation_idx, error_message, retry_count, kwargs))


class TestEventLogProvider(DataJuicerTestCaseBase):
    def test_provider_adapts_ctx_into_existing_event_logging_api(self):
        executor = EventLogExecutorStub()
        cfg = {"export_path": "/tmp/output.jsonl"}
        provider = EventLogProvider(executor, cfg, "default")

        ctx = Ctx(
            run_id="run-1",
            job_id="job-1",
            executor_type="default",
            job_name="demo",
            namespace="data_juicer.default",
            project_name="demo",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:02+00:00",
            extra={"job_config": {"executor_type": "default"}, "num_operators": 3},
        )
        op_ctx = OpCtx(
            op_id="0:000:clean_text",
            op_name="clean_text",
            op_type="mapper",
            op_index=0,
            partition_id=2,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:01+00:00",
            metrics={"duration_seconds": 1.0, "input_rows": 10, "output_rows": 8, "retry_count": 1},
            input_snapshot=DatasetSnapshot(storage_kind="hf"),
            extra={"op_args": {"text_key": "text"}, "dag": {"node_id": "node-2-0"}},
        )

        provider.on_pipeline_started(ctx)
        provider.on_operator_started(ctx, op_ctx)
        provider.on_operator_completed(ctx, op_ctx)
        provider.on_operator_failed(ctx, op_ctx, RuntimeError("boom"))
        provider.on_pipeline_completed(ctx)

        self.assertEqual(executor.calls[0], ("job_start", {"executor_type": "default"}, 3))
        self.assertEqual(executor.calls[1][0], "op_start")
        self.assertEqual(executor.calls[1][1:4], (2, "clean_text", 0))
        self.assertEqual(executor.calls[2][0], "op_complete")
        self.assertEqual(executor.calls[2][5:8], (None, 10, 8))
        self.assertEqual(executor.calls[3][0], "op_failed")
        self.assertEqual(executor.calls[3][4], "boom")
        self.assertEqual(executor.calls[4][0], "job_complete")


if __name__ == "__main__":
    unittest.main()
