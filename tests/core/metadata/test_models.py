import unittest

from data_juicer.core.metadata.models import Ctx, DatasetRef, DatasetSnapshot, OpCtx, OpKey, SchemaField
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestMetadataModels(DataJuicerTestCaseBase):
    def test_op_key_is_hashable(self):
        key = OpKey(partition_id=1, op_index=2)
        mapping = {key: "value"}
        self.assertEqual(mapping[OpKey(partition_id=1, op_index=2)], "value")

    def test_dataset_snapshot_defaults(self):
        snapshot = DatasetSnapshot()
        self.assertEqual(snapshot.refs, [])
        self.assertIsNone(snapshot.rows)
        self.assertEqual(snapshot.schema, [])
        self.assertIsNone(snapshot.storage_kind)

    def test_ctx_and_op_ctx_hold_runtime_state(self):
        ref = DatasetRef(namespace="file", name="/tmp/input.jsonl", role="input", source_type="local")
        schema = [SchemaField(name="text", type="str")]
        snapshot = DatasetSnapshot(refs=[ref], rows=3, schema=schema, storage_kind="hf")
        op_ctx = OpCtx(op_id="0:000:test_op", op_name="test_op", op_type="mapper", op_index=0)
        ctx = Ctx(
            run_id="run-1",
            job_id="job-1",
            executor_type="default",
            job_name="demo",
            namespace="data_juicer.default",
            project_name="demo",
            started_at="2026-01-01T00:00:00+00:00",
            input_snapshot=snapshot,
        )
        ctx.op_ctxs[OpKey(partition_id=0, op_index=0)] = op_ctx
        ctx.latest_snapshot_by_partition[0] = snapshot

        self.assertEqual(ctx.input_snapshot, snapshot)
        self.assertEqual(ctx.op_ctxs[OpKey(partition_id=0, op_index=0)].op_name, "test_op")
        self.assertEqual(ctx.latest_snapshot_by_partition[0].rows, 3)


if __name__ == "__main__":
    unittest.main()
