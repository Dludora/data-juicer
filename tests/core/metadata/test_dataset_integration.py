import os
import tempfile
import unittest

from data_juicer.core.data import NestedDataset
from data_juicer.core.metadata.manager import MetadataManager
from data_juicer.core.metadata.provider import MetadataProvider
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RecordingProvider(MetadataProvider):
    name = "recording"

    def __init__(self):
        self.events = []

    def on_operator_started(self, ctx, op_ctx):
        self.events.append(("started", op_ctx.op_name, op_ctx.input_snapshot.storage_kind))

    def on_operator_completed(self, ctx, op_ctx):
        self.events.append(("completed", op_ctx.op_name, op_ctx.output_snapshot.storage_kind))


class ExecutorStub:
    def __init__(self, work_dir):
        self.work_dir = work_dir

    def get_dag_metadata(self):
        return {"enabled": False, "node_count": 0, "edge_count": 0, "parallel_groups_count": 0}

    def get_dag_node_id(self, op_name, op_idx, partition_id=0):
        return f"node-{partition_id}-{op_idx}"


class UpperCaseOp:
    def __init__(self):
        self._name = "upper_case"
        self._op_cfg = {"text_key": "text"}

    def use_cuda(self):
        return False

    def run(self, dataset, exporter=None, tracer=None):
        return dataset.map(lambda row: {"text": row["text"].upper()})


class TestMetadataDatasetIntegration(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="metadata_dataset_")

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil

            shutil.rmtree(self.tmp_dir)

    def test_nested_dataset_process_updates_metadata_manager(self):
        dataset = NestedDataset.from_list([{"text": "hello"}, {"text": "world"}])
        provider = RecordingProvider()
        manager = MetadataManager(
            ExecutorStub(self.tmp_dir),
            {
                "project_name": "demo",
                "dataset_path": os.path.join(self.tmp_dir, "input.jsonl"),
                "metadata": {"enabled": True, "capture": {"schema": True, "rows": False}},
            },
            "default",
            providers=[provider],
        )
        manager.start_pipeline(
            input_dataset_obj=dataset,
            operators=[UpperCaseOp()],
        )

        result = dataset.process([UpperCaseOp()], open_monitor=False, metadata_manager=manager)

        self.assertEqual(result["text"], ["HELLO", "WORLD"])
        key = next(iter(manager.ctx.op_ctxs.keys()))
        self.assertEqual(manager.ctx.op_ctxs[key].status, "completed")
        self.assertEqual(manager.ctx.latest_snapshot_by_partition[0].storage_kind, "inmemory")
        self.assertEqual(
            provider.events,
            [("started", "upper_case", "hf"), ("completed", "upper_case", "inmemory")],
        )


if __name__ == "__main__":
    unittest.main()
