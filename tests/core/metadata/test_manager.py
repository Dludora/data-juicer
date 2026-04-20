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

    def on_pipeline_started(self, ctx):
        self.events.append(("pipeline_started", ctx.run_id))

    def on_pipeline_completed(self, ctx):
        self.events.append(("pipeline_completed", ctx.status))

    def on_operator_started(self, ctx, op_ctx):
        self.events.append(("operator_started", op_ctx.op_name, op_ctx.partition_id))

    def on_operator_completed(self, ctx, op_ctx):
        self.events.append(("operator_completed", op_ctx.op_name, op_ctx.status))


class FakeExecutor:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.started_nodes = []
        self.completed_nodes = []
        self.failed_nodes = []

    def get_dag_metadata(self):
        return {
            "enabled": True,
            "node_count": 2,
            "edge_count": 1,
            "parallel_groups_count": 0,
            "execution_plan_path": os.path.join(self.work_dir, "dag.json"),
        }

    def get_dag_node_id(self, op_name, op_idx, partition_id=0):
        return f"node-{partition_id}-{op_idx}-{op_name}"

    def _mark_dag_node_started(self, node_id):
        self.started_nodes.append(node_id)

    def _mark_dag_node_completed(self, node_id, duration):
        self.completed_nodes.append((node_id, duration))

    def _mark_dag_node_failed(self, node_id, error_message, duration):
        self.failed_nodes.append((node_id, error_message, duration))


class FakeOp:
    def __init__(self, name="fake_mapper"):
        self._name = name
        self._op_cfg = {"keep": True}


class TestMetadataManager(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="metadata_manager_")
        self.cfg = {
            "project_name": "demo_project",
            "job_id": "job-42",
            "dataset_path": os.path.join(self.tmp_dir, "input.jsonl"),
            "export_path": os.path.join(self.tmp_dir, "output.jsonl"),
            "metadata": {
                "enabled": True,
                "namespace": "demo.namespace",
                "capture": {"schema": True, "rows": False},
            },
        }
        self.executor = FakeExecutor(self.tmp_dir)
        self.provider = RecordingProvider()
        self.manager = MetadataManager(
            self.executor,
            self.cfg,
            "default",
            providers=[self.provider],
        )

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil

            shutil.rmtree(self.tmp_dir)

    def test_pipeline_and_operator_lifecycle_updates_ctx(self):
        dataset = NestedDataset.from_list([{"text": "hello"}])
        op = FakeOp()

        ctx = self.manager.start_pipeline(
            input_dataset_obj=dataset,
            operators=[op],
        )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.namespace, "demo.namespace")
        self.assertEqual(ctx.extra["num_operators"], 1)
        self.assertEqual(ctx.latest_snapshot_by_partition[0].storage_kind, "hf")
        self.assertEqual(ctx.extra["recipe"]["project_name"], "demo_project")
        self.assertIn("recipe_hash", ctx.extra)

        op_ctx = self.manager.start_operator(op=op, op_index=0, input_dataset_obj=dataset, partition_id=0)
        self.assertEqual(op_ctx.extra["dag"]["node_id"], "node-0-0-fake_mapper")
        self.assertEqual(self.executor.started_nodes, ["node-0-0-fake_mapper"])
        self.assertTrue(op_ctx.extra["source"]["file_path"].endswith("test_manager.py"))
        self.assertIsNotNone(op_ctx.extra["source"]["git_commit"])
        self.assertIsNotNone(op_ctx.extra["source"]["repo_owner"])
        self.assertIn("git_branch", op_ctx.extra["source"])
        self.assertIn("git_author_name", op_ctx.extra["source"])
        self.assertIn("git_committer_email", op_ctx.extra["source"])

        self.manager.complete_operator(
            op_index=0,
            output_dataset_obj=dataset,
            partition_id=0,
            metrics={"duration_seconds": 1.25},
        )
        self.assertEqual(self.executor.completed_nodes, [("node-0-0-fake_mapper", 1.25)])
        self.assertEqual(ctx.op_ctxs[op_ctx_key(0, 0)].status, "completed")
        self.assertEqual(ctx.latest_snapshot_by_partition[0].refs[0].source_type, "inmemory")

        self.manager.complete_pipeline(output_dataset_obj=dataset)
        self.assertEqual(ctx.status, "completed")
        self.assertEqual(
            self.provider.events,
            [
                ("pipeline_started", ctx.run_id),
                ("operator_started", "fake_mapper", 0),
                ("operator_completed", "fake_mapper", "completed"),
                ("pipeline_completed", "completed"),
            ],
        )


def op_ctx_key(partition_id, op_index):
    from data_juicer.core.metadata.models import OpKey

    return OpKey(partition_id=partition_id, op_index=op_index)


if __name__ == "__main__":
    unittest.main()
