import os
import tempfile
import unittest

from data_juicer.core.data import NestedDataset
from data_juicer.core.metadata.resolver import MetadataResolver
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestMetadataResolver(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="metadata_resolver_")
        self.local_input = os.path.join(self.tmp_dir, "input.jsonl")
        self.local_output = os.path.join(self.tmp_dir, "output.jsonl")
        open(self.local_input, "w").close()
        self.cfg = {
            "dataset_path": f"0.5 s3://bucket/data/input.jsonl {self.local_input}",
            "export_path": self.local_output,
            "export_type": "jsonl",
            "metadata": {
                "enabled": True,
                "capture": {"schema": True, "rows": True},
            },
        }
        self.resolver = MetadataResolver(self.cfg)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.tmp_dir):
            import shutil

            shutil.rmtree(self.tmp_dir)

    def test_resolve_input_refs_from_weighted_dataset_path(self):
        refs = self.resolver.resolve_input_refs()
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0].source_type, "s3")
        self.assertEqual(refs[1].source_type, "local")

    def test_build_pipeline_snapshots_include_schema_and_rows(self):
        dataset = NestedDataset.from_list([{"text": "hello", "score": 1}, {"text": "world", "score": 2}])
        input_snapshot = self.resolver.build_pipeline_input_snapshot(dataset)
        output_snapshot = self.resolver.build_pipeline_output_snapshot(dataset)

        self.assertIsNotNone(input_snapshot)
        self.assertEqual(input_snapshot.rows, 2)
        self.assertEqual([field.name for field in input_snapshot.schema], ["text", "score"])
        self.assertEqual(output_snapshot.refs[0].uri, self.local_output)

    def test_resolve_iceberg_output_ref(self):
        iceberg_cfg = {
            "export_type": "iceberg",
            "export_extra_args": {
                "table_identifier": "db.processed_table",
                "catalog_kwargs": {"name": "prod"},
            },
            "metadata": {"enabled": True, "capture": {"schema": True, "rows": False}},
        }
        resolver = MetadataResolver(iceberg_cfg)
        refs = resolver.resolve_output_refs()

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].source_type, "iceberg")
        self.assertEqual(refs[0].namespace, "iceberg://prod")
        self.assertEqual(refs[0].name, "db.processed_table")


if __name__ == "__main__":
    unittest.main()
