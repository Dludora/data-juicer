import copy
import os
import os.path as osp
import shutil
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.mm_utils import load_images_byte


class TestRayExporter(DataJuicerTestCaseBase):

    def setUp(self):
        """Set up test data"""
        super().setUp()

        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = f'{cur_dir}/tmp/{self.__class__.__name__}/{self._testMethodName}'
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.data = [
            {'text': 'hello', Fields.stats: {'score': 1}, HashKeys.hash: 'a1'},
            {'text': 'world', Fields.stats: {'score': 2}, HashKeys.hash: 'b2'},
            {'text': 'test', Fields.stats: {'score': 3}, HashKeys.hash: 'c3'}
        ]
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up temporary outputs"""

        self.dataset = None
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        super().tearDown()

    def _pop_raw_data_keys(self, keys):
        res = copy.deepcopy(self.data)
        for d_i in res:
            for k in keys:
                d_i.pop(k, None)

        return res

    @TEST_TAG('ray')
    def test_json_not_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.json')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats, HashKeys.hash]))

    @TEST_TAG('ray')
    def test_jsonl_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.jsonl')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self.data)

    @TEST_TAG('ray')
    def test_parquet_keep_stats(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.parquet')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_parquet(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self._pop_raw_data_keys([HashKeys.hash]))

    @TEST_TAG('ray')
    def test_lance_keep_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.lance')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_lance(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats]))

    @TEST_TAG('ray')
    def test_webdataset_multi_images(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'images': [img1_path, img2_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path])},
            {
                'json': {
                    'text': 'world',
                    'images': [img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img2_path, img3_path])},
            {
                'json': {
                    'text': 'test',
                    'images': [img1_path, img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path, img3_path])}
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path)
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])

        for i in range(len(data)):
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(
                res_list[i]['jpgs'],
                [Image.open(io.BytesIO(v)) for v in data[i]['jpgs']]
            )

    @TEST_TAG('ray')
    def test_webdataset_multi_videos_frames_bytes(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'videos': ['video1.mp4', 'video2.mp4']
                    },
                'mp4s': [
                    load_images_byte([img1_path]),  # as video1 frames bytes
                    load_images_byte([img1_path, img2_path])   # as video2 frames path
                    ]
            },
            {
                'json': {
                    'text': 'world',
                    'videos': ['video1.mp4']
                    },
                'mp4s': [
                    load_images_byte([img2_path, img3_path])  # as video1 frames
                    ]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp4s']) > 1:
                tgt_mp4s = [[Image.open(io.BytesIO(f_i)) for f_i in v_i] for v_i in data[i]['mp4s']]
            else:
                tgt_mp4s = [Image.open(io.BytesIO(f_i)) for f_i in data[i]['mp4s'][0]]
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(res_list[i]['mp4s'], tgt_mp4s)

    @TEST_TAG('ray')
    def test_webdataset_multi_videos_frames_path(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img8.jpg')
        img2_path = osp.join(data_dir, 'img9.jpg')
        img3_path = osp.join(data_dir, 'img10.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'videos': ['video1.mp4', 'video2.mp4']
                    },
                'mp4s': [
                    [img1_path],  # as video1 frames path
                    [img1_path, img2_path]   # as video2 frames path
                    ]
            },
            {
                'json': {
                    'text': 'world',
                    'videos': ['video1.mp4']
                    },
                'mp4s': [
                    [img2_path, img3_path]  # as video1 frames path
                    ]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp4s']) > 1:
                tgt_mp4s = [[Image.open(f_i, formats=['jpeg']) for f_i in v_i] for v_i in data[i]['mp4s']]
            else:
                tgt_mp4s = [Image.open(f_i, formats=['jpeg']) for f_i in data[i]['mp4s'][0]]
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(res_list[i]['mp4s'], tgt_mp4s)

    @TEST_TAG('ray')
    def test_webdataset_multi_audios_path(self):
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.utils.mm_utils import load_audio

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        audio1_path = osp.join(data_dir, 'audio1.wav')
        audio2_path = osp.join(data_dir, 'audio2.wav')
        audio3_path = osp.join(data_dir, 'audio3.ogg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    },
                'mp3s': [audio1_path]
            },
            {
                'json': {
                    'text': 'world',
                    },
                'mp3s': [audio2_path, audio3_path]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))

        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp3s']) <= 1:
                mp3s_list = [res_list[i]['mp3s']]
            else:
                mp3s_list = res_list[i]['mp3s']

            tgt_mp3s = [load_audio(f_i) for f_i in data[i]['mp3s']]
            
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])

            for j in range(len(mp3s_list)):
                arr, sampling_rate = mp3s_list[j]
                tgt_arr, tgt_sampling_rate = tgt_mp3s[j]
                import numpy as np
                np.testing.assert_array_equal(arr, tgt_arr)
                self.assertEqual(sampling_rate, tgt_sampling_rate)

class TestRayExporterPaimon(DataJuicerTestCaseBase):
    def _build_fake_paimon_modules(self, catalog_factory_create, schema_cls=None):
        fake_pypaimon = types.ModuleType("pypaimon")
        fake_pypaimon.__path__ = []
        fake_pypaimon.Schema = schema_cls or MagicMock(name="Schema")

        fake_pypaimon_catalog = types.ModuleType("pypaimon.catalog")
        fake_pypaimon_catalog.__path__ = []
        fake_pypaimon_catalog_factory = types.ModuleType("pypaimon.catalog.catalog_factory")
        fake_pypaimon_catalog_factory.CatalogFactory = types.SimpleNamespace(create=catalog_factory_create)

        fake_pypaimon.catalog = fake_pypaimon_catalog
        fake_pypaimon_catalog.catalog_factory = fake_pypaimon_catalog_factory

        return {
            "pypaimon": fake_pypaimon,
            "pypaimon.catalog": fake_pypaimon_catalog,
            "pypaimon.catalog.catalog_factory": fake_pypaimon_catalog_factory,
        }

    def test_write_paimon_uses_write_ray_when_available(self):
        dataset = MagicMock(name="ray_dataset")

        table_write = MagicMock(name="table_write")
        write_builder = MagicMock(name="write_builder")
        write_builder.new_write.return_value = table_write

        table = MagicMock(name="table")
        table.new_batch_write_builder.return_value = write_builder

        catalog = MagicMock(name="catalog")
        catalog.get_table.return_value = table
        mock_create = MagicMock(return_value=catalog)

        fake_modules = self._build_fake_paimon_modules(mock_create)

        export_extra_args = {
            "catalog_options": {"metastore": "filesystem", "warehouse": "file:///tmp/warehouse"},
            "table_identifier": "db.sample_table",
            "concurrency": 4,
            "ray_remote_args": {"num_cpus": 1},
        }

        with patch.dict(sys.modules, fake_modules):
            RayExporter.write_paimon(dataset, "/tmp/fallback.jsonl", export_extra_args=export_extra_args)

        mock_create.assert_called_once_with(export_extra_args["catalog_options"])
        catalog.get_table.assert_called_once_with("db.sample_table")
        table_write.write_ray.assert_called_once_with(
            dataset,
            concurrency=4,
            ray_remote_args={"num_cpus": 1},
        )
        table_write.close.assert_called_once()
        write_builder.new_commit.assert_not_called()

    def test_write_paimon_falls_back_to_arrow_commit_when_write_ray_unavailable(self):
        import pandas as pd

        dataset = MagicMock(name="ray_dataset")
        dataset.limit.return_value.to_pandas.return_value = pd.DataFrame({"dt": ["2024-01-01"], "value": ["x"]})
        dataset.to_arrow.return_value = "arrow_table"

        table_write = types.SimpleNamespace(
            write_arrow=MagicMock(),
            prepare_commit=MagicMock(return_value=["commit_msg"]),
            close=MagicMock(),
        )
        table_commit = types.SimpleNamespace(commit=MagicMock(), close=MagicMock())
        overwrite_builder = MagicMock(name="overwrite_builder")
        overwrite_builder.new_write.return_value = table_write
        overwrite_builder.new_commit.return_value = table_commit

        write_builder = MagicMock(name="write_builder")
        write_builder.overwrite.return_value = overwrite_builder

        table = MagicMock(name="table")
        table.new_batch_write_builder.return_value = write_builder

        catalog = MagicMock(name="catalog")
        catalog.get_table.side_effect = [ValueError("missing table"), table]
        mock_create = MagicMock(return_value=catalog)

        schema_cls = MagicMock(name="Schema")
        schema_cls.from_pyarrow_schema = MagicMock(return_value="paimon_schema")

        fake_modules = self._build_fake_paimon_modules(mock_create, schema_cls=schema_cls)

        export_extra_args = {
            "catalog_options": {"metastore": "filesystem", "warehouse": "file:///tmp/warehouse"},
            "table_identifier": "db.sample_table",
            "overwrite": True,
            "schema_kwargs": {"partition_keys": ["dt"]},
        }

        with patch.dict(sys.modules, fake_modules):
            RayExporter.write_paimon(dataset, "/tmp/fallback.parquet", export_extra_args=export_extra_args)

        mock_create.assert_called_once_with(export_extra_args["catalog_options"])
        schema_cls.from_pyarrow_schema.assert_called_once()
        schema_call_kwargs = schema_cls.from_pyarrow_schema.call_args.kwargs
        self.assertEqual(schema_call_kwargs["partition_keys"], ["dt"])
        self.assertIn("pa_schema", schema_call_kwargs)
        catalog.create_table.assert_called_once_with(
            "db.sample_table",
            schema="paimon_schema",
            ignore_if_exists=True,
        )
        write_builder.overwrite.assert_called_once_with()
        table_write.write_arrow.assert_called_once_with("arrow_table")
        table_write.prepare_commit.assert_called_once_with()
        table_commit.commit.assert_called_once_with(["commit_msg"])
        table_write.close.assert_called_once()
        table_commit.close.assert_called_once()

    def test_write_paimon_falls_back_to_json_when_pypaimon_missing(self):
        dataset = MagicMock(name="ray_dataset")

        with patch.object(RayExporter, "write_json", return_value="json_fallback") as mock_write_json:
            result = RayExporter.write_paimon(dataset, "/tmp/fallback.jsonl", export_extra_args={})

        mock_write_json.assert_called_once_with(dataset, "/tmp/fallback.jsonl")
        self.assertEqual(result, "json_fallback")


if __name__ == '__main__':
    unittest.main()
