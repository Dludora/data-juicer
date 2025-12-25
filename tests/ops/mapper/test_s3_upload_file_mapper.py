import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch, ANY
from botocore.exceptions import ClientError

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# 请根据实际项目结构调整 import 路径
from data_juicer.ops.mapper.s3_upload_file_mapper import S3UploadFileMapper


class S3UploadFileMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        # 1. Patch boto3 client
        self.patcher = patch("boto3.client")
        self.mock_boto = self.patcher.start()

        # 2. Setup Mock S3 object
        self.mock_s3 = MagicMock()
        self.mock_boto.return_value = self.mock_s3

        # 3. 创建临时工作目录
        self.temp_dir = tempfile.mkdtemp()

        # 4. 创建一些虚拟的本地文件
        self.file1_name = "file1.txt"
        self.file1_path = os.path.join(self.temp_dir, self.file1_name)
        with open(self.file1_path, "w") as f:
            f.write("content 1")

        self.file2_name = "file2.jpg"
        self.file2_path = os.path.join(self.temp_dir, self.file2_name)
        with open(self.file2_path, "w") as f:
            f.write("content 2")

        # 5. 设置 AWS 凭证环境变量 (防止算子初始化报错)
        os.environ["AWS_ACCESS_KEY_ID"] = "test_key_id"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
        os.environ["AWS_REGION"] = "us-east-1"

        # 6. 通用参数
        self.bucket = "test-bucket"
        self.prefix = "dataset/"
        self.base_params = {
            "upload_field": "files",
            "s3_bucket": self.bucket,
            "s3_prefix": self.prefix,
        }

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # 清理环境变量
        del os.environ["AWS_ACCESS_KEY_ID"]
        del os.environ["AWS_SECRET_ACCESS_KEY"]
        del os.environ["AWS_REGION"]
        super().tearDown()

    def _run_op(self, ds_list, **kwargs):
        """辅助运行函数"""
        params = self.base_params.copy()
        params.update(kwargs)
        op = S3UploadFileMapper(**params)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        res_list = dataset.to_list()
        return sorted(res_list, key=lambda x: x["id"])

    def test_upload_basic(self):
        """测试基础上传功能：本地路径应变为 S3 URL"""
        ds_list = [{"files": [self.file1_path], "id": 1}]

        res_list = self._run_op(ds_list)

        # 预期 S3 URL
        expected_url = f"s3://{self.bucket}/{self.prefix}{self.file1_name}"

        # 1. 验证路径更新
        self.assertEqual(res_list[0]["files"][0], expected_url)

        # 2. 验证 boto3 upload_file 被调用
        # 参数: (LocalPath, Bucket, Key)
        self.mock_s3.upload_file.assert_called_with(self.file1_path, self.bucket, self.prefix + self.file1_name)

    def test_upload_nested_structure(self):
        """测试嵌套列表结构保持"""
        ds_list = [{"files": [[self.file1_path], self.file2_path], "id": 1}]

        res_list = self._run_op(ds_list)

        res_files = res_list[0]["files"]
        expected_url1 = f"s3://{self.bucket}/{self.prefix}{self.file1_name}"
        expected_url2 = f"s3://{self.bucket}/{self.prefix}{self.file2_name}"

        # 验证结构类型
        self.assertIsInstance(res_files, list)
        self.assertIsInstance(res_files[0], list)
        self.assertIsInstance(res_files[1], str)

        # 验证内容
        self.assertEqual(res_files[0][0], expected_url1)
        self.assertEqual(res_files[1], expected_url2)

    def test_skip_existing_true(self):
        """测试 skip_existing=True：当 S3 存在文件时，不执行上传"""
        # 模拟 head_object 成功（即文件存在）
        self.mock_s3.head_object.return_value = {}

        ds_list = [{"files": [self.file1_path], "id": 1}]

        res_list = self._run_op(ds_list, skip_existing=True)

        # 验证路径依然被更新为 S3 URL (这是预期行为)
        expected_url = f"s3://{self.bucket}/{self.prefix}{self.file1_name}"
        self.assertEqual(res_list[0]["files"][0], expected_url)

        # 关键验证：upload_file 不应该被调用
        self.mock_s3.upload_file.assert_not_called()
        # head_object 应该被调用
        self.mock_s3.head_object.assert_called()

    def test_skip_existing_false(self):
        """测试 skip_existing=False：即使 S3 存在文件，也强制上传"""
        self.mock_s3.head_object.return_value = {}  # 文件存在

        ds_list = [{"files": [self.file1_path], "id": 1}]

        self._run_op(ds_list, skip_existing=False)

        # 关键验证：upload_file 必须被调用
        self.mock_s3.upload_file.assert_called()

    def test_remove_local_after_upload(self):
        """测试 remove_local=True：上传成功后删除本地文件"""
        ds_list = [{"files": [self.file1_path], "id": 1}]

        # 确认文件开始时存在
        self.assertTrue(os.path.exists(self.file1_path))

        self._run_op(ds_list, remove_local=True)

        # 验证 boto3 被调用
        self.mock_s3.upload_file.assert_called()

        # 验证本地文件被删除
        self.assertFalse(os.path.exists(self.file1_path))

    def test_already_s3_url(self):
        """测试输入已经是 S3 URL 的情况：应保持原样且不上传"""
        existing_url = "s3://other-bucket/file.txt"
        ds_list = [{"files": [existing_url], "id": 1}]

        self.mock_s3.upload_file.reset_mock()

        res_list = self._run_op(ds_list)

        # 验证 URL 没变
        self.assertEqual(res_list[0]["files"][0], existing_url)

        # 验证没发生上传
        self.mock_s3.upload_file.assert_not_called()

    def test_local_file_not_found(self):
        """测试本地文件不存在的情况"""
        missing_path = os.path.join(self.temp_dir, "ghost.txt")
        ds_list = [{"files": [missing_path], "id": 1}]

        res_list = self._run_op(ds_list)

        # 失败时应保留原始路径
        self.assertEqual(res_list[0]["files"][0], missing_path)

        # 验证没调用上传
        self.mock_s3.upload_file.assert_not_called()

    def test_upload_failure(self):
        """测试 S3 上传抛出异常的情况"""
        # 模拟上传失败
        self.mock_s3.upload_file.side_effect = Exception("Network Error")

        ds_list = [{"files": [self.file1_path], "id": 1}]

        # 捕获日志或只检查结果
        res_list = self._run_op(ds_list)

        # 失败时保留本地路径
        self.assertEqual(res_list[0]["files"][0], self.file1_path)


if __name__ == "__main__":
    unittest.main()
