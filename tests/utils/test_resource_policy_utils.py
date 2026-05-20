import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.nltk_utils import ensure_nltk_resource
from data_juicer.utils.resource_policy_utils import (
    ResourceResolutionError,
    get_resource_policy,
    resolve_asset_source,
    resolve_model_source,
    should_allow_public_fallback,
    should_auto_install_package,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ResourcePolicyUtilsTest(DataJuicerTestCaseBase):
    def test_default_policy(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in [
                "DJ_RESOURCE_OFFLINE_MODE",
                "DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK",
                "DJ_RESOURCE_LOCAL_CACHE_ROOTS",
                "DJ_MODEL_BASE_URL",
                "DJ_ASSET_BASE_URL",
                "DJ_HF_ENDPOINT",
                "DJ_HF_HOME",
                "DJ_HF_LOCAL_FILES_ONLY",
                "DJ_NLTK_DATA_DIR",
                "DJ_NLTK_ALLOW_DOWNLOAD",
                "DJ_PACKAGE_AUTO_INSTALL",
            ]:
                os.environ.pop(key, None)
            policy = get_resource_policy()
            self.assertFalse(policy["offline_mode"])
            self.assertTrue(should_allow_public_fallback(policy))
            self.assertTrue(should_auto_install_package(policy))
            self.assertIsNone(policy["model_base_url"])
            self.assertIsNone(policy["asset_base_url"])

    def test_offline_policy(self):
        with patch.dict(
            os.environ,
            {
                "DJ_RESOURCE_OFFLINE_MODE": "true",
                "DJ_RESOURCE_ALLOW_PUBLIC_FALLBACK": "true",
                "DJ_PACKAGE_AUTO_INSTALL": "true",
            },
            clear=False,
        ):
            policy = get_resource_policy()
            self.assertTrue(policy["offline_mode"])
            self.assertFalse(should_allow_public_fallback(policy))
            self.assertFalse(should_auto_install_package(policy))

    def test_resolve_model_source_from_local_cache_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_name = "test-model.bin"
            model_path = os.path.join(tmpdir, model_name)
            with open(model_path, "w") as fout:
                fout.write("ok")

            with patch.dict(os.environ, {"DJ_RESOURCE_LOCAL_CACHE_ROOTS": tmpdir}, clear=False):
                source = resolve_model_source(model_name)
                self.assertEqual(source["kind"], "local_path")
                self.assertEqual(source["source"], "local_cache_root")
                self.assertEqual(source["value"], model_path)

    def test_resolve_asset_source_offline_without_local_copy(self):
        with patch.dict(os.environ, {"DJ_RESOURCE_OFFLINE_MODE": "true"}, clear=False):
            with self.assertRaises(ResourceResolutionError):
                resolve_asset_source("stopwords")

    def test_resolve_asset_source_from_mirror(self):
        with patch.dict(os.environ, {"DJ_ASSET_BASE_URL": "https://mirror.example.com/data_juicer"}, clear=False):
            source = resolve_asset_source("stopwords")
            self.assertEqual(source["kind"], "remote_url")
            self.assertEqual(source["source"], "mirror")
            self.assertEqual(source["value"], "https://mirror.example.com/data_juicer/stopwords.json")


class ResourcePolicyNltkTest(DataJuicerTestCaseBase):
    def test_nltk_download_blocked(self):
        mock_nltk = MagicMock()
        mock_nltk.data.path = []
        mock_nltk.data.find.side_effect = LookupError()
        mock_nltk.download = MagicMock()

        with patch.dict("sys.modules", {"nltk": mock_nltk}):
            with patch.dict(
                os.environ,
                {
                    "DJ_RESOURCE_OFFLINE_MODE": "true",
                    "DJ_NLTK_ALLOW_DOWNLOAD": "false",
                },
                clear=False,
            ):
                ok = ensure_nltk_resource("tokenizers/punkt/english.pickle", "punkt")
                self.assertFalse(ok)
                mock_nltk.download.assert_not_called()


class ResourcePolicyLazyLoaderTest(DataJuicerTestCaseBase):
    def test_auto_install_blocked_by_policy(self):
        with patch.dict(os.environ, {"DJ_PACKAGE_AUTO_INSTALL": "false"}, clear=False):
            with patch("data_juicer.utils.lazy_loader.importlib.import_module", side_effect=ImportError):
                with self.assertRaises(ImportError) as ctx:
                    LazyLoader.check_packages(["definitely_not_installed_package"])
                self.assertIn("auto installation is disabled", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
