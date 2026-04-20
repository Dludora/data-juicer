import unittest

from data_juicer.core.metadata.provider import load_metadata_providers
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExecutorStub:
    event_logger = object()


class TestMetadataProviderLoading(DataJuicerTestCaseBase):
    def test_missing_external_provider_is_skipped(self):
        cfg = {
            "metadata": {
                "enabled": True,
                "providers": [
                    {"name": "provider_that_does_not_exist", "enabled": True, "config": {}}
                ],
            }
        }

        providers = load_metadata_providers(ExecutorStub(), cfg, "default")

        self.assertEqual(providers, [])


if __name__ == "__main__":
    unittest.main()
