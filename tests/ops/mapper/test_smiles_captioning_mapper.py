from copy import deepcopy
from unittest.mock import MagicMock, patch

from data_juicer.ops.mapper.smiles_captioning_mapper import SmilesCaptioningMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SmilesCaptioningMapperTest(DataJuicerTestCaseBase):
    def _run_op(self, samples, op):
        return [op.process(deepcopy(sample)) for sample in samples]

    @patch("data_juicer.ops.mapper.smiles_captioning_mapper.get_model")
    @patch("data_juicer.ops.mapper.smiles_captioning_mapper.prepare_model")
    def test_generate_caption(self, mock_prepare_model, mock_get_model):
        mock_prepare_model.return_value = "mock_model_key"
        mock_client = MagicMock(return_value="A small aliphatic alcohol.")
        mock_get_model.return_value = mock_client

        op = SmilesCaptioningMapper(num_proc=1)
        samples = [{"SMILES": "CCO"}]

        results = self._run_op(samples, op)

        self.assertEqual(results, [{"SMILES": "CCO", "description": "A small aliphatic alcohol."}])
        mock_client.assert_called_once()

    @patch("data_juicer.ops.mapper.smiles_captioning_mapper.get_model")
    @patch("data_juicer.ops.mapper.smiles_captioning_mapper.prepare_model")
    def test_skip_existing_caption(self, mock_prepare_model, mock_get_model):
        mock_prepare_model.return_value = "mock_model_key"

        op = SmilesCaptioningMapper(num_proc=1)
        samples = [{"SMILES": "CCO", "description": "Existing caption."}]

        results = self._run_op(samples, op)

        self.assertEqual(results, [{"SMILES": "CCO", "description": "Existing caption."}])
        mock_get_model.assert_not_called()

    @patch("data_juicer.ops.mapper.smiles_captioning_mapper.get_model")
    @patch("data_juicer.ops.mapper.smiles_captioning_mapper.prepare_model")
    def test_retry_caption_generation(self, mock_prepare_model, mock_get_model):
        mock_prepare_model.return_value = "mock_model_key"
        mock_client = MagicMock(side_effect=["", "A protonated amino alcohol."])
        mock_get_model.return_value = mock_client

        op = SmilesCaptioningMapper(overwrite=True, try_num=2, num_proc=1)
        samples = [{"SMILES": "NCCO", "description": "Old caption."}]

        results = self._run_op(samples, op)

        self.assertEqual(results, [{"SMILES": "NCCO", "description": "A protonated amino alcohol."}])
        self.assertEqual(mock_client.call_count, 2)
