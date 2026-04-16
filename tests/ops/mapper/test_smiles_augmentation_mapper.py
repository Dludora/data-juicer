from unittest.mock import patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.smiles_augmentation_mapper import SmilesAugmentationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SmilesAugmentationMapperTest(DataJuicerTestCaseBase):
    def _run_op(self, samples, op):
        dataset = Dataset.from_list(samples)
        return op.run(dataset).to_list()

    @patch(
        "data_juicer.ops.mapper.smiles_augmentation_mapper."
        "SmilesAugmentationMapper._generate_augmented_smiles"
    )
    def test_keep_original_sample(self, mock_generate_augmented_smiles):
        mock_generate_augmented_smiles.return_value = ["OCC", "C(O)C"]

        op = SmilesAugmentationMapper(aug_num=2, keep_original_sample=True, num_proc=1)
        samples = [{"SMILES": "CCO", "description": "ethanol"}]

        results = self._run_op(samples, op)
        expected = [
            {"SMILES": "CCO", "description": "ethanol"},
            {"SMILES": "OCC", "description": "ethanol"},
            {"SMILES": "C(O)C", "description": "ethanol"},
        ]

        self.assertEqual(results, expected)

    @patch(
        "data_juicer.ops.mapper.smiles_augmentation_mapper."
        "SmilesAugmentationMapper._generate_augmented_smiles"
    )
    def test_output_key(self, mock_generate_augmented_smiles):
        mock_generate_augmented_smiles.return_value = ["OCC"]

        op = SmilesAugmentationMapper(
            aug_num=1,
            keep_original_sample=False,
            output_key="augmented_smiles",
            num_proc=1,
        )
        samples = [{"SMILES": "CCO", "description": "ethanol"}]

        results = self._run_op(samples, op)
        expected = [
            {
                "SMILES": "CCO",
                "description": "ethanol",
                "augmented_smiles": "OCC",
            }
        ]

        self.assertEqual(results, expected)

    @patch(
        "data_juicer.ops.mapper.smiles_augmentation_mapper."
        "SmilesAugmentationMapper._generate_augmented_smiles"
    )
    def test_no_augmented_smiles(self, mock_generate_augmented_smiles):
        mock_generate_augmented_smiles.return_value = []

        op = SmilesAugmentationMapper(keep_original_sample=True, num_proc=1)
        samples = [{"SMILES": "invalid-smiles", "description": "broken"}]

        results = self._run_op(samples, op)
        expected = [{"SMILES": "invalid-smiles", "description": "broken"}]

        self.assertEqual(results, expected)
