import copy
from typing import Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

OP_NAME = "smiles_augmentation_mapper"


def _load_rdkit_chem():
    LazyLoader.check_packages(["rdkit"])
    from rdkit import Chem

    return Chem


@OPERATORS.register_module(OP_NAME)
class SmilesAugmentationMapper(Mapper):
    """Generates randomized but equivalent SMILES strings for molecular augmentation.

    This operator parses the input SMILES with RDKit and generates alternative SMILES
    serializations by randomizing the traversal order on the same molecular graph. The
    operator can either expand each input sample into multiple augmented samples or write
    the generated SMILES into another field while keeping the original one intact."""

    _batched_op = True
    _requirements = ["rdkit"]

    def __init__(
        self,
        aug_num: PositiveInt = 1,
        keep_original_sample: bool = True,
        *,
        smiles_key: str = "SMILES",
        output_key: Optional[str] = None,
        isomeric_smiles: bool = True,
        max_try: Optional[PositiveInt] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param aug_num: Number of augmented SMILES strings to generate for
            each input sample.
        :param keep_original_sample: Whether to keep the original sample in
            the output dataset.
        :param smiles_key: Input field name that stores the source SMILES
            string. It is 'SMILES' by default.
        :param output_key: Output field name used to store the randomized
            SMILES string. If None, overwrite the source SMILES field.
        :param isomeric_smiles: Whether to preserve stereochemistry and
            isotopic information when generating randomized SMILES.
        :param max_try: Maximum number of randomization attempts for each
            sample. If None, use max(aug_num * 10, 20).
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.aug_num = aug_num
        self.keep_original_sample = keep_original_sample
        self.smiles_key = smiles_key
        self.output_key = output_key or smiles_key
        self.isomeric_smiles = isomeric_smiles
        self.max_try = max_try or max(aug_num * 10, 20)

        # Let the tracer compare the field actually modified by this mapper.
        self.text_key = self.output_key

    def _generate_augmented_smiles(self, smiles: str):
        chem = _load_rdkit_chem()

        mol = chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES encountered in [{self._name}]: {smiles}")
            return []

        augmented_smiles = []
        seen = {smiles}

        for _ in range(self.max_try):
            if len(augmented_smiles) >= self.aug_num:
                break

            randomized_smiles = chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                isomericSmiles=self.isomeric_smiles,
            )
            if not randomized_smiles or randomized_smiles in seen:
                continue

            seen.add(randomized_smiles)
            augmented_smiles.append(randomized_smiles)

        return augmented_smiles

    def _build_original_sample(self, sample):
        original_sample = copy.deepcopy(sample)
        if self.output_key != self.smiles_key:
            original_sample[self.output_key] = sample.get(self.smiles_key, "")
        return original_sample

    def process_batched(self, samples):
        new_samples = []
        sample_num = len(samples[self.smiles_key])

        for i in range(sample_num):
            sample = {key: samples[key][i] for key in samples}
            smiles = sample.get(self.smiles_key, "")

            if self.keep_original_sample:
                new_samples.append(self._build_original_sample(sample))

            if not isinstance(smiles, str) or not smiles.strip():
                continue

            for augmented_smiles in self._generate_augmented_smiles(smiles.strip()):
                augmented_sample = copy.deepcopy(sample)
                augmented_sample[self.output_key] = augmented_smiles
                if self.output_key == self.smiles_key:
                    augmented_sample[self.smiles_key] = augmented_smiles
                new_samples.append(augmented_sample)

        if not new_samples:
            empty_samples = {key: [] for key in samples}
            if self.output_key not in empty_samples:
                empty_samples[self.output_key] = []
            return empty_samples

        res_samples = {}
        keys = new_samples[0].keys()
        for key in keys:
            res_samples[key] = [sample[key] for sample in new_samples]

        return res_samples
