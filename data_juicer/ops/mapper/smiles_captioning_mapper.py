from collections import defaultdict
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = "smiles_captioning_mapper"


@OPERATORS.register_module(OP_NAME)
class SmilesCaptioningMapper(Mapper):
    """Generates molecule descriptions from SMILES strings with an API model.

    This operator turns a SMILES string into a short natural-language description. It is
    designed for chemistry datasets that contain fields such as `SMILES` and
    `description`, and uses an OpenAI-compatible API backend through Data-Juicer's unified
    API model wrapper."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a chemistry assistant that describes molecules from SMILES strings. "
        "Write concise, faithful captions grounded in the molecular structure only. "
        "Focus on obvious structural characteristics such as ring systems, aromaticity, "
        "charge state, and major functional groups. Do not invent biological roles, "
        "activities, or properties that are not directly inferable from the SMILES."
    )
    DEFAULT_INPUT_TEMPLATE = (
        "Given the following SMILES string, write a concise molecular caption in English. "
        "Return plain text only.\n\nSMILES: {smiles}"
    )

    def __init__(
        self,
        api_model: str = "gpt-4o",
        *,
        smiles_key: str = "SMILES",
        caption_key: str = "description",
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        overwrite: bool = False,
        try_num: PositiveInt = 3,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_model: API model name.
        :param smiles_key: Input field name that stores the source SMILES
            string. It is 'SMILES' by default.
        :param caption_key: Output field name used to store the generated
            molecular description. It is 'description' by default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the captioning task.
        :param input_template: Template for building the user input. It must
            contain the placeholder '{smiles}'.
        :param overwrite: Whether to overwrite an existing non-empty caption.
        :param try_num: Number of retries when the API call fails or returns an
            empty caption.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.2, 'top_p': 0.9}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.smiles_key = smiles_key
        self.caption_key = caption_key
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.overwrite = overwrite
        self.try_num = try_num
        self.sampling_params = sampling_params or {}

        self.model_key = prepare_model(
            model_type="api",
            model=api_model,
            endpoint=api_endpoint,
            response_path=response_path,
            **(model_params or {}),
        )

        # Let the tracer compare the generated caption field.
        self.text_key = self.caption_key

    def build_input(self, sample):
        mapping = defaultdict(str, sample)
        mapping["smiles"] = sample.get(self.smiles_key, "")
        return self.input_template.format_map(mapping)

    def parse_output(self, raw_output):
        if not isinstance(raw_output, str):
            return ""
        return raw_output.strip().strip('"').strip()

    def process_single(self, sample, rank=None):
        existing_caption = sample.get(self.caption_key, "")
        if isinstance(existing_caption, str) and existing_caption.strip() and not self.overwrite:
            sample[self.caption_key] = existing_caption
            return sample

        smiles = sample.get(self.smiles_key, "")
        if not isinstance(smiles, str) or not smiles.strip():
            sample[self.caption_key] = existing_caption if isinstance(existing_caption, str) else ""
            return sample

        client = get_model(self.model_key, rank=rank)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_input(sample)},
        ]

        generated_caption = existing_caption if isinstance(existing_caption, str) else ""
        for _ in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                parsed_output = self.parse_output(output)
                if parsed_output:
                    generated_caption = parsed_output
                    break
            except Exception as e:
                logger.warning(f"Exception: {e}")

        sample[self.caption_key] = generated_caption
        return sample
