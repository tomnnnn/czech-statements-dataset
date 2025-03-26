import os
import json
import torch
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..llm_api import LanguageModelAPI

logger = logging.getLogger(__name__)

class VLLM_Local(LanguageModelAPI):
    def __init__(
        self,
        model_path: str,
        distribute_load: bool = False,
        gpu_memory_utilization: float = 0.95,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} gpus")

        # Get rope scaling configuration if provided
        rope_scaling = self._get_rope_config(kwargs.get("rope_scaling", None))

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LLM(
            model=kwargs.get("model_file", model_path) or model_path,
            tokenizer=model_path,
            tensor_parallel_size=num_gpus if distribute_load else 1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=kwargs.get("ctx_len", None),
            rope_scaling=rope_scaling,
        )

    def _get_rope_config(self, rope_config):
        """
        Get rope scaling configuration from the input argument.
        """
        if rope_config:
            if isinstance(rope_config, str):
                try:
                    rope_scaling = json.loads(rope_config)  # Convert JSON string to dict
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format for rope_scaling argument.")
            elif isinstance(rope_config, dict):
                rope_scaling = rope_config  # Use directly if already a dictionary
                logger.info("Using following rope scaling configuration: %s", rope_scaling)
            else:
                raise TypeError("rope_scaling must be a dictionary or a valid JSON string.")
        else:
            rope_scaling = None  # Disable rope scaling if not provided

        return rope_scaling


    def prepare_input(self, prompts):
        convos = super().prepare_input(prompts)

        if self.chat_format:
            convos =  [
                self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in convos
            ]

        return convos

    def _infer(
        self, conversations: list, batch_size: int = 1, max_new_tokens: int = 512
    ):
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=1,
            repetition_penalty=1.05,
            top_k=1,
            seed=42,
            max_tokens=max_new_tokens,
        )

        outputs = self.model.generate(conversations, sampling_params)

        return [output.outputs[0].text for output in outputs]
