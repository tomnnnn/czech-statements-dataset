from ..llm_api import LanguageModelAPI
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch


class VLLM_Local(LanguageModelAPI):
    def __init__(self, model_path: str, distribute_load: bool = False, gpu_memory_utilization: float = 0.95, **kwargs):
        super().__init__(model_path)
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} gpus")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            token=os.getenv("HF_TOKEN", None)
        )

        self.model = LLM(
            model=kwargs.get("filename", model_path),
            tokenizer=model_path,
            tensor_parallel_size=num_gpus if distribute_load else 1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=50000,
        )


    def prepare_input(self, prompts):
        convos = super().prepare_input(prompts)

        return [self.tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos]

    def _infer(self, conversations: list, batch_size: int = 1, max_new_tokens: int = 512):
        sampling_params = SamplingParams(temperature=0.2, top_p=1, repetition_penalty=1.05, top_k=1, seed=42, max_tokens=max_new_tokens)
        outputs = self.model.generate(conversations, sampling_params)

        return [output.outputs[0].text for output in outputs]
