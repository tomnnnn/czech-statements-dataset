from ..llm_api import LanguageModelAPI
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VLLM_Local(LanguageModelAPI):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=os.getenv("HF_TOKEN", None))
        self.model = LLM(model=model_path)



    def prepare_input(self, prompts):
        convos = super().prepare_input(prompts)
        return [self.tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos]

    def _infer(self, conversations: list, batch_size: int = 1, max_new_tokens: int = 512):
        sampling_params = SamplingParams(temperature=0, top_p=1, repetition_penalty=1.05, max_tokens=max_new_tokens)
        outputs = self.model.generate(conversations, sampling_params)

        return [output.outputs[0].text for output in outputs]
