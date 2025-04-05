from ..llm_api import LanguageModelAPI
import transformers
import random

class MockLanguageModelAPI(LanguageModelAPI):
    def __init__(
        self,
        model_path: str,
        distribute_load: bool = False,
        gpu_memory_utilization: float = 0.85,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)
        print(kwargs)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)


    def prepare_input(self, prompts):
        return prompts

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000, **kwargs):
        token_lengths = [self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1] for prompt in conversations]
        return [str(length) for length in token_lengths]
