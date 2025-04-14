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


    def prepare_input(self, prompts):
        return prompts

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000, **kwargs):
        responses = ["Pravda" for _ in range(len(conversations))]
        return responses
