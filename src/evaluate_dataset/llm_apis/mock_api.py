from ..llm_api import LanguageModelAPI
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

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000, **kwargs):
        labels = ["pravda", "nepravda"]

        return [labels[random.randint(0, 1)] for _ in range(len(conversations))]
