from .base import LanguageModelAPI
import asyncio

class MockLanguageModelAPI(LanguageModelAPI):
    def __init__(
        self,
        model_path: str,
        **kwargs,
    ):
        super().__init__(model_path, **kwargs)


    def prepare_input(self, prompts):
        return prompts

    async def _infer(self, conversations, batch_size=8, max_new_tokens=1000, **kwargs):
        await asyncio.sleep(0.1)
        responses = ["Pravda" for _ in range(len(conversations))]
        return responses
