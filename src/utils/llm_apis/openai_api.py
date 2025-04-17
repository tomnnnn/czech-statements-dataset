from .base import LanguageModelAPI
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm_asyncio
import logging
import pprint
import asyncio

logger = logging.getLogger(__name__)

class OpenAI_API(LanguageModelAPI):
    """
    Implementation of LanguageModelInterface using OpenAI's API
    """
    examples = []
    system_prompt = ""
    generation_prompt = ""

    def __init__(self,model_path="gpt-4o", **kwargs):
        super().__init__(model_path)

        self.show_progress = kwargs.get("show_progress", True)
        self.client = AsyncOpenAI(base_url=kwargs.get("api_base_url", None))
        self.sync_client = OpenAI(base_url=kwargs.get("api_base_url", None))

    async def _completion(self, chat, max_new_tokens):
        response = await self.client.chat.completions.create(
            model=self.model_path,
            messages=chat,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        return response.choices[0].message.content

    async def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        coroutines = [
            self._completion(chat, max_new_tokens)
            for chat in conversations
        ]

        results = await tqdm_asyncio.gather(*coroutines, desc="Collecting model responses", unit="response", disable=not self.show_progress)

        return results
    
    def _infer_sync(self, conversations, batch_size=8, max_new_tokens=1000):
        results = []
        for chat in conversations:
            response = self.sync_client.chat.completions.create(
                model=self.model_path,
                messages=chat,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )

            results.append(response.choices[0].message.content)

        return results
