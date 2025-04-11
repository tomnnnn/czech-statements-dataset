from ..llm_api import LanguageModelAPI
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import requests
import time
import logging
import pprint

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
        # wait for server to be up by checking api_base_url/health
        
        # while True:
        #     try:
        #         response = requests.get("http://0.0.0.0:8000/health")
        #
        #         if response.status_code == 200:
        #             print("Server is up and running.")
        #             break
        #         else:
        #             print(f"Server is not ready yet, status code: {response.status_code}")
        #             time.sleep(10)
        #     except Exception as e:
        #         print(f"Waiting for server to be up: {e}...")
        #         time.sleep(10)

        self.client = AsyncOpenAI(base_url=kwargs.get("api_base_url", None))

    async def _completion(self, chat, max_new_tokens):
        response = await self.client.chat.completions.create(
            model=self.model_path,
            messages=chat,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        return response.choices[0].message.content

    async def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        logger.info("Running inference")
        coroutines = [
            self._completion(chat, max_new_tokens)
            for chat in conversations
        ]

        results = await tqdm_asyncio.gather(*coroutines, desc="Collecting model responses", unit="response")

        pprint.pp(results)
        logger.info("Returning results")

        return results
