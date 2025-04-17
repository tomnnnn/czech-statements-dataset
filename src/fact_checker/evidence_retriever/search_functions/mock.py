from .search_function import SearchFunction
from src.dataset_manager.models import Segment
import asyncio
import time


class MockSearchFunction(SearchFunction):
    def __init__(self, corpus, model_name="BAAI/bge-m3", **kwargs):
        self.corpus = corpus

    def search(self, query: str | list, k: int = 10) -> list[Segment]:
        time.sleep(0.1)
        return [
            self.corpus[0]
            for _ in range(k)
        ]

    async def search_async(self, query: str | list, k: int = 10) -> list[Segment]:
        await asyncio.sleep(0.1)
        return [
            self.corpus[0]
            for _ in range(k)
        ]

    def index(self):
        time.sleep(0.1)

    async def index_async(self):
        await asyncio.sleep(0.1)


