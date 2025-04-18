from dataset_manager.models import Segment
from .search_function import SearchFunction
import numpy as np
import numpy as np
import faiss
import logging
import numpy as np
import faiss
import aiohttp
from googleapiclient.discovery import build
import aiogoogle
import os

logger = logging.getLogger(__name__)

class Google(SearchFunction):
    def __init__(self, *args, **kwargs):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.service = build("customsearch", "v1", developerKey=self.api_key)


    def search(self, query: str, k: int = 10) -> list[Segment]:
        res = self.service.cse().list(q=query, cx="f2513f07c25b448b7", num=k).execute()
        return res['items']

    async def search_async(self, query: str | list, k: int = 10) -> list[Segment]:
        async with aiogoogle.Aiogoogle(api_key=self.api_key) as google:
            service = await google.discover("customsearch", "v1")
            response = await google.as_api_key(
                service.cse.list(q=query, cx="f2513f07c25b448b7", num=k)
            )
            return response['items']

    def __del__(self):
        self.service.close()
