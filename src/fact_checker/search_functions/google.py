import logging
from googleapiclient.discovery import build
import aiogoogle
import os
import aiogoogle

logger = logging.getLogger(__name__)

class GoogleSearch():
    def __init__(self, *args, **kwargs):
        self.api_key = os.environ.get("GOOGLE_API_KEY") or "EMPTY"
        self.service = build("customsearch", "v1", developerKey=self.api_key)

    def search(self, query: str, k: int = 10) -> list[dict]:
        res = self.service.cse().list(q=query, cx="f2513f07c25b448b7", num=k).execute()
        return res['items']

    async def search_async(self, query: str | list, k: int = 10) -> list[dict]:
        async with aiogoogle.Aiogoogle(api_key=self.api_key) as google:
            service = await google.discover("customsearch", "v1")
            response = await google.as_api_key(
                service.cse.list(q=query, cx="f2513f07c25b448b7", num=k)
            )
            return response['items']

    def __del__(self):
        self.service.close()
