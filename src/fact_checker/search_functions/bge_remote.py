from dataset_manager.models import Segment
from .base import SearchFunction
import logging
from typing import Optional
import aiohttp
import requests


logger = logging.getLogger(__name__)

class Remote(SearchFunction):
    def __init__(self, api_base="http://127.0.0.1:4242", **kwargs):
        self.api_base = api_base
        self.session = aiohttp.ClientSession()

    async def add_index(self, segments: list[Segment], save_path: Optional[str], load_if_exists: bool, save: bool, key: str|int = "_default"):
        """
        Creates or loads an index and adds it to internal indices dictionary.

        Args:
            segments (list[Segment]): List of segments to index.
            key (str): Key for the index.
            save_path (Optional[str]): Path to save the index.
            load (Union[Literal["auto"], bool]): Whether to load the index if it exists.
        """

    def search(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        """
        Searches the index for the given query.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.
            key (str): The key for the index.

        Returns:
            list[Segment]: A list of segments matching the query.
        """

        response = requests.post(f"{self.api_base}/search", json={"query": query, "k": k, "statement_id": key})
        if response.status_code != 200:
            logger.error("Failed to search the index")
            return []

        json_resp = response.json()
        segments = [Segment(**item["data"]) for item in json_resp["results"]]

        return segments



    async def search_async(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        async with self.session.post(f"{self.api_base}/search", json={"query": query, "k": k, "statement_id": key}) as response:
            if response.status != 200:
                logger.error("Failed to search the index")
                return []

            json_resp = await response.json()
            segments = [Segment(**item["data"]) for item in json_resp["results"]]

            return segments
