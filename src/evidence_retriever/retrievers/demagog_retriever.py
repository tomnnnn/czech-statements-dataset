import asyncio
import datetime
import sys

import aiohttp

from ..evidence_retriever import EvidenceRetriever


class DemagogRetriever(EvidenceRetriever):
    """
    Assumes evidence links are already scraped and stored in the statements database.
    """

    def __prepare_query(self, claim: str) -> str:
        return claim

    async def retrieve(self, statement: dict, top_k: int = 10) -> dict:
        return {
            "id": statement["id"],
            "query": None,
            "date": None,
            "results": statement['evidence_links'],
        }
