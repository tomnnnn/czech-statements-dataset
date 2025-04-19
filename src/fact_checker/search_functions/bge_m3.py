from typing import Optional, TypedDict, Union
import asyncio
import numpy as np
from sqlalchemy import Index
import torch
from dataset_manager.models import Segment
from sentence_transformers import SentenceTransformer
import os
import faiss
from fact_checker.search_functions.base import SearchFunction

class IndexEntry(TypedDict):
    index: faiss.Index
    corpus: list[Segment]


class BGE_M3(SearchFunction):
    indices: dict[str|int, IndexEntry]
    model: SentenceTransformer

    def __init__(self, model: SentenceTransformer, batch_size = 32, **kwargs):
        self.model = model
        self.indices = {}
        self.batch_size = batch_size
        self.sem = asyncio.Semaphore(50)

    def _encode_documents(self, documents: list[str]) -> np.ndarray:
        return self.model.encode(documents, convert_to_numpy=True)

    async def _safe_encode_batch(self, batch: list[str]) -> np.ndarray:
        size = len(batch)
        while size > 0:
            try:
                async with self.sem:
                    return await asyncio.to_thread(self._encode_documents, batch[:size])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM on batch of size {size}. Reducing...")
                    await asyncio.sleep(1)  # give memory time to clear
                    size = size // 2
                else:
                    raise e

        print(f"Could not encode even a single document in batch: {batch}")
        return np.array([])

    async def _encode_documents_async(self, documents: list[str]) -> np.ndarray:
        batches = [documents[i:i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        results = await asyncio.gather(*(self._safe_encode_batch(batch) for batch in batches))
        return np.vstack(results)


    def key_exists(self, key: str|int = "_default") -> bool:
        """
        Check if the key exists in the indices dictionary.

        Args:
            key (str): Key for the index.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.indices

    async def add_index(self, segments: list[Segment], save_path: Optional[str], load_if_exists: bool, save: bool, key: str|int = "_default"):
        """
        Creates or loads an index and adds it to internal indices dictionary.

        Args:
            segments (list[Segment]): List of segments to index.
            key (str): Key for the index.
            save_path (Optional[str]): Path to save the index.
            load (Union[Literal["auto"], bool]): Whether to load the index if it exists.
        """
        if load_if_exists and save_path and os.path.exists(save_path):
            index = faiss.read_index(save_path)
        else:
            texts = [i.text for i in segments]
            embeddings = await self._encode_documents_async(texts)
            dim = embeddings.shape[1]

            index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            index.add(embeddings)

            if save and save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                faiss.write_index(index, save_path)

            torch.cuda.empty_cache()

        self.indices[key] = {
            "index": index,
            "corpus": segments
        }


    def _search_index(self, query_embeddings: np.ndarray, k: int, key: str|int = "_default") -> list[list[Segment]]:
        index = self.indices[key]["index"]
        corpus = self.indices[key]["corpus"]

        _, ids = index.search(query_embeddings, k=k)
        try:
            return [
                [corpus[idx] for idx in id_list if idx != -1]
                for id_list in ids
            ]  # type: ignore
        except Exception as e:
            print(f"Error in search: {e}")
            print(len(corpus))
            print(ids)
            return [[corpus[0]]]


    def search(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        query_embeddings = self.model.encode([query], convert_to_numpy=True, batch_size=32)
        results = self._search_index(query_embeddings, k, key)

        torch.cuda.empty_cache()
        return results[0]

    async def search_async(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        query_embeddings = await self._encode_documents_async([query])
        results = self._search_index(query_embeddings, k, key)

        torch.cuda.empty_cache()
        return results[0]

