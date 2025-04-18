from dataset_manager.models import Segment
from .base import SearchFunction
import numpy as np
import faiss
import torch
import numpy as np
import faiss
import logging
import numpy as np
import faiss
import asyncio

logger = logging.getLogger(__name__)

class BGE_M3(SearchFunction):
    def __init__(self, corpus, model_name="BAAI/bge-m3", **kwargs):
        super().__init__(corpus, **kwargs)
        self.corpus = corpus

        self.model=kwargs.get('model', None)

        # Save/load config
        self.save_index = kwargs.get('save_index', False)
        self.load_index = kwargs.get('load_index', False)
        self.index_path = kwargs.get('index_path', "")

        self.index = None

    def _encode_documents(self, documents):
        return self.model.encode(documents, convert_to_numpy=True)

    async def _encode_documents_async(self, documents):
        return await asyncio.to_thread(self.model.encode, documents, convert_to_numpy=True)

    def _search_index(self, query_embeddings: np.ndarray, k: int) -> list[list[Segment]]:
        # TODO: simplify to single search
        _, ids = self.index.search(query_embeddings, k)
        try:
            return [
                [self.corpus[idx] for idx in id_list if idx != -1]
                for id_list in ids
            ]  # type: ignore
        except Exception as e:
            print(f"Error in search: {e}")
            print(len(self.corpus))
            print(ids)
            return [[self.corpus[0]]]

    def search(self, query: str, k: int = 10) -> list[Segment]:
        query_embeddings = self._encode_documents([query])
        results = self._search_index(query_embeddings, k)

        torch.cuda.empty_cache()
        return results[0]

    async def search_async(self, query: str | list, k: int = 10) -> list[Segment]:
        query_embeddings = await self._encode_documents_async([query])
        results = self._search_index(query_embeddings, k)

        torch.cuda.empty_cache()
        return results[0]

    def create_index(self):
        if self.load_index and self.index_path:
            self.index = faiss.read_index(self.index_path)
        else:
            embeddings = np.array(self._encode_documents([i.text for i in self.corpus]), dtype=np.float32)
            dim = embeddings.shape[1]
            self.index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            self.index.add(embeddings)
            if self.save_index and self.index_path:
                faiss.write_index(self.index, self.index_path)

    async def index_async(self):
        """Asynchronous version of the index builder."""
        if self.load_index and self.index_path:
            self.index = await asyncio.to_thread(faiss.read_index, self.index_path)
        else:
            print("Creating index...")
            texts = [i.text for i in self.corpus]
            embeddings = np.array(await self._encode_documents_async(texts), dtype=np.float32)
            dim = embeddings.shape[1]
            self.index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            self.index.add(embeddings)
            if self.save_index and self.index_path:
                await asyncio.to_thread(faiss.write_index, self.index, self.index_path)

            torch.cuda.empty_cache()
