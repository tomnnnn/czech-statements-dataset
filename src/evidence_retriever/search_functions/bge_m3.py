from .search_function import SearchFunction
from FlagEmbedding import BGEM3FlagModel, FlagModel
import numpy as np
import faiss
from typing import Literal

class BGE_M3(SearchFunction):
    def __init__(self, corpus, vector_type: Literal["dense_vecs", "colbert_vecs", "lexical_weights"] = "dense_vecs", **kwargs):
        super().__init__(corpus)
        self.corpus = corpus
        self.vector_type = vector_type

        # Initialize the model
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        # Get embeddings and ensure they're in float32 format for FAISS
        self.embeddings = np.array(self.model.encode([i["text"] for i in corpus], return_sparse=True)[self.vector_type], dtype=np.float32)
        
        # Optionally save the index later
        self.save_index = kwargs.get('save_index', False)
        self.index_path = kwargs.get('index_path', "")

        # Initialize the index
        self._index()

    def _index(self):
        if self.index_path:
            self.index = faiss.read_index(self.index_path)
        else:
            dim = self.embeddings.shape[1]  # Get the dimensionality of embeddings
            self.index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)  # Using inner product for similarity
            
            # Add embeddings to the index
            self.index.add(self.embeddings)

            # Save the index if needed
            if self.save_index:
                faiss.write_index(self.index, "index.faiss")

    def _search_index(self, query_embeddings: np.ndarray, k: int) -> list[list[dict]]:
        """Helper function to search the FAISS index and return results in the expected format."""
        dists, ids = self.index.search(query_embeddings, k)
        return [
            [{**self.corpus[idx], "score": score} for idx, score in zip(id_list, dist_list)]
            for id_list, dist_list in zip(ids, dists)
        ]

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Search a single query and return the top-k results."""
        query_embedding = np.array(
            self.model.encode_queries([query], return_dense=True)[self.vector_type], 
            dtype=np.float32
        )
        return self._search_index(query_embedding, k)[0]  # Extract first result since it's a single query

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[dict]]:
        """Search multiple queries in batch and return top-k results for each."""
        query_embeddings = np.array(
            self.model.encode(queries, return_dense=True)[self.vector_type], 
            dtype=np.float32
        )
        return self._search_index(query_embeddings, k)
