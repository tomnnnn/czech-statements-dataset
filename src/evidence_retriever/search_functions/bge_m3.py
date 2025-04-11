from dataset_manager.models import Segment
from .search_function import SearchFunction
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import faiss
import pprint

import threading
import numpy as np
import faiss
import pprint
import logging

logger = logging.getLogger(__name__)

class BGE_M3(SearchFunction):
    def __init__(self, corpus, **kwargs):
        super().__init__(corpus, **kwargs)
        self.corpus = corpus

        # Initialize the lock
        self.lock = threading.Lock()

        pprint.pp(kwargs)

        # Initialize the model
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        

        # Initialize the index
        save_index = kwargs.get('save_index', False)
        load_index = kwargs.get('load_index', False)
        index_path = kwargs.get('index_path', "")

        self._index(corpus, save_index, load_index, index_path)



    def _index(self, corpus, save, load, path):
        if load:
            print(f"Loading index from {path}")
            self.index = faiss.read_index(path)
        else:
            print("Creating index from corpus")
            # Get embeddings and ensure they're in float32 format for FAISS
            embeddings = np.array(self.model.encode([i.text for i in corpus], return_dense=True)["dense_vecs"], dtype=np.float32)

            dim = embeddings.shape[1]  # Get the dimensionality of embeddings
            self.index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)  # Using inner product for similarity
            
            # Add embeddings to the index
            self.index.add(embeddings)

            # Save the index if needed
            if save:
                print(f"Saving index to {path}")
                faiss.write_index(self.index, path)


    def _encode_queries(self, queries):
        """ Generate embeddings for a list of texts using the model. """
        logger.debug(f"Encoding queries: {queries}")

        query_embedding = np.array(self.model.encode_queries(queries, return_dense=True)["dense_vecs"], dtype=np.float32)
        return query_embedding
    

    def _search_index(self, query_embeddings: np.ndarray, k: int) -> list[list[Segment]]:
        """Helper function to search the FAISS index and return results in the expected format."""
        logger.debug(f"Searching index. Query shape: {query_embeddings.shape}, k: {k}")

        _ , ids = self.index.search(query_embeddings, k)

        return [
            [self.corpus[idx] for idx in id_list if idx != -1]  # Skip invalid indices
            for id_list in ids
        ] # type: ignore


    def search(self, query: str | list, k: int = 10) -> list[Segment] | list[list[Segment]]:
        """Search query/queries and return the top-k results for it/each of them."""
        
        queries = [query] if isinstance(query, str) else query

        query_embeddings = self._encode_queries(queries)
        results = self._search_index(query_embeddings, k)
        
        return results if isinstance(query, list) else results[0]

