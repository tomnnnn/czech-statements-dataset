from .bge_m3 import BGE_M3
import asyncio
from tqdm.asyncio import tqdm_asyncio
from .bm25 import BM25
import os
from dataset_manager import Dataset
from FlagEmbedding import FlagReranker

class HybridSearch():
    def __init__(self, dataset_path: str):
        dataset = Dataset(dataset_path)
        self.dense_retriever = BGE_M3(128)
        self.sparse_retriever = BM25()
        self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

        statements = dataset.get_statements()[:3]
        statement_ids = [s.id for s in statements]
        self.segment_map = dataset.get_segments_by_statements(statement_ids)

    async def create_indices(self):
        dense_create_tasks = [
            self.dense_retriever.create_index(
               [segment.text for segment in segments], 
                os.path.join("indices_hybrid", "dense", f"{statement_id}.faiss")
            )
            for statement_id, segments in self.segment_map.items()
        ]

        sparse_create_tasks = [
            self.sparse_retriever.create_index(
                [segment.text for segment in segments],
                os.path.join("indices_hybrid", "sparse", str(statement_id))
            )
            for statement_id, segments in self.segment_map.items()
        ]

        await tqdm_asyncio.gather(*dense_create_tasks)
        await tqdm_asyncio.gather(*sparse_create_tasks)


    async def load_indices(self, statement_ids: list[int]|None = None):
        """
        Loads the indices for the given statement IDs.

        Args:
            statement_ids (list): List of statement IDs to load.
        """
        if not statement_ids:
            dataset = Dataset(os.path.join(os.environ.get("SCRATCHDIR", "datasets"), "dataset.db"))
            statements = dataset.get_statements()
            statement_ids = [s.id for s in statements]


        for statement_id in statement_ids:
            await self.dense_retriever.add_index(
                self.segment_map[statement_id],
                os.path.join("indices_hybrid", "dense", f"{statement_id}.faiss"),
                load_if_exists=True,
                save=False,
                key=statement_id
            )

            await self.sparse_retriever.add_index(
                self.segment_map[statement_id],
                os.path.join("indices_hybrid", "sparse", str(statement_id)),
                load_if_exists=True,
                save=False,
                key=statement_id
            )

    def search(
        self,
        query,
        statement_id,
        k=3,
    ):

        dense_results = self.dense_retriever.search(
            query,
            k=k,
            key=statement_id
        )

        sparse_results = self.sparse_retriever.search(
            query,
            k=k,
            key=statement_id
        )

        # Combine the results
        combined_results = dense_results + sparse_results
        sentence_pairs = [
            (query, result.text) for result in combined_results
        ]

        # Rerank the combined results
        scores = self.reranker.compute_score(sentence_pairs)

        # Sort the results based on the scores
        sorted_results = sorted(zip(combined_results, scores), key=lambda x: x[1], reverse=True)

        # Extract the sorted text results
        sorted_combined_results = [result[0] for result in sorted_results]

        # Return the sorted results (or use as needed)
        return sorted_combined_results
