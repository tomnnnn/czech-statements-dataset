from typing import Optional, TypedDict
import os
import bm25s
import simplemma

from dataset_manager.models import Segment
from .base import SearchFunction

class IndexEntry(TypedDict):
    index: bm25s.BM25
    corpus: list[Segment]

class BM25(SearchFunction):
    indices: dict[str|int, IndexEntry]

    def __init__(self, **kwargs):
        self.indices = { }


    def _stemmer(self, words):
        return [simplemma.lemmatize(word, "cs") for word in words]


    async def create_index(self, texts: list[str], save_dir: str):
        """
        Creates a BM25 index from the given texts and saves it to the specified path.

        Args:
            texts (list[str]): List of texts to index.
            save_path (str): Path to save the index.
        """
        tokenized_texts = bm25s.tokenize(texts, stemmer=self._stemmer)
        index = bm25s.BM25(k1=0.9, b=0.4)
        index.index(tokenized_texts)
        index.save(save_dir)


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
            index = bm25s.BM25.load(save_path)
            self.indices[key] = {
                "index": index,
                "corpus": segments
            }
            return

        tokenized_segments = bm25s.tokenize([i.text for i in segments], stemmer=self._stemmer)
        index = bm25s.BM25(k1=0.9, b=0.4)
        index.index(tokenized_segments)

        self.indices[key] = {
            "index": index,
            "corpus": segments
        }


    def search(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        index = self.indices[key]["index"]
        corpus = self.indices[key]["corpus"]

        tokenized_query = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)
        ids, scores = index.retrieve(tokenized_query, k=k, n_threads=10, show_progress=False)

        result = [
            corpus[idx]
            for idx,_ in zip(ids[0], scores[0])
        ]

        return result[0]

    async def search_async(self, query: str, k: int = 10, key: str|int = "_default") -> list[Segment]:
        # TODO: Implement async search
        return self.search(query, k=k, key=key)
