import bm25s
import simplemma

from dataset_manager.models import Segment
from .search_function import SearchFunction

class BM25(SearchFunction):
    def __init__(self, corpus, **kwargs):
        super().__init__(corpus)
        self.corpus = corpus
        self._index()


    def _stemmer(self, words):
        return [simplemma.lemmatize(word, "cs") for word in words]


    def _index(self):
        corpus_tokens = bm25s.tokenize([item.text for item in self.corpus], stemmer=self._stemmer)
        retriever = bm25s.BM25(k1=0.9, b=0.4)
        retriever.index(corpus_tokens)

        self.retriever = retriever


    def search(self, query: str|list, k: int = 10) -> list[Segment]|list[list[Segment]]:
        query = [query] if isinstance(query, str) else query

        results = []

        for q in query:
            tokens = bm25s.tokenize(q, stemmer=self._stemmer, show_progress=False)
            indices, scores = self.retriever.retrieve(tokens, k=k, n_threads=10, show_progress=False)
            run = [
                self.corpus[idx]
                for idx,_ in zip(indices[0], scores[0])
            ]
            results.append(run)

        return results if isinstance(query, list) else results[0]
