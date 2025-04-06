from ..retriever import Retriever
import dspy
import random
from typing import Literal

class MockRetriever(Retriever):
    def __init__(self, search_algorithm: Literal['bge-m3', 'mb25'], corpus, num_docs=10, **kwargs):
        self.corpus = corpus
        self.num_docs = num_docs
        pass

    def forward(self, statement: str) -> dspy.Prediction:
        # Randomly select num_docs segments from the corpus
        selected_segments = random.sample(self.corpus, k=self.num_docs)

        return dspy.Prediction(
            statement=statement,
            segments=selected_segments,
            scores=[random.uniform(0, 1) for _ in range(self.num_docs)],
            metadata={"retriever": "mock"}
        )

