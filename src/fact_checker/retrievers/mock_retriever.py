from src.dataset_manager.models import Segment
from ..search_functions import SearchFunction
import dspy
import random

class MockRetriever(dspy.Module):
    def __init__(self, num_docs=10, **kwargs):
        self.num_docs = num_docs

    def _format_segments(self, segments: list[Segment]) -> list[dict]:
        return [
            {
                "title": segment.article.title[:3000],
                "text": segment.text[:3000],
                "url": segment.article.source[:3000],
            }
            for segment in segments
        ]

    def forward(self, statement: str, corpus: list[Segment]) -> dspy.Prediction:
        # Randomly select num_docs segments from the corpus
        selected_segments = random.sample(corpus, k=self.num_docs)

        return dspy.Prediction(
            statement=statement,
            evidence=self._format_segments(selected_segments),
            segments=selected_segments,
        )

