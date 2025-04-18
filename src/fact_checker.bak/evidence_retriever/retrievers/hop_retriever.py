import logging
import dspy
from src.dataset_manager.models import Segment, Statement
from ..search_functions import SearchFunction

logger = logging.getLogger(__name__)


class GenerateQuery(dspy.Signature):
    výrok: str = dspy.InputField()
    autor: str = dspy.InputField()
    datum: str = dspy.InputField()
    co_zjistit: list[str] = dspy.InputField()

    query: str = dspy.OutputField()


class AppendNotes(dspy.Signature):
    výrok: str = dspy.InputField()
    autor: str = dspy.InputField()
    datum: str = dspy.InputField()
    co_zjistit: list[str] = dspy.InputField()
    nasbírané_texty: list[str] = dspy.InputField()

    co_dalšího_zjistit: list[str] = dspy.OutputField()


class HopRetriever(dspy.Module):
    def __init__(self, num_docs=4, num_hops=4, **kwargs):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.append_notes = dspy.ChainOfThought(AppendNotes)

    def _format_segments(self, segments: list[Segment]) -> list[dict]:
        return [
            {
                "title": segment.article.title[:3000],
                "text": segment.text[:3000],
                "url": segment.article.source[:3000],
            }
            for segment in segments
        ]

    def _extract_text(self, segments: list[Segment]) -> list[str]:
        return [segment.text for segment in segments]

    def forward(
        self, statement: Statement, search_func: SearchFunction
    ) -> dspy.Prediction:
        notes = [statement]
        retrieved_segments = []
        queries = []

        for i in range(self.num_hops):
            # Generate query and search for new segments
            print(f"Hop {i + 1} of {self.num_hops}")
            query = self.generate_query(
                výrok=statement.statement,
                autor=statement.author,
                datum=statement.date,
                co_zjistit=notes,
            ).query
            queries.append(query)

            new_segments = search_func.search(str(query), self.num_docs)

            # Update retrieved segments and texts
            retrieved_segments.extend(new_segments)

            # Update notes for the next iteration
            prediction = self.append_notes(
                výrok=statement.statement,
                autor=statement.author,
                datum=statement.date,
                co_zjistit=notes,
                nasbírané_texty=self._extract_text(retrieved_segments),
            )

            notes.extend(prediction.co_dalšího_zjistit)

        evidence = self._format_segments(retrieved_segments)

        return dspy.Prediction(
            segments=retrieved_segments,
            evidence=evidence,
            used_queries=queries,
            notes=notes,
        )
