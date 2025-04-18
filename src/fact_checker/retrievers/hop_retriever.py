import logging

import dspy

from src.dataset_manager.models import Segment, Statement

from ..search_functions import SearchFunction

logger = logging.getLogger(__name__)


class EstablishGoals(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()

    what_to_find: list[str] = dspy.OutputField()
    search_language: str = dspy.OutputField()


class GenerateQuery(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    what_to_find: list[str] = dspy.InputField()
    search_language: str = dspy.InputField()

    search_query: str = dspy.OutputField()


class AppendNotes(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    what_to_find: list[str] = dspy.InputField()
    collected_documents: list[str] = dspy.InputField()

    what_to_find: list[str] = dspy.OutputField()


class HopRetriever(dspy.Module):
    def __init__(self, num_docs=4, num_hops=4, **kwargs):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.establish_goals = dspy.ChainOfThought(EstablishGoals)
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
        logger.info("Establishing goals...")
        initial_thought = self.establish_goals(
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
        )

        what_to_find = initial_thought.what_to_find
        search_language = initial_thought.search_language
        retrieved_segments = []
        queries = []

        logger.info("Retrieving documents...")
        for i in range(self.num_hops):
            # Generate query and search for new segments
            logger.info(f"Hop {i + 1} of {self.num_hops}")
            query = (
                self.generate_query(
                    statement=statement.statement,
                    author=statement.author,
                    date=statement.date,
                    what_to_find=what_to_find,
                    search_language=search_language,
                )
            ).search_query

            queries.append(query)
            new_segments = search_func.search(str(query), self.num_docs)

            # Update retrieved segments and texts
            retrieved_segments.extend(new_segments)

            # Update notes for the next iteration
            prediction = self.append_notes(
                statement=statement.statement,
                author=statement.author,
                date=statement.date,
                what_to_find=what_to_find,
                collected_documents=self._extract_text(retrieved_segments),
            )

            what_to_find = prediction.what_to_find

        evidence = self._format_segments(retrieved_segments)

        return dspy.Prediction(
            segments=retrieved_segments,
            evidence=evidence,
            used_queries=queries,
            what_to_find=what_to_find,
        )
