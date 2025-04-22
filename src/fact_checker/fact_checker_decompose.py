import json
import logging
from typing import Literal

import dspy
import mlflow
from src.dataset_manager.models import Segment, Statement

from fact_checker.search_functions.bge_remote import RemoteSearchFunction

from .retrievers import HopRetriever

logger = logging.getLogger(__name__)


class VeracityTernary(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()

    label: Literal["pravda", "nepravda", "neověřitelné"] = dspy.OutputField()


class VeracityBinary(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()

    label: Literal["pravda", "nepravda"] = dspy.OutputField()


class Decompose(dspy.Signature):
    text: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()

    statements: list[str] = dspy.OutputField(
        description="A list of contextually complete, atomic, and verifiable factual statements from the input. "
            "Each statement must stand alone with enough context for verification, be neutrally phrased, and remain in the original language. "
            "Include only factual claims — omit rhetorical, speculative, or personal experience-based content that cannot be independently verified."
    )

class Analyze(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()

    what_to_find: list[str] = dspy.OutputField(
        description="A list of specific informations or evidence to search for in the documents. "
        "This should be a concise and clear statement of what the user is looking for."
    )
    search_language: Literal["en", "cs", "sk"] = dspy.OutputField(
        description="The language in which the search queries should be generated."
    )


class GenerateQueries(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    what_to_find: list[str] = dspy.InputField(
        description="A list of specific informations or evidence to search for in the documents."
    )
    search_language: str = dspy.InputField(
        description="The language in which the search queries should be generated."
    )

    search_queries: list[str] = dspy.OutputField(
        description="List of search queries to be used to retrieve needed documents from an index."
    )


class Retriever(dspy.Module):
    def __init__(self, num_docs=4, **kwargs):
        self.num_docs = num_docs
        self.analyze_statement = dspy.ChainOfThought(Analyze)
        self.generate_queries = dspy.ChainOfThought(GenerateQueries)
        self.doc_retriever = RemoteSearchFunction()

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
        self, statement_id: int, statement: str, author: str, date: str
    ) -> dspy.Prediction:
        analysis = self.analyze_statement(statement=statement, author=author, date=date)
        what_to_find = analysis.what_to_find
        search_language = analysis.search_language

        queries = self.generate_queries(
            statement=statement,
            author=author,
            date=date,
            what_to_find=what_to_find,
            search_language=search_language,
        ).search_queries

        segments = []
        for query in queries:
            with mlflow.start_span("document_retrieval") as span:
                span.set_inputs(
                    {
                        "query": query,
                        "k": self.num_docs,
                        "key": statement_id,
                    }
                )
                new_segments = self.doc_retriever.search(
                    query=query, k=self.num_docs, key=statement_id
                )
                span.set_outputs(
                    {
                        "segments": [
                            {"segment_id": segment.id, "text": segment.text}
                            for segment in new_segments
                        ]
                    }
                )

            segments.extend(new_segments)

        # filter duplicates
        segments = list({segment.id: segment for segment in segments}.values())

        # format segments into a list of dictionaries
        evidence = self._format_segments(segments)

        return dspy.Prediction(
            evidence=evidence,
            used_queries=queries,
            what_to_find=what_to_find,
        )


class FactCheckerDecomposer(dspy.Module):
    def __init__(
        self,
        num_docs=4,
        mode: Literal["ternary", "binary"] = "ternary",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.retriever = Retriever(num_docs=num_docs)

        veracity = VeracityTernary if mode == "ternary" else VeracityBinary

        self.decompose = dspy.Predict(Decompose)
        self.retrieve = Retriever(num_docs=num_docs)
        self.classify = dspy.ChainOfThought(veracity)

    def forward(self, statement: Statement) -> dspy.Prediction:
        decomposed_statements = self.decompose(
            text=statement.statement,
            author=statement.author,
            date=statement.date,
        ).statements

        """
        2 versions: 
            1] broken down statements will be each verified seperately
            2) broken down statements will be verified together with joined evidence
        """

        # Version 2: Verified together
        evidence = []
        used_queries = []
        for decomp_statement in decomposed_statements:
            retrieve_result = self.retrieve(
                statement_id=statement.id,
                statement=decomp_statement,
                author=statement.author,
                date=statement.date,
            )

            used_queries.extend(retrieve_result.used_queries)
            evidence.extend(retrieve_result.evidence)

        label = self.classify(
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
            evidence=evidence,
        ).label

        return dspy.Prediction(
            statement=statement.statement,
            atomic_statements=decomposed_statements,
            evidence=evidence,
            label=label,
        )
