import asyncio
import logging

import dspy
import mlflow
from dataset_manager.models import Segment, Statement

from ..link_entities import detect_entities, get_abstracts
from ..search_functions.remote import RemoteSearchFunction

logger = logging.getLogger(__name__)


class GenerateQuery(dspy.Signature):
    """
    Generate a search query in a way to retrieve the most relevant documents that
    potentially support or refute the statement. Based on already used queries and info field,
    focus on capturing parts of context that are not yet covered by the retrieved documents.
    Avoid adding bias to query by suggesting truthfulness or falsity of the statement (do not use "is it true that...").
    Choose language according to expected language of the documents.
    """

    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    info: str = dspy.InputField(
        description="A summary of the known context and relevant information about the statement."
    )
    already_used_queries: list[str] = dspy.InputField(
        description="Do not use these queries again."
    )

    nl_query: str = dspy.OutputField(
        description="A natural language search query, focusing on one specific aspect of the statement."
    )
    keyword_query: str = dspy.OutputField(description="A keyword based search query")


class UpdateInfo(dspy.Signature):
    """
    Update the current information based on the new evidence. The main purpose of information
    is to inform future searches and to provide a summary of the current state of knowledge.
    Extract relevant and important information (if there is any) but do not assess the veracity
    of the statement.
    """

    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    current_info: list[str] = dspy.InputField(
        description="Currently known information about the statement and its context."
    )
    new_evidence: str = dspy.InputField()

    updated_info: list[str] = dspy.OutputField(
        description="Updated information summary based on the new evidence. Can stay the same if the evidence is not relevant."
    )
    terminate: bool = dspy.OutputField(
        description="A flag indicating whether we have enough information to support or refute the statement."
    )


class HopRetriever(dspy.Module):
    def __init__(
        self,
        num_docs: int = 4,
        num_hops: int = 4,
        search_endpoint: str = "http://localhost:4242/search",
        **kwargs
    ):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.update_info = dspy.ChainOfThought(UpdateInfo)
        self.doc_retriever = RemoteSearchFunction(search_endpoint)

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

    def link_entities(self, statement: str) -> list[str]:
        """
        Extracts entities from the statement and returns their wikipedia abstracts if appropriate pages exist.
        """
        entities = asyncio.run(detect_entities(statement, lang="cs", do_lemmatize=True))
        abstracts = []
        if entities:
            abstracts = asyncio.run(get_abstracts(entities, lang="cs"))

        return abstracts

    def forward(self, statement: Statement) -> dspy.Prediction:
        info = ""
        queries = []
        evidence = []

        for _ in range(self.num_hops):
            query = (
                self.generate_query(
                    statement=statement.statement,
                    author=statement.author,
                    date=statement.date,
                    info=info,
                    already_used_queries=queries,
                )
            ).nl_query

            queries.append(query)

            with mlflow.start_span(name="search_documents") as span:
                span.set_inputs(
                    {
                        "query": query,
                        "k": self.num_docs,
                        "key": statement.id,
                    }
                )
                new_segments = self.doc_retriever.search(
                    query=query, k=self.num_docs, key=statement.id
                )
                span.set_outputs(
                    {"segments": [segment.text for segment in new_segments]}
                )

            # Update retrieved segments and texts
            evidence.extend(new_segments)

            # Remove duplicates
            retrieved_segments = list(
                {segment.text: segment for segment in evidence}.values()
            )

            # Update info based on the new evidence
            new_evidence = self._extract_text(retrieved_segments)
            info_update = self.update_info(
                statement=statement.statement,
                author=statement.author,
                date=statement.date,
                current_info=info,
                new_evidence=new_evidence,
            )

            info = info_update.updated_info

            # if info_update.terminate:
            #     break

        return dspy.Prediction(
            evidence=self._format_segments(evidence),
            used_queries=queries,
        )
