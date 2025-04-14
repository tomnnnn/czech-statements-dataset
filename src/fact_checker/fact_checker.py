import logging

import dspy
from dataset_manager.models import Statement
from tqdm.asyncio import tqdm_asyncio
from .evidence_retriever import HopRetriever
from utils.llm_apis import LanguageModelAPI

from .evidence_retriever import Retriever

from .evidence_retriever.search_functions import SearchFunction
from .fc_state import FactCheckingState
from .veracity_predictor import Predictor

logger = logging.getLogger(__name__)


class FactChecker:
    def __init__(
        self,
        evidence_retriever: Retriever,
        veracity_predictor: Predictor,
        **kwargs,
    ):
        self.show_progress = kwargs.get("show_progress", True)

        self.predictor = veracity_predictor
        self.retriever = dspy.asyncify(evidence_retriever)

    async def _retrieve_segments(
        self, statement: Statement, search_function: SearchFunction
    ) -> tuple[int, list[dict], dict]:
        """
        Retrieve segments for a given statement using the retriever.

        Args:
        statement (Statement): The statement to evaluate.

        Returns:
        Tuple: A tuple containing the statement ID and a list of segments.
        """

        logger.info(f"Retrieving segments for statement {statement.id}")

        statement_str = f"{statement.statement} - {statement.author}, {statement.date}"


        retrieved_evidence = await self.retriever(
            statement_str, search_function
        )

        segments = retrieved_evidence.segments
        enriched_segments = [
            {
                "title": segment.article.title[:3000],
                "text": segment.text[:3000],
                "url": segment.article.source[:3000],
            }
            for segment in segments
        ]

        return statement.id, enriched_segments, retrieved_evidence.used_queries

    async def _gather_evidence(
        self, statements: list[Statement], search_function: SearchFunction
    ) -> tuple[dict[int, list[dict]], dict]:
        """
        Gather evidence for a given statement using the retriever.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        Dict: A dictionary mapping statement IDs to lists of segments.
        """

        evidence = {}
        used_queries = {}

        coroutines = [
            self._retrieve_segments(statement, search_function)
            for statement in statements
        ]

        results = await tqdm_asyncio.gather(
            *coroutines,
            desc="Gathering evidence",
            unit="statement",
            disable=not self.show_progress,
        )

        for statement_id, segments, queries in results:
            evidence[statement_id] = segments
            used_queries[statement_id] = queries

        return evidence, used_queries

    def _build_examples(self) -> list[str]:
        raise NotImplementedError


    async def run(
        self,
        statements: list[Statement],
        search_function: SearchFunction,
        show_progress=False,
    ):
        """
        Run the fact-checking process on a list of statements.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        List: Results of the evaluation.
        """

        try:
            evidence, used_queries = await self._gather_evidence(statements, search_function)

            predictions_coroutines = [
                self.predictor.predict(statement, evidence[statement.id])
                for statement in statements
            ]

            labels, responses = await tqdm_asyncio.gather(*predictions_coroutines, disable=not show_progress)

            states = []
            for statement,label,response in zip(statements, labels, responses):
                states.append(FactCheckingState(
                    statement=statement,
                    evidence=evidence[statement.id],
                    label=label,
                    metadata={
                        "used_queries": used_queries[statement.id],
                        "response": response
                    }
                ))

            return states

        except Exception as e:
            logger.error(f"Error during fact-checking: {e}\nStatement: {statements}")
            return None
