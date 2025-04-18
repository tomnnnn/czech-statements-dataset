from typing import Literal

import dspy
import json
import logging
from src.dataset_manager.dataset import Dataset
from src.dataset_manager.models import Statement
from .retrievers import HopRetriever
from .search_functions.base import SearchFunction

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


class FactChecker(dspy.Module):
    def __init__(
        self,
        retrieval_hops=4,
        per_hop_documents=4,
        mode: Literal["ternary", "binary"] = "ternary",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.retriever = HopRetriever(
            num_hops=retrieval_hops, 
            num_docs=per_hop_documents
        )

        veracity = VeracityTernary if mode == "ternary" else VeracityBinary

        self.classify = dspy.ChainOfThought(veracity)


    async def forward(self, settings, statement: Statement, search_func: SearchFunction) -> dspy.Prediction:
        # get evidence
        logger.info("Retrieving evidence...")
        evidence = (await self.retriever(settings, statement, search_func)).evidence

        # classify
        logger.info("Classifying statement...")
        label = (await self.classify(
            settings=settings,
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
            evidence=json.dumps(evidence, ensure_ascii=False),
        )).label

        # create and return the prediction
        return dspy.Prediction(
            metadata={"retriever": "hop_retriever"},
            statement=statement.statement,
            evidence=evidence,
            label=label,
        )
