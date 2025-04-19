import dspy
import json
import logging
from typing import Literal
from dataset_manager.models import Statement
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


    def forward(self, statement: Statement) -> dspy.Prediction:
        # get evidence
        logger.info("Retrieving evidence...")
        evidence = self.retriever(statement).evidence

        # classify
        logger.info("Classifying statement...")
        label = self.classify(
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
            evidence=json.dumps(evidence, ensure_ascii=False),
        ).label

        # create and return the prediction
        return dspy.Prediction(
            metadata={"retriever": "hop_retriever"},
            statement=statement.statement,
            evidence=evidence,
            label=label,
        )
