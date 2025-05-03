import dspy
import json
import logging
from typing import Literal
from dataset_manager.models import Statement
from .retrievers import HopRetriever
import mlflow

logger = logging.getLogger(__name__)

class VeracityTernary(dspy.Signature):
    """
    Based on given evidence, determine whether the statement is supported or refuted.
    When dealing with exact numbers, use 10 % tolerance except for when the statement
    itself explicitly emphasizes the exactness of the number. Treat phrases like "I think"
    or "I've heard" as signs of uncertainty, but don't verify the speaker's intent. 
    Assume such statements are approximate factual claims unless clearly speculative.
    """
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()

    label: Literal["pravda", "nepravda", "neověřitelné"] = dspy.OutputField()

class VeracityBinary(dspy.Signature):
    """
    Based on given evidence, determine whether the statement is supported or refuted.
    When dealing with exact numbers, use 10 % tolerance except for when the statement
    itself explicitly emphasizes the exactness of the number. Treat phrases like "I think"
    or "I've heard" as signs of uncertainty, but don't verify the speaker's intent. 
    Assume such statements are approximate factual claims unless clearly speculative.
    """
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
        search_endpoint="http://localhost:4242/search",
        mode: Literal["ternary", "binary"] = "ternary",
        **kwargs,
    ):
        self.retriever = HopRetriever(
            num_hops=retrieval_hops, 
            num_docs=per_hop_documents,
            search_endpoint=search_endpoint,
        )

        veracity = VeracityTernary if mode == "ternary" else VeracityBinary

        self.classify = dspy.ChainOfThought(veracity)

        mlflow.log_params({
            "fact_checker": "iterative_hop",
            "retrieval_hops": retrieval_hops,
            "per_hop_documents": per_hop_documents,
        })


    def forward(self, statement: Statement) -> dspy.Prediction:
        # get evidence
        logger.info("Retrieving evidence...")
        retriever_result = self.retriever(statement)
        evidence = retriever_result.evidence
        used_queries = retriever_result.used_queries

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
            used_queries=used_queries
        )
