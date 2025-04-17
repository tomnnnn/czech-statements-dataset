from typing import Literal

import dspy
import json
from src.dataset_manager.dataset import Dataset
from src.dataset_manager.models import Statement
from src.fact_checker.evidence_retriever.retrievers import HopRetriever
from src.fact_checker.evidence_retriever.retrievers.hop_retriever_alt import HopRetrieverAlt
from src.fact_checker.evidence_retriever.search_functions.search_function import SearchFunction

class Veracity(dspy.Signature):
    statement: str = dspy.InputField()
    author: str = dspy.InputField()
    date: str = dspy.InputField()
    evidence: list[dict] = dspy.InputField()

    label: Literal["pravda", "nepravda", "neověřitelné"] = dspy.OutputField()


class FactChecker(dspy.Module):
    def __init__(
        self,
        dataset: Dataset,
        retrieval_hops=4,
        per_hop_documents=4,
        optimized_retriever_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.retriever = HopRetriever(
            num_hops=retrieval_hops, 
            num_docs=per_hop_documents
        )

        self.classify = dspy.ChainOfThought(Veracity)
        self.dataset = dataset

        # load optimized retriever if path is provided
        if optimized_retriever_path:
            self.retriever.load(optimized_retriever_path)

    def forward(self, statement: Statement, search_func: SearchFunction) -> dspy.Prediction:
        print("Running fact check for statement:", statement.statement)
        # get evidence
        print("Retrieving evidence...")
        evidence = self.retriever(statement, search_func).evidence

        # classify
        print("Classifying statement...")
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
