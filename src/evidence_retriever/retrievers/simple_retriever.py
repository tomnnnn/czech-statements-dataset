from src.dataset_manager.models import Segment
import dspy
from src.evidence_retriever.search_functions.search_function import SearchFunction
from src.evidence_retriever.retriever import AsyncRetriever


class SimpleRetriever():
    def __init__(self, search_function: SearchFunction, num_docs=3):
        self.search_function = search_function
        self.num_docs = num_docs

    async def __call__(self, statement: str) -> dspy.Prediction:
        query = statement
        documents = self.search_function.search(query, self.num_docs)

        return dspy.Prediction(
            segments = documents,
        )

