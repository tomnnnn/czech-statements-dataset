from src.dataset_manager.models import Segment
import dspy
from ..search_functions import SearchFunction
from ..base import Retriever


class SimpleRetriever(Retriever):
    def __init__(self, num_docs=3):
        self.num_docs = num_docs

    def forward(self, statement: str, search_func: SearchFunction) -> dspy.Prediction:
        query = statement
        documents = search_func.search(query, self.num_docs)

        return dspy.Prediction(
            segments = documents,
            used_queries = [query]
        )

