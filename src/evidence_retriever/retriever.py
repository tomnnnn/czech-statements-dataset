import dspy
import logging
from typing import Literal

from src.dataset_manager.models import Segment

from .search_functions.search_function import SearchFunction

logger = logging.getLogger(__name__)

class Retriever(dspy.Module):
    def __init__(self, search_function: SearchFunction, num_docs=10, **kwargs):
        raise NotImplementedError("The Retriever class is an abstract base class. Please implement a subclass.")

    def forward(self, statement: str) -> dspy.Prediction:
        raise NotImplementedError("The forward method is not implemented. Please implement it in the subclass.")


class AsyncRetriever:
    def __init__(self, search_function: SearchFunction, num_docs=3, **kwargs):
        raise NotImplementedError("The AsyncRetriever class is an abstract base class. Please implement a subclass.")

    async def __call__(self, statement: str) -> list[Segment]:
        raise NotImplementedError("The __call__ method is not implemented. Please implement it in the subclass.")
