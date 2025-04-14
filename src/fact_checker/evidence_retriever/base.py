import dspy
import logging
from typing import Literal

from src.dataset_manager.models import Segment

from .search_functions.search_function import SearchFunction

logger = logging.getLogger(__name__)

class Retriever(dspy.Module):
    def __init__(self, num_docs=10, **kwargs):
        raise NotImplementedError("The Retriever class is an abstract base class. Please implement a subclass.")

    def forward(self, statement: str, search_func: SearchFunction) -> dspy.Prediction:
        raise NotImplementedError("The forward method is not implemented. Please implement it in the subclass.")
