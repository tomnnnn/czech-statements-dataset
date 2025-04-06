import dspy
import logging
from typing import Literal

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Retriever(dspy.Module):
    def __init__(self, search_algorithm: str, corpus, num_docs=10, **kwargs):
        raise NotImplementedError("The Retriever class is an abstract base class. Please implement a subclass.")

    def forward(self, statement: str) -> dspy.Prediction:
        raise NotImplementedError("The forward method is not implemented. Please implement it in the subclass.")
