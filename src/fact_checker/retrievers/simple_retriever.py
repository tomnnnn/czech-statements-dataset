import dspy
from ..search_functions import SearchFunction

class SimpleRetriever(dspy.Module):
    def __init__(self, num_docs=3):
        self.num_docs = num_docs

    async def forward(self, statement: str, search_func: SearchFunction) -> dspy.Prediction:
        query = statement
        documents = search_func.search(query, self.num_docs)

        return dspy.Prediction(
            segments = documents,
            used_queries = [query]
        )

