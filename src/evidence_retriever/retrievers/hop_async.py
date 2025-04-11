from typing import TypedDict
from src.dataset_manager.models import Segment
from src.evidence_retriever.search_functions.search_function import SearchFunction
from src.evidence_retriever.retriever import AsyncRetriever

class State(TypedDict):
    statement: str
    context: list[Segment]
    goals: list[str]

class AsyncHopRetriever(AsyncRetriever):
    def __init__(self, search_function: SearchFunction, num_docs=3, num_hops=3, **kwargs):
        self.search_function = search_function
        self.num_docs = num_docs
        self.num_hops = num_hops
        self.query_template = """..."""
        self.new_goals_template = """..."""

    def _hop(self, statement: str, state: State) -> State:
        # Implement the hop logic here
        # This is a placeholder implementation
        pass


    async def __call__(self, statement: str) -> list[Segment]:
        # initialize context and goals

        state = State(
            statement=statement,
            context=[],
            goals=[],
        )

        for i in range(self.num_hops):
            # perform hop
            new_state = self._hop(statement, state)

            # update state
            state["context"].extend(new_state["context"])

        return state["context"]
