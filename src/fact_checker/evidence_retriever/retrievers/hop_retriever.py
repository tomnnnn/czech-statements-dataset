import dspy
import sys
import logging
import time
from ..search_functions import SearchFunction
from ..base import Retriever

logger = logging.getLogger(__name__)

class HopRetriever(Retriever):
    def __init__(self, num_docs=4, num_hops=4, **kwargs):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("výrok, co_zjistit, jazyk -> query")
        self.append_notes = dspy.ChainOfThought( "výrok, co_zjistit, nasbírané_texty -> co_dalšího_zjistit: list[str]")

    def forward(self, statement: str, search_func: SearchFunction) -> dspy.Prediction:
        notes = [statement]
        retrieved_segments = [] # all segments retrieved
        retrieved_texts = [] # retrieved segments without metadata
        queries = []
        query_lang = "cs"
        print(f"Statement: {statement}", file=sys.stderr)

        for i in range(self.num_hops):
            print(f"Hop {i+1}/{self.num_hops}", file=sys.stderr)

            # Generate query and search for new segments
            start = time.time()
            query = self.generate_query(výrok=statement, co_zjistit=notes, jazyk=query_lang).query
            queries.append(query)

            print(f"Generating query took {time.time() - start} seconds", file=sys.stderr)

            start = time.time()
            new_segments = search_func.search(str(query), self.num_docs)

            print(f"Searching took {time.time() - start} seconds", file=sys.stderr)

            # Update retrieved segments and texts
            retrieved_segments.extend(new_segments)
            retrieved_texts.extend([segment.text for segment in new_segments])

            # Update notes for the next iteration
            start = time.time()
            prediction = self.append_notes(výrok=statement, co_zjistit=notes, nasbírané_texty=retrieved_texts)

            print(f"Appending notes took {time.time() - start} seconds", file=sys.stderr)
            notes.extend(prediction.co_dalšího_zjistit)

        return dspy.Prediction(notes=notes, segments=retrieved_segments, metadata={"retriever": "hop_retriever"}, used_queries=queries)
