import dspy
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class HopRetriever(dspy.Module):
    def __init__(self, search_func: Callable[[str, int], dict|list], num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought("výrok, co_zjistit, jazyk -> query")
        self.append_notes = dspy.ChainOfThought( "výrok, co_zjistit, nasbírané_texty -> co_dalšího_zjistit: list[str]")
        self.search = search_func

    def forward(self, statement: str):
        notes = [statement]
        retrieved_segments = [] # all segments retrieved
        retrieved_texts = [] # retrieved segments without metadata
        query_lang = "cs"

        for _ in range(self.num_hops):
            # Generate query and search for new segments
            start = time.time()
            query = self.generate_query(výrok=statement, co_zjistit=notes, jazyk=query_lang).query

            if (time.time() - start) > 1:
                logger.info(f"Generating query took {time.time() - start} seconds")

            start = time.time()
            new_segments = self.search(query, self.num_docs)
            if (time.time() - start) > 1:
                logger.info(f"Searching took {time.time() - start} seconds")

            # Update retrieved segments and texts
            retrieved_segments.extend(new_segments)
            retrieved_texts.extend([segment["text"] for segment in new_segments])

            # Update notes for the next iteration
            start = time.time()
            prediction = self.append_notes(výrok=statement, co_zjistit=notes, nasbírané_texty=retrieved_texts)

            if (time.time() - start) > 1:
                logger.info(f"Appending notes took {time.time() - start} seconds")
            notes.extend(prediction.co_dalšího_zjistit)

        return dspy.Prediction(notes=notes, segments=list({frozenset(d.items()): d for d in retrieved_segments}.values()))
