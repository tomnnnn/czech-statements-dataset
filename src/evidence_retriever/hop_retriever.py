import dspy
from typing import Callable

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
            query = self.generate_query(výrok=statement, co_zjistit=notes, jazyk=query_lang).query
            new_segments = self.search(query, self.num_docs)

            # Update retrieved segments and texts
            retrieved_segments.extend(new_segments)
            retrieved_texts.extend([segment["text"] for segment in new_segments])

            # Update notes for the next iteration
            prediction = self.append_notes(výrok=statement, co_zjistit=notes, nasbírané_texty=retrieved_texts)
            notes.extend(prediction.co_dalšího_zjistit)

        return dspy.Prediction(notes=notes, segments=list({frozenset(d.items()): d for d in retrieved_segments}.values()))
