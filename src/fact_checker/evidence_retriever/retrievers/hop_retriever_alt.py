
import logging
import os
import dspy
from sentence_transformers import SentenceTransformer
from src.dataset_manager.models import Segment, Statement
from src.fact_checker.evidence_retriever.search_functions.bge_m3 import BGE_M3
from ..search_functions import SearchFunction
from concurrent.futures import ThreadPoolExecutor
logger = logging.getLogger(__name__)


class GenerateQuery(dspy.Signature):
    výrok: str = dspy.InputField()
    autor: str = dspy.InputField()
    datum: str = dspy.InputField()
    co_zjistit: list[str] = dspy.InputField()

    query: str = dspy.OutputField()


class AppendNotes(dspy.Signature):
    výrok: str = dspy.InputField()
    autor: str = dspy.InputField()
    datum: str = dspy.InputField()
    co_zjistit: list[str] = dspy.InputField()
    nasbírané_texty: list[str] = dspy.InputField()

    co_dalšího_zjistit: list[str] = dspy.OutputField()


class HopRetrieverAlt(dspy.Module):
    """
    Alternative HopRetriever that creates the search index before searching.
    """
    def __init__(self, dataset, num_docs=4, num_hops=4, **kwargs):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.append_notes = dspy.ChainOfThought(AppendNotes)
        self.dataset = dataset
        self.encoder = SentenceTransformer("BAAI/BGE-M3")

    def _format_segments(self, segments: list[Segment]) -> list[dict]:
        return [
            {
                "title": segment.article.title[:3000],
                "text": segment.text[:3000],
                "url": segment.article.source[:3000],
            }
            for segment in segments
        ]

    def _extract_text(self, segments: list[Segment]) -> list[str]:
        return [segment.text for segment in segments]

    def create_search_function(self, statement: Statement) -> SearchFunction:
        # TODO: decouple from BGE_M3
        segments = self.dataset.get_segments_by_statements([statement.id])[statement.id]
        index_path = os.path.join("indexes", f"{statement.id}.faiss")
        load_index = os.path.exists(index_path)

        print("Loading index:", load_index)
        print("Creating index:", index_path)
        search_func = BGE_M3(
            segments,
            save_index=not load_index,
            load_index=load_index,
            index_path=index_path,
            model=self.encoder,
        )

        search_func.create_index()

        print("Index created:", index_path)
        return search_func


    def forward(self, statement: Statement) -> dspy.Prediction:
        notes = [statement]
        retrieved_segments = []
        queries = []

        print("Fetching segments for statement:", statement.id)
        search_func = self.create_search_function(statement)

        for i in range(self.num_hops):
            # Generate query and search for new segments
            print("Hop number:", i)
            query = self.generate_query(
                výrok=statement.statement,
                autor=statement.author,
                datum=statement.date,
                co_zjistit=notes,
            ).query
            queries.append(query)

            new_segments = search_func.search(str(query), self.num_docs)

            # Update retrieved segments and texts
            retrieved_segments.extend(new_segments)

            # Update notes for the next iteration
            prediction = self.append_notes(
                výrok=statement.statement,
                autor=statement.author,
                datum=statement.date,
                co_zjistit=notes,
                nasbírané_texty=self._extract_text(retrieved_segments),
            )

            notes.extend(prediction.co_dalšího_zjistit)

        evidence = self._format_segments(retrieved_segments)

        return dspy.Prediction(
            segments=retrieved_segments,
            evidence=evidence,
            used_queries=queries,
            notes=notes,
        )
