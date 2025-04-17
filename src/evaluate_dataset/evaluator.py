import logging
import subprocess
import requests
import time
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from dataset_manager import Dataset
from fact_checker.evidence_retriever.search_functions import BGE_M3, BM25
from src.fact_checker.evidence_retriever.search_functions.mock import MockSearchFunction
from src.fact_checker.fc_state import FactCheckingResult
from .config import Config
import asyncio
from dataset_manager.models import Statement
from fact_checker import FactChecker
import sklearn
import numpy as np
import random
from tqdm.asyncio import tqdm_asyncio
import os

logger = logging.getLogger(__name__)


class FactCheckingEvaluator:
    def __init__(self, dataset: Dataset, fact_checker: FactChecker, cfg: Config, seed: int = 42):
        random.seed(seed)

        self.dataset = dataset
        self.fact_checker = fact_checker
        self.cfg = cfg
        self.seed = seed
        self.parallelized = cfg.index > 0

    def _cut_for_parallelization(self, statements) -> list[Statement]:
        """
        Determines range of statements to evaluate based on parallelization index and max processes. If index is 0, all statements are evaluated.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        List: Subset of statements to evaluate.
        """

        lower_index = max(0, (self.cfg.index - 1) * len(statements) // self.cfg.max)
        upper_index = len(statements) - 1 if self.cfg.index == 0 else self.cfg.index * len(statements) // self.cfg.max - 1
        logger.info(f"Parallelization index: {self.cfg.index}, lower index: {lower_index}, upper index: {upper_index}")

        return statements[lower_index:upper_index + 1]

    def _sample_dataset(self) -> list[Statement]:
        """
        Sample a subset of the dataset for evaluation.

        Returns:
        List: Sampled statements from the dataset.
        """

        statements = self.dataset.get_statements(self.cfg.allowed_labels, self.cfg.min_evidence_count)
        labels = [statement.label for statement in statements]

        if self.cfg.test_portion < 1:
            _, sample = train_test_split(statements, test_size=self.cfg.test_portion, random_state=self.seed, stratify=labels)
        else:
            sample = statements

        parallelized_sample = self._cut_for_parallelization(sample)

        return parallelized_sample


    async def _create_search_functions(self, statements: list[Statement]):
        segments = self.dataset.get_segments_by_statements([s.id for s in statements])

        # Filter statements with segments
        statements = [s for s in statements if s.id in segments]

        search_functions = {}

        semaphore = asyncio.Semaphore(20)

        # Load model
        model = SentenceTransformer("BAAI/BGE-M3")

        async def build_search_fn(statement: Statement):
            async with semaphore:
                segment_list = segments[statement.id]
                index_path = f"indexes/{statement.id}.faiss"
                load_index = os.path.exists(index_path)

                search_fn = BGE_M3(
                    segment_list,
                    save_index=not load_index,
                    load_index=load_index,
                    index_path=index_path,
                    model=model
                )
                await search_fn.index_async()
                return statement.id, search_fn

                # search_fn = BM25(segment_list)
                # return statement.id, search_fn

        results = await tqdm_asyncio.gather(*(build_search_fn(s) for s in statements), desc="Building evidence document indices", unit="indices")

        for statement_id, search_fn in results:
            search_functions[statement_id] = search_fn

        return search_functions


    def _evaluate(self, predicted: list[FactCheckingResult], reference):
        # NOTE: currently, the labels are ordinary strings. Using enums to unify labels is a good idea.
        predicted_labels = [statement['label'].lower() for statement in predicted]
        reference_labels = [statement.label.lower() for statement in reference if statement.id in [p['statement_id'] for p in predicted]]

        # calculate metrics
        return sklearn.metrics.classification_report(reference_labels, predicted_labels, output_dict=True, labels=self.cfg.allowed_labels, zero_division=np.nan)


    def _wait_for_server(self):
        # wait for server to be up by checking api_base_url/health
        # TODO: move to seperate class
        subprocess.Popen(["vllm", "serve", "--enable-chunked-prefill","true","--gpu-memory-utilization", "0.85", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"])

        print(f"Waiting for LLM server to be up...")
        while True:
            try:
                response = requests.get("http://0.0.0.0:8000/health")

                if response.status_code == 200:
                    print("Server is up and running.")
                    break
                else:
                    print(f"Server is not ready yet, status code: {response.status_code}")
                    time.sleep(10)
            except Exception:
                time.sleep(10)

    async def run(self) -> tuple:
        """
        Run the evaluation process.

        Returns:
        Tuple: Predictions and evaluation metrics.
        """
        statements = self._sample_dataset()
        search_functions = await self._create_search_functions(statements)

        self._wait_for_server()

        statements = [s for s in statements if s.id in search_functions.keys()]

        coroutines = [self.fact_checker.run([statement], search_functions[statement.id]) for statement in statements]

        predictions = await tqdm_asyncio.gather(*coroutines, desc="Evaluating statements", unit="statement")
        predictions = [p for p in predictions if p]

        # flatten the predictions
        predictions = [prediction for sublist in predictions for prediction in sublist]
        metrics = self._evaluate(predictions, statements)

        return predictions, metrics
