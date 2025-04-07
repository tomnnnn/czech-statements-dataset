import logging
from sklearn.model_selection import train_test_split
from dataset_manager import Dataset
from .config import Config
from dataset_manager.models import Statement
from .fact_checker import FactChecker
import sklearn
import numpy as np
import random

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

        statements = self.dataset.get_statements(self.cfg.allowed_labels)
        labels = [statement.label for statement in statements]

        if self.cfg.test_portion < 1:
            _, sample = train_test_split(statements, test_size=self.cfg.test_portion, random_state=self.seed, stratify=labels)
        else:
            sample = statements

        parallelized_sample = self._cut_for_parallelization(sample)

        return parallelized_sample


    def _evaluate(self, predicted, reference):
        # NOTE: currently, the labels are ordinary strings. Using enums to unify labels is a good idea.
        predicted_labels = [statement['label'].lower() for statement in predicted]
        reference_labels = [statement.label.lower() for statement in reference]

        # calculate metrics
        return sklearn.metrics.classification_report(reference_labels, predicted_labels, output_dict=True, labels=self.cfg.allowed_labels, zero_division=np.nan)

    def run(self) -> tuple:
        """
        Run the evaluation process.

        Returns:
        Tuple: Predictions and evaluation metrics.
        """
        statements = self._sample_dataset()
        print("Sampled statements:", statements)
        predictions = self.fact_checker.run(statements)
        metrics = self._evaluate(predictions, statements)

        return predictions, metrics
