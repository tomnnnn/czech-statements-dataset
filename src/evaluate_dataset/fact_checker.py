import json
import logging
from collections import defaultdict
import re
import yaml
from tqdm import tqdm

from dataset_manager.models import Statement
from dataset_manager.orm import rows2dict
from evidence_retriever import Retriever
from .config import Config
from .llm_api import LanguageModelAPI

logger = logging.getLogger(__name__)


class FactChecker:
    def __init__(
        self,
        model: LanguageModelAPI,
        evidence_retriever: Retriever,
        cfg: Config,
    ):
        self.model = model
        self.cfg = cfg
        self.retriever = evidence_retriever

        # model instructions (system prompt)
        # load prompt config
        prompt_config_path = cfg.prompt_config

        with open(prompt_config_path, "r", encoding="utf-8") as f:
            prompt_config = yaml.safe_load(f)
            system_prompt = prompt_config.get("system_prompt", "")
            generation_prompt = prompt_config.get("generation_prompt", "")
            prompt_template = prompt_config.get("prompt_template", "")

        self.model.set_system_prompt(system_prompt)
        self.model.set_generation_prompt(generation_prompt)
        self.prompt_template = prompt_template

    def _gather_evidence(self, statements: list[Statement]) -> dict[int, list[str]]:
        """
        Gather evidence for a given statement using the retriever.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        Dict: A dictionary mapping statement IDs to lists of segments.
        """

        evidence = defaultdict(list)
        for statement in tqdm(statements, desc="Gathering evidence", unit="statement"):
            segments = self.retriever(statement.statement).segments
            segment_texts = [segment['text'] for segment in segments]

            evidence[statement.id] = segment_texts

        return evidence

    def _build_prompts(
        self, statements: list[Statement], evidence: dict[int, list[str]]
    ) -> list[str]:
        """
        Build prompts for the model based on statements and their evidence.

        Args:
        statements (List): List of statements to evaluate.
        evidence (Dict): A dictionary mapping statement IDs to lists of segments.

        Returns:
        List: List of prompts for language model inference.
        """

        prompts = []
        statement_dicts = rows2dict(statements)

        for statement in statement_dicts:
            evidence_json = json.dumps(evidence[statement["id"]], ensure_ascii=False)
            prompt = self.prompt_template.format(**statement, evidence=evidence_json)
            prompts.append(prompt)

        return prompts

    def _build_examples(self) -> list[str]:
        raise NotImplementedError

    def _extract_label(self, prediction: str) -> str:
        """
        Extract the label from the model's prediction.

        Args:
        prediction (str): The model's prediction.

        Returns:
        str: The extracted label.
        """
        if prediction:
            labels_regex = "|".join(self.cfg.allowed_labels)
            labels = re.findall(labels_regex, prediction.lower())
            return labels[-1] if labels else "nolabel"

        return "nolabel"

    def run(self, statements: list[Statement]):
        """
        Run the fact-checking process on a list of statements.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        List: Results of the evaluation.
        """

        evidence = self._gather_evidence(statements)
        prompts = self._build_prompts(statements, evidence)

        responses = self.model(prompts)
        predicted_labels = [self._extract_label(prediction) for prediction in responses]

        return [
            {
                "statement": statement.statement,
                "evidence": evidence[statement.id],
                "prompt": prompt,
                "response": response,
                "label": label,
             }
            for statement, label, response, prompt in zip(statements, predicted_labels, responses, prompts)
        ]
