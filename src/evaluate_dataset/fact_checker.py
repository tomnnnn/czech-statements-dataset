import json
import logging
from collections import defaultdict
import re
import yaml

from dataset_manager.models import Statement
from dataset_manager.orm import rows2dict
from evidence_retriever import Retriever, SimpleRetriever
from .config import Config
from .llm_api import LanguageModelAPI
from tqdm.asyncio import tqdm_asyncio

from threading import Lock

logger = logging.getLogger(__name__)


class FactChecker:
    def __init__(
        self,
        model: LanguageModelAPI,
        evidence_retriever: Retriever|SimpleRetriever,
        cfg: Config,
    ):
        self.model = model
        self.cfg = cfg
        self.retriever = evidence_retriever
        self.lock = Lock()

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


    async def _retrieve_segments(self, statement: Statement) -> tuple[int, list[dict]]:
        """
        Retrieve segments for a given statement using the retriever.

        Args:
        statement (Statement): The statement to evaluate.

        Returns:
        Tuple: A tuple containing the statement ID and a list of segments.
        """

        logger.info(f"Retrieving segments for statement {statement.id}")

        statement_str = f"{statement.statement} - {statement.author}, {statement.date}"

        if isinstance(self.retriever, SimpleRetriever):
            segments = (await self.retriever(statement=statement_str)).segments
        else:
            segments = self.retriever(statement=statement_str).segments

        enriched_segments = [
            {
                "title": segment.article.title[:3000],
                "text": segment.text[:3000],
                "url": segment.article.source[:3000],
            }
            for segment in segments
        ]

        return statement.id, enriched_segments


    async def _gather_evidence(self, statements: list[Statement]) -> dict[int, list[dict]]:
        """
        Gather evidence for a given statement using the retriever.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        Dict: A dictionary mapping statement IDs to lists of segments.
        """

        logger.info("Gathering evidence")

        evidence = defaultdict(list)

        for statement in statements:
            id, segments = await self._retrieve_segments(statement)
            evidence[id] = segments

        # coroutines = [
        #     self._retrieve_segments(statement)
        #     for statement in statements
        # ]
        #
        # results = await tqdm_asyncio.gather(*coroutines, desc="Gathering evidence", unit="statement")
        #
        # for statement_id, segments in results:
        #     evidence[statement_id] = [
        #         f"{segment['title']}: {segment['text']} ({segment['url']})"
        #         for segment in segments
        #     ]

        return evidence

    def _build_prompts(
        self, statements: list[Statement], evidence: dict[int, list[dict]]
    ) -> list[str]:
        """
        Build prompts for the model based on statements and their evidence.

        Args:
        statements (List): List of statements to evaluate.
        evidence (Dict): A dictionary mapping statement IDs to lists of segments.

        Returns:
        List: List of prompts for language model inference.
        """

        logger.info("Building prompts")

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

    async def run(self, statements: list[Statement]):
        """
        Run the fact-checking process on a list of statements.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        List: Results of the evaluation.
        """

        evidence = await self._gather_evidence(statements)
        prompts = self._build_prompts(statements, evidence)

        logger.info("Running model inference")
        responses = await self.model(prompts)
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
