import json
import logging
from collections import defaultdict
import re
import yaml

from dataset_manager.models import Statement
from dataset_manager.orm import rows2dict
from evidence_retriever import Retriever, SimpleRetriever
from src.evidence_retriever.retrievers.hop_retriever import HopRetriever
from .config import Config
from .llm_api import LanguageModelAPI
import dspy
from tqdm.asyncio import tqdm_asyncio
from evidence_retriever.search_functions import SearchFunction
logger = logging.getLogger(__name__)


class FactChecker:
    def __init__(
        self,
        model: LanguageModelAPI,
        evidence_retriever: Retriever,
        cfg: Config,
        **kwargs
    ):
        self.model = model
        self.cfg = cfg
        self.show_progress= kwargs.get("show_progress", True)

        self.retriever = dspy.asyncify(evidence_retriever)

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


    async def _retrieve_segments(self, statement: Statement, search_function: SearchFunction) -> tuple[int, list[dict], dict]:
        """
        Retrieve segments for a given statement using the retriever.

        Args:
        statement (Statement): The statement to evaluate.

        Returns:
        Tuple: A tuple containing the statement ID and a list of segments.
        """

        logger.info(f"Retrieving segments for statement {statement.id}")

        statement_str = f"{statement.statement} - {statement.author}, {statement.date}"

        # if isinstance(self.retriever, SimpleRetriever):
        #     retrieved_evidence = (await self.retriever(statement=statement_str, search_func=search_function))
        # else:
        #     retrieved_evidence = self.retriever(statement=statement_str, search_func=search_function)

        retrieved_evidence = await self.retriever(statement=statement_str, search_func=search_function)

        segments = retrieved_evidence.segments
        enriched_segments = [
            {
                "title": segment.article.title[:3000],
                "text": segment.text[:3000],
                "url": segment.article.source[:3000],
            }
            for segment in segments
        ]

        return statement.id, enriched_segments, retrieved_evidence.used_queries


    async def _gather_evidence(self, statements: list[Statement], search_function: SearchFunction) -> tuple[dict[int, list[dict]], dict]:
        """
        Gather evidence for a given statement using the retriever.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        Dict: A dictionary mapping statement IDs to lists of segments.
        """

        logger.info("Gathering evidence")

        evidence = {}
        used_queries = {}

        # for statement in statements:
        #     id, segments, queries = await self._retrieve_segments(statement, search_function)
        #     evidence[id] = segments
        #     used_queries[id] = queries

        coroutines = [
            self._retrieve_segments(statement, search_function)
            for statement in statements
        ]

        results = await tqdm_asyncio.gather(*coroutines, desc="Gathering evidence", unit="statement", disable=not self.show_progress)

        for statement_id, segments, queries in results:
            evidence[statement_id] = segments
            used_queries[statement_id] = queries

        return evidence, used_queries

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

    async def fact_check(self, statement: Statement, evidence: list[dict]):
        """
        Fact-check a single statement using the model.

        Args:
        statement (Statement): The statement to evaluate.
        evidence (List): List of segments related to the statement.

        Returns:
        Dict: The result of the evaluation.
        """

        prompt = self.prompt_template.format(
            statement=statement.statement,
            author=statement.author,
            date=statement.date,
            evidence=json.dumps(evidence, ensure_ascii=False),
        )

        response = await self.model(prompt)
        label = self._extract_label(response[0])

        return {
            "id": statement.id,
            "statement": statement.statement,
            "evidence": evidence,
            "prompt": prompt,
            "response": response,
            "label": label,
         }

    async def run(self, statements: list[Statement], search_function: SearchFunction, show_progress=False):
        """
        Run the fact-checking process on a list of statements.

        Args:
        statements (List): List of statements to evaluate.

        Returns:
        List: Results of the evaluation.
        """

        try:
            evidence, used_queries = await self._gather_evidence(statements, search_function)
            prompts = self._build_prompts(statements, evidence)

            logger.info("Running model inference")
            responses = await self.model(prompts)
            predicted_labels = [self._extract_label(prediction) for prediction in responses]

            return [
                {
                    "id": statement.id,
                    "statement": statement.statement,
                    "queries": used_queries[statement.id],
                    "evidence": evidence[statement.id],
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                 }
                for statement, label, response, prompt in zip(statements, predicted_labels, responses, prompts)
            ]
        except Exception as e:
            logger.error(f"Error during fact-checking: {e}\nStatement: {statements}")
            return None
