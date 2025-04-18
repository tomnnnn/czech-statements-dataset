import json
import re
import yaml

from dataset_manager.models import Statement
from utils.llm_apis import LanguageModelAPI

from ..base import Predictor

class BasicPredictor(Predictor):
    def __init__(self, model: LanguageModelAPI, prompt_config: str | dict):
        """
        Initialize the BasicPredictor with a language model and prompt configuration.

        Args:
        model (LanguageModelAPI): The language model to use for predictions.
        prompt_config (str | dict): Path to a YAML file or a dictionary containing the prompt configuration.

        Raises:
        ValueError: If the prompt configuration is invalid or missing required keys.
        """
        self.model = model

        if isinstance(prompt_config, str):
            if not prompt_config.endswith(".yaml"):
                raise ValueError("Expected a YAML file for prompt configuration.")
            with open(prompt_config, "r") as f:
                config = yaml.safe_load(f)
        elif isinstance(prompt_config, dict):
            config = prompt_config

        self._validate_prompt_config(config)

        self.model.set_system_prompt(config["system_prompt"])
        self.model.set_generation_prompt(config.get("generation_prompt", ""))
        self.allowed_labels = config["allowed_labels"]
        self.prompt_template = config["prompt_template"]

    def _validate_prompt_config(self, config: dict):
        required_keys = ["prompt_template", "allowed_labels", "system_prompt"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")


    def _parse_prediction(self, prediction: str) -> str:
        """
        Extract the label from the model's prediction.

        Args:
        prediction (str): The model's prediction.

        Returns:
        str: The extracted label.
        """
        if prediction:
            labels_regex = "|".join(self.allowed_labels)
            labels = re.findall(labels_regex, prediction.lower())
            return labels[-1] if labels else "nolabel"

        return "nolabel"

    async def predict(self, statement: Statement, evidence: list[dict]) -> tuple[str, str]:
        prompt = self.prompt_template.format(
                statement=str(statement),
                evidence=json.dumps(evidence, ensure_ascii=False),
            )

        # Generate predictions
        response = (await self.model(prompt))[0]

        # Parse predictions
        label = self._parse_prediction(response)

        return label, response

    def predict_sync(self, statement: Statement, evidence: list[dict]) -> tuple[str, str]:
        prompt = self.prompt_template.format(
                statement=str(statement),
                evidence=json.dumps(evidence, ensure_ascii=False),
            )

        # Generate predictions
        response = self.model.generate_sync(prompt)[0]

        # Parse predictions
        label = self._parse_prediction(response)

        return label, response
