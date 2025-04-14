import json
import re

from dataset_manager.models import Statement
from utils.llm_apis import LanguageModelAPI

from ..base import FactCheckingState, Predictor

class BasicPredictor(Predictor):
    def __init__(
        self,
        model: LanguageModelAPI,
        system_prompt: str,
        generation_prompt: str,
        prompt_template: str,
        allowed_labels: list[str],
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.allowed_labels = allowed_labels

        self.model.set_system_prompt(system_prompt)
        self.model.set_generation_prompt(generation_prompt)

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
