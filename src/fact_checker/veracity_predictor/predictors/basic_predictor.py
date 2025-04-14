import json
import re

from utils.llm_apis import LanguageModelAPI

from ..base import ContextualizedStatement, Predictor


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

    async def predict(self, statements: list[ContextualizedStatement]) -> list[dict]:
        prompts = [
            self.prompt_template.format(
                statement=statement["statement"],
                evidence=json.dumps(statement["evidence"], ensure_ascii=False),
            )
            for statement in statements
        ]

        # Generate predictions
        responses = await self.model(prompts)

        # Parse predictions
        labels = [self._parse_prediction(response) for response in responses]

        predictions = [
            {
                "id": statement["id"],
                "statement": statement["statement"],
                "evidence": statement["evidence"],
                "label": label,
            }
            for statement, label in zip(statements, labels)
        ]

        return predictions
