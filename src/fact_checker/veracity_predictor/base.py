from dataset_manager.models import Statement
from utils.llm_apis import LanguageModelAPI
from ..fc_state import FactCheckingResult

class Predictor():
    """
    Prompts LLM with statement and evidence documents to generate a veracity score or label.
    """
    def __init__(
        self,
        model: LanguageModelAPI,
        system_prompt: str,
        generation_prompt: str,
        prompt_template: str,
        allowed_labels: list[str],
    ):
        raise NotImplementedError("Subclasses should implement this method.")


    async def predict(self, statement: Statement, evidence: list[dict]) -> tuple[str, str]:
        """
        Based on statement and evidence docments, predicts the statement's veracity label.
        """
        raise NotImplementedError("Subclasses should implement this method.")
