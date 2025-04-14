from ..contextualized_statement import ContextualizedStatement
from utils.llm_apis import LanguageModelAPI

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


    async def predict(self, statements: list[ContextualizedStatement]) -> list[dict]:
        """
        Predict the veracity of a list of statements.

        Args:
            statements (list[ContextualizedStatement]): List of statements to evaluate.

        Returns:
            list[float]: List of veracity scores for each statement.
        """
        raise NotImplementedError("Subclasses should implement this method.")
