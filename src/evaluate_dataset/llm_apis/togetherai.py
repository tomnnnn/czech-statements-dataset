from llm_api import LangugaeModelAPI
from together import Together
import os

class TogetherAI_API(LanguageModelAPI):
    """
    Implmentation of LanguageModelInterface through Together AI's interface
    """
    examples = []
    system_prompt = ""
    generation_prompt = ""

    def __init__(self, model_path, max_tokens=1000):
        super().__init__(model_path, max_tokens)
        self.client = Together(api_key=os.getenv("MODEL_API_KEY"))

    def _infer(self, conversations, batch_size=8):
        results = []
        for chat in conversations:
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=chat,
                max_tokens=self.max_tokens,
            )

            results.append(completion.choices[0].message.content)

        return results
