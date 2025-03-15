from ..llm_api import LanguageModelAPI
from together import Together
import os
import pprint

class TogetherAI_API(LanguageModelAPI):
    """
    Implmentation of LanguageModelInterface through Together AI's interface
    """
    examples = []
    system_prompt = ""
    generation_prompt = ""

    def __init__(self, model_path):
        super().__init__(model_path)
        self.client = Together(api_key=os.getenv("MODEL_API_KEY"))

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        results = []
        for chat in conversations:
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=chat,
                max_tokens=max_new_tokens,
            )

            results.append(completion.choices[0].message.content)

        return results
