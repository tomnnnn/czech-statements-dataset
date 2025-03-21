from ..llm_api import LanguageModelAPI
import os
from openai import OpenAI

class OpenAI_API(LanguageModelAPI):
    """
    Implementation of LanguageModelInterface using OpenAI's API
    """
    examples = []
    system_prompt = ""
    generation_prompt = ""

    def __init__(self,model_path="gpt-4o", **kwargs):
        super().__init__(model_path)
        self.client = OpenAI(api_key=os.getenv("MODEL_API_KEY"))

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        results = []
        for chat in conversations:
            print(len("".join([x["content"] for x in chat])))
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=chat,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )

            results.append(response.choices[0].message.content)

        return results
