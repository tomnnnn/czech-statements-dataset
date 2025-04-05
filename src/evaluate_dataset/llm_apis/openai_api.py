from ..llm_api import LanguageModelAPI
import os
from openai import OpenAI
from tqdm import tqdm

class OpenAI_API(LanguageModelAPI):
    """
    Implementation of LanguageModelInterface using OpenAI's API
    """
    examples = []
    system_prompt = ""
    generation_prompt = ""

    def __init__(self,model_path="gpt-4o", **kwargs):
        super().__init__(model_path)
        self.client = OpenAI(api_key=os.getenv("MODEL_API_KEY"), base_url=kwargs.get("api_base_url", None))

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        results = []

        with tqdm(total=len(conversations), desc="Processing Conversations", unit="conversation") as pbar:
            for chat in conversations:
                print(len("".join([x["content"] for x in chat])))  # Optional: You can remove this line if not needed
                response = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=chat,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )

                results.append(response.choices[0].message.content)
                
                # Update progress bar
                pbar.update(1)

        return results
