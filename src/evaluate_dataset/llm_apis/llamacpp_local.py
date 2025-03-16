from ..llm_api import LanguageModelAPI
from llama_cpp import Llama


class LlamaCpp_Local(LanguageModelAPI):
    def __init__(self, model_path: str, filaname: str):
        super().__init__(model_path)
        self.model = Llama.from_pretrained(
            repo_id=model_path,
            filename=filaname,
            verbose=False
        )

    def _infer(self, conversations: list, batch_size: int = 1, max_new_tokens: int = 512):
        completions = []
        for convo in conversations:
            completion = self.model.create_chat_completion(
                convo, 
                temperature=0, 
                top_p=1, 
                max_tokens=max_new_tokens
            )
            completions.append(completion)

        return completions
