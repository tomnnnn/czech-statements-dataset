import pprint
from ..llm_api import LanguageModelAPI
from llama_cpp import Llama


class LlamaCpp_Local(LanguageModelAPI):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path)
        print("filename", kwargs.get('filename', ''))
        self.model = Llama.from_pretrained(
            repo_id=model_path,
            filename=kwargs.get('filename', ''),
            verbose=False,
            n_ctx=kwargs.get('n_ctx', 50000),
            n_gpu_layers=1
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


        return [c['choices'][0]['message']['content'] for c in completions]
