from ..llm_api import LanguageModelAPI
from llama_cpp import Llama
import asyncio


class LlamaCpp_Local(LanguageModelAPI):
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)

        if not model_path:
            # use local model file
            self.model = Llama(
                model_path=kwargs.get('model_file', ''),
                verbose=False,
                n_ctx=kwargs.get('ctx_len', 30000),
                n_gpu_layers=1
            )
        else:
            # use model from huggingface, model_file refers to a remote file
            self.model = Llama.from_pretrained(
                repo_id=model_path,
                filename=kwargs.get('model_file', '') or '',
                verbose=False,
                n_ctx=kwargs.get('ctx_len', 30000),
                n_gpu_layers=1,
            )

    async def _infer(self, conversations: list, batch_size: int = 1, max_new_tokens: int = 512):
        completions = []
        for convo in conversations:
            if self.chat_format:
                completion = self.model.create_chat_completion(
                    convo, 
                    temperature=0, 
                    top_p=1, 
                    max_tokens=max_new_tokens
                )
            else:
                completion = self.model(
                    convo, 
                    temperature=0,
                    max_tokens=max_new_tokens,
                    top_p=1, 
                )
            completions.append(completion)


        if self.chat_format:
            return [c['choices'][-1]['message']['content'] for c in completions]
        else:
            return[c['choices'][0]['text'] for c in completions]
