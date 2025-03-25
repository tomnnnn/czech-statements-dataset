from ..llm_api import LanguageModelAPI

class MockLanguageModelAPI(LanguageModelAPI):
    def __init__(
        self,
        model_path: str,
        distribute_load: bool = False,
        gpu_memory_utilization: float = 0.85,
        **kwargs,
    ):
        print(kwargs)

    def _infer(self, conversations, batch_size=8, max_new_tokens=1000, **kwargs):
        if kwargs.get('chat', False):
            return [
                "\r\n".join(passage["content"] if len(passage["content"]) < 200 else passage["content"][:200] + "..." for passage in conversation)
                for conversation in conversations
            ]
        else:
            return [convo[:1000] for convo in conversations]
