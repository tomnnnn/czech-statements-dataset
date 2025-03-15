from ..llm_api import LanguageModelAPI

class MockLanguageModelAPI(LanguageModelAPI):
    def _infer(self, conversations, batch_size=8, max_new_tokens=1000):
        return [
            "\r\n".join(passage["content"] if len(passage["content"]) < 200 else passage["content"][:200] + "..." for passage in conversation)
            for conversation in conversations
        ]
