class SearchFunction():
    def __init__(self, corpus: list[dict], **kwargs):
        pass

    def _index(self):
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[dict]:
        raise NotImplementedError

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[dict]]:
        raise NotImplementedError
