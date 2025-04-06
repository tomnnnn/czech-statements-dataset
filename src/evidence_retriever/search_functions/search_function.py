from dataset_manager.models import Segment


class SearchFunction():
    def __init__(self, corpus: list[Segment], **kwargs):
        pass

    def _index(self):
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> list[Segment]:
        raise NotImplementedError

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Segment]]:
        raise NotImplementedError
