from dataset_manager.models import Segment


class SearchFunction():
    def __init__(self, corpus: list[Segment], **kwargs):
        pass

    def search(self, query: str|list, k: int = 10) -> list[Segment]|list[list[Segment]]:
        raise NotImplementedError
