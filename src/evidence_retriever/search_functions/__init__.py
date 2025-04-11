from .bge_m3 import BGE_M3
from src.dataset_manager.models import Segment
from .bm25 import BM25
from .search_function import SearchFunction

_search_function_dict = {
    "bm25": BM25,
    "bge-m3": BGE_M3
}

def search_function_factory(search_function_name: str, corpus: list[Segment], **kwargs) -> SearchFunction:
    return _search_function_dict[search_function_name](corpus, **kwargs)

def search_function_from_index(search_function_name: str, index_path: str) -> SearchFunction:
    """
    Load the index from a file.
    """
    search_function = _search_function_dict[search_function_name]()
    search_function.from_index(index_path)
    return search_function

def search_function_from_corpus(search_function_name: str, corpus: list[Segment], **kwargs) -> SearchFunction:
    """
    Load the index from a corpus.
    """
    search_function = _search_function_dict[search_function_name]()
    search_function.from_corpus(corpus, **kwargs)
    return search_function
