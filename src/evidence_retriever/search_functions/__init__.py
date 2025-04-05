from .bge_m3 import BGE_M3
from .bm25 import BM25
from .search_function import SearchFunction

_search_function_dict = {
    "bm25": BM25,
    "bge3": BGE_M3
}

def search_function_factory(search_function_name: str, corpus: list[dict], **kwargs) -> SearchFunction:
    return _search_function_dict[search_function_name](corpus, **kwargs)

