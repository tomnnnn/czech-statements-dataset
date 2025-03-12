from .bing_retriever import BingRetriever
from .google_retriever import GoogleRetriever
from .criteria_retriever import CriteriaRetriever
from .demagog_retriever import DemagogRetriever
from ..evidence_retriever import EvidenceRetriever


retrievers_dict = {
    'bing': BingRetriever,
    'google': GoogleRetriever,
    'criteria': CriteriaRetriever,
    'demagog': DemagogRetriever
}

def evidence_retriever_factory(api: str, api_key: str|None = None) -> EvidenceRetriever:
    if api not in retrievers_dict:
        raise ValueError(f"Unsupported evidence retriever API: {api}")

    return retrievers_dict[api](api_key)
