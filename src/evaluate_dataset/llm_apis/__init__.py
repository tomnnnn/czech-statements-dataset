from .openai_api import OpenAI_API
from .togetherai_api import TogetherAI_API
from .transformer_local import Transformer_Local
from .vllm_local import VLLM_Local
from .mock_api import MockLanguageModelAPI

from ..llm_api import LanguageModelAPI

llm_api_dict = {
    "openai": OpenAI_API,
    "togetherai": TogetherAI_API,
    "transformers": Transformer_Local,
    "vllm": VLLM_Local,
    "mock": MockLanguageModelAPI,
}

def llm_api_factory(api, *args) -> LanguageModelAPI:
    return llm_api_dict[api](*args)
