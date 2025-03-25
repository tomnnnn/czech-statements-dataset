import importlib
from ..llm_api import LanguageModelAPI

llm_api_dict = {
    "openai": "openai_api.OpenAI_API",
    "togetherai": "togetherai_api.TogetherAI_API",
    "transformers": "transformer_local.Transformer_Local",
    "vllm": "vllm_local.VLLM_Local",
    "mock": "mock_api.MockLanguageModelAPI",
    "llamacpp": "llamacpp_local.LlamaCpp_Local"
}

def llm_api_factory(api, *args, **kwargs) -> LanguageModelAPI:
    if api not in llm_api_dict:
        raise ValueError(f"Unknown API: {api}")

    module_name, class_name = llm_api_dict[api].rsplit(".", 1)
    
    # Import using the package name (assuming this file is in the same package)
    module = importlib.import_module(f"{__package__}.{module_name}")
    
    cls = getattr(module, class_name)
    return cls(*args, **kwargs)
