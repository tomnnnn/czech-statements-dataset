from dataset_manager.orm import rows2dict
from .evaluator import FactCheckingEvaluator
from .fact_checker import FactChecker
from .llm_apis import llm_api_factory
from .config import load_config
from dataset_manager import Dataset
from evidence_retriever import HopRetriever
import json
from dataclasses import asdict
import dspy

if __name__ == "__main__":
    config = load_config()
    llm = llm_api_factory(config.model_api, config.model_name, **asdict(config))

    # TODO: move to the retriever
    lm = dspy.LM("hosted_vllm/" + config.model_name, api_base=config.api_base_url)
    dspy.configure(lm=lm, provide_traceback=True)

    dataset = Dataset(config.dataset_path)
    corpus = rows2dict(dataset.get_segments())

    retriever = HopRetriever("bm25", corpus, num_docs=3, num_hops=2)
    fc = FactChecker(llm, retriever,config)

    evaluator = FactCheckingEvaluator(dataset, fc, config)

    predictions, metrics = evaluator.run()

    # Save predictions and metrics to JSON files
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Predictions and metrics saved to predictions.json and metrics.json.")
