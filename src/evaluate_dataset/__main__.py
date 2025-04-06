from .evaluator import FactCheckingEvaluator
from .fact_checker import FactChecker
from .llm_apis import llm_api_factory
from .config import load_config
from dataset_manager import Dataset
from evidence_retriever import HopRetriever
import json
from dataclasses import asdict
import dspy
import os

if __name__ == "__main__":
    config = load_config()
    llm = llm_api_factory(config.model_api, config.model_name, **asdict(config))

    # TODO: move to the retriever
    lm = dspy.LM("hosted_vllm/" + config.model_name, api_base=config.api_base_url)
    dspy.configure(lm=lm, provide_traceback=True)

    dataset = Dataset(config.dataset_path)
    corpus = dataset.get_segments()

    retriever = HopRetriever(config.search_algorithm, corpus, num_docs=config.search_k_segments, num_hops=config.num_hops)
    fc = FactChecker(llm, retriever,config)

    evaluator = FactCheckingEvaluator(dataset, fc, config)

    predictions, metrics = evaluator.run()

    predictions_path = os.path.join(config.out_folder, f"predictions_{config.index}.json")
    metrics_path = os.path.join(config.out_folder, f"metrics_{config.index}.json")

    os.makedirs(config.out_folder, exist_ok=True)

    # Save predictions and metrics to JSON files
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")
    print("Evaluation completed.")

