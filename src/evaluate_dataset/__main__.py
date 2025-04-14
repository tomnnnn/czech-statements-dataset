from .evaluator import FactCheckingEvaluator
from fact_checker import FactChecker, BasicPredictor, HopRetriever, SimpleRetriever
from .config import load_config
import yaml
from utils.llm_apis import llm_api_factory
from dataset_manager import Dataset
import json
import asyncio
from dataclasses import asdict
import dspy
import logging
import os
import time

logger = logging.getLogger(__name__)

retriever_dict = {
    "hop": HopRetriever,
    "simple": SimpleRetriever
}

def load_prompt_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config['system_prompt'], config['generation_prompt'], config['prompt_template'], config['allowed_labels']


async def main():
    start = time.time()
    config = load_config()

    # TODO: move to the retriever
    lm = dspy.LM("openai/" + config.model_name, api_base=config.api_base_url, max_tokens=3000)
    dspy.configure(lm=lm)

    dataset = Dataset(config.dataset_path, read_only=True)

    llm = llm_api_factory(config.model_api, config.model_name, **asdict(config), show_progress=False)
    system_prompt, generation_prompt, prompt_template, allowed_labels = load_prompt_config(config.prompt_config)

    retriever = retriever_dict[config.retriever](num_docs=config.num_docs)
    predictor = BasicPredictor(llm,system_prompt,generation_prompt,prompt_template,allowed_labels)
    fc = FactChecker(retriever,predictor)

    evaluator = FactCheckingEvaluator(dataset, fc, config)

    startup_time = time.time() - start

    start = time.time()
    predictions, metrics = await evaluator.run()
    benchmark_time = time.time() - start

    results_dirpath = os.path.join(config.out_folder, config.model_name.split('/')[-1])
    predictions_path = os.path.join(results_dirpath, f"predictions_{config.index}.json")
    metrics_path = os.path.join(results_dirpath, f"metrics_{config.index}.json")

    os.makedirs(results_dirpath, exist_ok=True)

    # Save predictions and metrics to JSON files
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


    print(f"Startup time: {startup_time:.2f} seconds")
    print(f"Benchmark time: {benchmark_time:.2f} seconds")
    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")
    print("Evaluation completed.")

if __name__ == "__main__":
    asyncio.run(main())
