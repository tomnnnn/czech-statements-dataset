import asyncio
import functools
import json
import logging
import os
import time
from dataclasses import asdict

import dspy
from dataset_manager import Dataset
from fact_checker import BasicPredictor, FactChecker, HopRetriever, SimpleRetriever
from utils.llm_apis import llm_api_factory

from .config import load_config
from .evaluator import FactCheckingEvaluator

logger = logging.getLogger(__name__)

retriever_dict = {"hop": HopRetriever, "simple": SimpleRetriever}


def measure_time(activity_desc):
    """
    Decorator to measure the execution time of a function.
    """

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{activity_desc}: {end_time - start_time:.2f} seconds")
            return result

        return wrapper

    return _decorator


def measure_time_async(activity_desc):
    """
    Decorator to measure the execution time of an asynchronous function.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            print(f"{activity_desc}: {end_time - start_time:.2f} seconds")
            return result

        return wrapper

    return decorator


@measure_time("Benchmark initialization")
def init_benchmark(config):
    # TODO: move to the retriever
    dspy_lm = dspy.LM(
        "openai/" + config.model_name, api_base=config.api_base_url, max_tokens=3000
    )
    dspy.configure(lm=dspy_lm)

    dataset = Dataset(config.dataset_path, read_only=True)
    lm = llm_api_factory(
        config.model_api, config.model_name, **asdict(config), show_progress=False
    )
    retriever = retriever_dict[config.retriever](num_docs=config.num_docs)
    predictor = BasicPredictor(lm, config.prompt_config)
    fc = FactChecker(retriever, predictor, show_progress=False)

    evaluator = FactCheckingEvaluator(dataset, fc, config)

    return evaluator


@measure_time_async("Benchmark run")
async def run_benchmark(benchmark):
    return await benchmark.run()


def save_results(config, metrics, predictions):
    results_dirpath = os.path.join(config.out_folder, config.model_name.split("/")[-1])
    os.makedirs(results_dirpath, exist_ok=True)

    predictions_path = os.path.join(results_dirpath, f"predictions_{config.index}.json")
    metrics_path = os.path.join(results_dirpath, f"metrics_{config.index}.json")

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")


async def main():
    config = load_config()

    benchmark = init_benchmark(config)
    predictions, metrics = await run_benchmark(benchmark)
    save_results(config, metrics, predictions)

    print("Evaluation completed.")


if __name__ == "__main__":
    asyncio.run(main())
