import argparse
import asyncio
import json
import os
import subprocess
import time
import logging
import dspy
import numpy as np
import requests
import mlflow

from dataset_manager import Dataset
from dspy.teleprompt import MIPROv2
from fact_checker import FactChecker
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from tqdm.asyncio import tqdm_asyncio

from src.dataset_manager.models import Statement
from src.fact_checker.search_functions import BGE_M3, BM25

# mute warnings from mlflow
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)


def get_allowed_labels(mode: str) -> list[str]:
    """
    Get the allowed labels based on the classification mode.
    """
    if mode == "binary":
        return ["pravda", "nepravda"]
    elif mode == "ternary":
        return ["pravda", "nepravda", "neověřitelné"]
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'binary' or 'ternary'.")


def evaluate(
    fact_checker: dspy.Module,
    examples: list[dspy.Example],
    output_folder: str,
    allowed_labels: list[str],
    result_filename: str = "evaluation_results.json",
):
    """
    Evaluate the fact checker on the provided examples and saves the results.
    """

    pred_labels = []
    ref_labels = []

    def metric(example, pred, trace=None):
        pred_labels.append(pred.label.lower())
        ref_labels.append(example.label.lower())

        return pred.label.lower() == example.label.lower()

    # Create the evaluator
    evaluate = dspy.Evaluate(
        devset=examples,
        metric=metric,
        num_threads=200,
        display_progress=True,
        display_table=True,
        return_all_scores=True,
        provide_traceback=True,
    )

    # Evaluate the model
    eval_results = evaluate(program=fact_checker)

    # Calculate metrics
    metrics = classification_report(
        ref_labels,
        pred_labels,
        labels=allowed_labels,
        output_dict=True,
        zero_division=np.nan, # type: ignore
    )

    output_path = os.path.join(output_folder, result_filename)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    return eval_results, metrics


def optimize(fact_checker: dspy.Module, train: list[dspy.Example], output_folder: str):
    """
    Optimize the fact checker in zeroshot settings using MIPROv2.
    """

    def metric(example, pred, trace=None):
        return pred.label.lower() == example.label.lower()

    teleprompter = MIPROv2(metric=metric, auto="light")

    zeroshot_optimized = teleprompter.compile(
        fact_checker.deepcopy(),
        trainset=train,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        requires_permission_to_run=False,
    )

    # Save the optimized model
    output_path = os.path.join(output_folder, "optimized_model.pkl")
    zeroshot_optimized.save(output_path)

    return zeroshot_optimized


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the FactChecker model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/dataset_demagog.sqlite",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hosted_vllm/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        help="Model name or path.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="fact-checker-eval",
        help="Path to save the evaluation results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="OptimizeFactCheckerRandom",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=50,
        help="Number of training examples.",
    )
    parser.add_argument(
        "--num-dev",
        type=int,
        default=50,
        help="Number of development examples.",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=4,
        help="Number of hops for the retriever.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=4,
        help="Number of documents to retrieve per hop.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        help="Classification mode for the fact checker.",
    )
    return parser.parse_args()


async def create_search_functions(dataset: Dataset, statements: list[Statement]):
    # Get segments for the statements
    print("Getting segments for the statements...")
    segments = dataset.get_segments_by_statements([s.id for s in statements])

    # Filter statements with segments
    statements = [s for s in statements if s.id in segments]

    search_functions = {}

    # Load model for encoding the segments
    print("Loading model for encoding the segments...")
    model = SentenceTransformer("BAAI/BGE-M3")

    semaphore = asyncio.Semaphore(100)

    async def build_search_fn(statement: Statement):
        async with semaphore:
            segment_list = segments[statement.id]
            index_path = f"indexes/{statement.id}.faiss"
            load_index = os.path.exists(index_path)

            search_fn = BGE_M3(
                segment_list,
                save_index=not load_index,
                load_index=load_index,
                index_path=index_path,
                model=model,
            )
            await search_fn.index_async()
            return statement.id, search_fn

        # search_fn = BM25(segments[statement.id])
        # return statement.id, search_fn

    print("Building search functions for the statements...")
    results = await tqdm_asyncio.gather(
        *(build_search_fn(s) for s in statements),
        desc="Building evidence document indices",
        unit="indices",
    )

    for statement_id, search_fn in results:
        search_functions[statement_id] = search_fn

    return search_functions


def split_sample(sample: list, allowed_labels: list[str], train_size=50, dev_size=50) -> tuple[list, list]:
    """
    Split the sample into train and dev sets.
    """
    label_map = {
        label: [s for s in sample if s.label.lower() == label]
        for label in allowed_labels
    }

    count_per_label_train = train_size // len(allowed_labels)
    count_per_label_dev = dev_size // len(allowed_labels)

    train_set = []
    dev_set = []

    for label, statements in label_map.items():
        # Shuffle the statements
        statements = shuffle(statements, random_state=42)

        # Split the statements into train and dev sets
        train_set.extend(statements[:count_per_label_train])
        dev_set.extend(statements[count_per_label_train:count_per_label_train + count_per_label_dev])

    return train_set, dev_set


def launch_vllm(name):
    """
    Launch the VLLM server with the specified model and wait for it to be ready.
    """
    log = open(f"vllm_{name}.log", "a")
    subprocess.Popen(
        [
            "vllm",
            "serve",
            "--enable-chunked-prefill",
            "true",
            "--gpu-memory-utilization",
            "0.85",
            "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        ],
        stdout=log,
        stderr=log,
    )

    print(f"Waiting for LLM server to be up...")
    while True:
        try:
            response = requests.get("http://0.0.0.0:8000/health")

            if response.status_code == 200:
                print("Server is up and running.")
                break
            else:
                print(f"Server is not ready yet, status code: {response.status_code}")
                time.sleep(10)
        except Exception:
            time.sleep(10)


async def main():
    args = parse_args()

    lm = dspy.LM(
        args.model,
        api_base=args.api_base,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0,
        max_tokens=3000,
        rpm=60
    )
    dspy.configure(lm=lm)

    mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
    mlflow.set_experiment(args.name)

    # Get allowed labels based on the mode
    allowed_labels = get_allowed_labels(args.mode)

    # Load the dataset
    dataset = Dataset(args.dataset)

    # Initialize the FactChecker
    fact_checker = FactChecker(args.num_hops, args.num_docs, mode=args.mode)

    # Sample a subset of the dataset for evaluation
    print("Gettig statements from the dataset...")
    statements = dataset.get_statements(min_evidence_count=5)

    # Create examples for evaluation
    print(f"Sampling {args.num_train + args.num_dev} statements from the dataset...")
    train_statements, dev_statements = split_sample(
        statements, allowed_labels, args.num_train, args.num_dev
    )

    # Create the search functions for the dataset
    print("Creating search functions for the dataset...")
    search_functions = await create_search_functions(dataset, dev_statements + train_statements)

    # Create the train and dev sets
    trainset = [
        dspy.Example(
            statement=statement,
            search_func=search_functions[statement.id],
            label=statement.label,
        ).with_inputs("statement", "search_func")
        for statement in train_statements
    ]

    devset = [
        dspy.Example(
            statement=statement,
            search_func=search_functions[statement.id],
            label=statement.label,
        ).with_inputs("statement", "search_func")
        for statement in dev_statements
    ]

    # # Launch the VLLM server and wait for it to be ready
    launch_vllm(args.name)

    # Prepare output folder
    output_folder = os.path.join(args.output_folder, args.name)
    os.makedirs(output_folder, exist_ok=True)

    do_optimize = args.num_train > 0

    run_name = "Optimization Run" if do_optimize else "Evaluation Run"

    with mlflow.start_run(run_name=run_name):
        print("Optimizing the fact checker..." if do_optimize else "Evaluating the fact checker...")

        evaluation_name = "evaluation_results.json" if not do_optimize else "optimized_evaluation_results.json"

        # Evaluate the fact checker
        if do_optimize:
            print("Pre-optimization evaluation...")

        evaluate(fact_checker, devset, output_folder, allowed_labels, evaluation_name)

        if do_optimize:
            # Optimize the fact checker
            print("Optimizing the fact checker...")
            optimized = optimize(fact_checker, trainset, output_folder)

            # Evaluate the optimized fact checker
            print("Post-optimization evaluation...")
            evaluate(optimized, devset, output_folder, allowed_labels, "post_optimize_evaluation.json")


if __name__ == "__main__":
    asyncio.run(main())
