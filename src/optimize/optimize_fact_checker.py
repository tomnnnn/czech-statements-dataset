import argparse
import asyncio
import json
import os
import numpy as np
import time

import dspy
import mlflow
start = time.time()
from dataset_manager import Dataset
print("Import time dataset manager:", time.time() - start)

from dspy.teleprompt import MIPROv2

start = time.time()
from fact_checker.dspy_fact_checker import FactChecker
print("Import time fact checker:", time.time() - start)


start = time.time()
from sentence_transformers import SentenceTransformer
print("Import time transformers:", time.time() - start)

from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from src.dataset_manager.models import Statement
from src.fact_checker.evidence_retriever.search_functions import BGE_M3
from tqdm.asyncio import tqdm_asyncio


def evaluate(
    fact_checker: dspy.Module,
    examples: list[dspy.Example],
    output_folder: str,
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
        num_threads=24,
        display_progress=True,
        display_table=True,
        return_outputs=True,
        return_all_scores=True,
        provide_traceback=True,
    )

    # Evaluate the model
    eval_results = evaluate(fact_checker)

    # Calculate metrics
    metrics = classification_report(ref_labels, pred_labels, labels=['pravda', 'nepravda', 'neověřitelné'], output_dict=True, zero_division=np.nan)

    output_path = os.path.join(output_folder, result_filename)
    with open(output_path, "w") as f:
        json.dump(metrics,f, indent=4, ensure_ascii=False)


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
        max_bootstrapped_demos=0,  # zeroshot
        max_labeled_demos=0,  # zeroshot
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
        default="http://0.0.0.0:8000/v1",
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

    semaphore = asyncio.Semaphore(20)

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

    print("Building search functions for the statements...")
    results = await tqdm_asyncio.gather(
        *(build_search_fn(s) for s in statements),
        desc="Building evidence document indices",
        unit="indices",
    )

    for statement_id, search_fn in results:
        search_functions[statement_id] = search_fn

    return search_functions


def split_sample(sample: list, train_size=50, dev_size=50) -> tuple[list, list]:
    """
    Split the sample into train and dev sets.
    """
    # Shuffle the sample
    shuffled = shuffle(sample, random_state=42)

    # Split the sample into train and dev sets
    train_set = shuffled[:train_size]
    dev_set = shuffled[train_size : train_size + dev_size]

    return train_set, dev_set


def main():
    args = parse_args()

    lm = dspy.LM(
        args.model,
        api_base=args.api_base,
        api_key="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",
        temperature=0.0,
        max_tokens=3000
    )
    dspy.configure(lm=lm)

    mlflow.dspy.autolog()
    mlflow.set_experiment(args.name)

    # Load the dataset
    dataset = Dataset(args.dataset)

    # Initialize the FactChecker
    fact_checker = FactChecker(dataset, args.num_hops, args.num_docs)

    # Sample a subset of the dataset for evaluation
    print("Gettig statements from the dataset...")
    statements = dataset.get_statements(min_evidence_count=5)

    # Create examples for evaluation
    print(f"Sampling {args.num_train + args.num_dev} statements from the dataset...")
    train_statements, dev_statements = split_sample(
        statements, args.num_train, args.num_dev
    )

    # Create the search functions for the dataset
    print("Creating search functions for the dataset...")
    search_functions = asyncio.run(create_search_functions(dataset, statements))

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

    # Prepare output folder
    output_folder = os.path.join(args.output_folder, args.name)
    os.makedirs(output_folder, exist_ok=True)

    # Evaluate the fact checker
    evaluate(fact_checker, devset, output_folder, "pre_optimize_evaluation.json")

    # Optimize the fact checker
    optimized = optimize(fact_checker, trainset, args.output_folder)

    # Evaluate the optimized fact checker
    evaluate(optimized, devset, output_folder, "post_optimize_evaluation.json")


if __name__ == "__main__":
    main()
