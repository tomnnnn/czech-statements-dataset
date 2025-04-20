import argparse
import sqlite3
import json
import pandas as pd
import os
import subprocess
import time
import logging
import dspy
import numpy as np
import requests
import mlflow

from dataset_manager import Dataset
from dspy.teleprompt import MIPROv2, SIMBA
from fact_checker.fact_checker import FactChecker
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# mute warnings from mlflow
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)

LOG_PATH = os.path.join("logs", f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
    ]
)


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
        num_threads=50,
        display_progress=True,
        display_table=True,
        return_all_scores=True,
        provide_traceback=True,
    )

    # Evaluate the model
    eval_results = evaluate(program=fact_checker)

    # Calculate metrics
    report = classification_report(
        ref_labels,
        pred_labels,
        labels=allowed_labels,
        output_dict=True,
        zero_division=np.nan, # type: ignore
    )

    for label, r in report.items(): # type: ignore
        if isinstance(r, dict):
            for metric_name, value in r.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
        else:
            # For accuracy
            mlflow.log_metric(label, r)

    report_pd = pd.DataFrame(report).transpose()
    mlflow.log_table(report_pd, artifact_file="classification_report.json")

    output_path = os.path.join(output_folder, result_filename)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return eval_results, report


def optimize(fact_checker: dspy.Module, train: list[dspy.Example], output_folder: str, seed: int = 42):
    """
    Optimize the fact checker in zeroshot settings using MIPROv2.
    """

    def metric(example, pred, trace=None):
        return pred.label.lower() == example.label.lower()

    teleprompter = MIPROv2(metric=metric, auto="medium")

    zeroshot_optimized = teleprompter.compile(
        fact_checker.deepcopy(),
        trainset=train,
        seed=seed,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        requires_permission_to_run=False,
    )

    # Save the optimized model
    output_path = os.path.join(output_folder, "optimized_model.pkl")
    zeroshot_optimized.save(output_path)

    return zeroshot_optimized

def optimize_simba(fact_checker: dspy.Module, train: list[dspy.Example], output_folder: str, seed: int = 42):
    """
    Optimize the fact checker in zeroshot settings using MIPROv2.
    """

    def metric(example, pred, trace=None):
        return pred.label.lower() == example.label.lower()

    teleprompter = SIMBA(metric=metric, max_steps=10, max_demos=10)

    zeroshot_optimized = teleprompter.compile(
        fact_checker.deepcopy(),
        trainset=train,
        seed=seed,
    )

    # Save the optimized model
    output_path = os.path.join(output_folder, "optimized_model.pkl")
    zeroshot_optimized.save(output_path)

    return zeroshot_optimized


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the FactChecker model.")
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="mipro",
        help="Optimizer to use. Options: 'mipro', 'simba'.",
    )
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


def split_sample(sample: list, allowed_labels: list[str], train_size=50, dev_size=50, seed=42) -> tuple[list, list]:
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
        statements = shuffle(statements, random_state=seed)

        # Split the statements into train and dev sets
        train_set.extend(statements[:count_per_label_train])
        dev_set.extend(statements[count_per_label_train:count_per_label_train + count_per_label_dev])


    return train_set, dev_set


def main():
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
    print(f"Allowed labels: {allowed_labels}")

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
        statements, allowed_labels, args.num_train, args.num_dev, seed=args.seed
    )

    # Create the train and dev sets
    trainset = [
        dspy.Example(
            statement=statement,
            label=statement.label,
        ).with_inputs("statement", "search_func")
        for statement in train_statements
    ]

    devset = [
        dspy.Example(
            statement=statement,
            label=statement.label,
        ).with_inputs("statement", "search_func")
        for statement in dev_statements
    ]

    do_optimize = args.num_train > 0
    run_name = f"Optimization Run {args.optimizer.upper()}" if do_optimize else "Evaluation Run"
    run_name += f" {args.mode.capitalize()}"

    # Prepare output folder
    output_folder = os.path.join(args.output_folder, args.name, run_name.replace(" ", "_"))
    os.makedirs(output_folder, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        print("Optimizing the fact checker..." if do_optimize else "Evaluating the fact checker...")

        mlflow.log_param("num_train", args.num_train)
        mlflow.log_param("num_dev", args.num_dev)
        mlflow.log_param("num_hops", args.num_hops)
        mlflow.log_param("num_docs", args.num_docs)
        mlflow.log_param("allowed_labels", allowed_labels)
        label_distribution = {label: len([s for s in train_statements + dev_statements if s.label.lower() == label]) for label in allowed_labels}
        for label, count in label_distribution.items():
            mlflow.log_param(f"label_{label}", count)

        evaluation_name = "evaluation_results.json" if not do_optimize else "pre_optimize_evaluation.json"

        # Evaluate the fact checker
        if do_optimize:
            print("Pre-optimization evaluation...")

        evaluate(fact_checker, devset, output_folder, allowed_labels, evaluation_name)

        if do_optimize:
            # Optimize the fact checker
            print("Optimizing the fact checker...")

            if args.optimizer == "mipro":
                optimized = optimize(fact_checker, trainset, output_folder, seed=args.seed)
            elif args.optimizer == "simba":
                optimized = optimize_simba(fact_checker, trainset, output_folder, seed=args.seed)
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}. Use 'mipro' or 'simba'.")

            # Evaluate the optimized fact checker
            print("Post-optimization evaluation...")
            evaluate(optimized, devset, output_folder, allowed_labels, "post_optimize_evaluation.json")

    print("Evaluation completed.")
    print("Results saved to:", os.path.join(output_folder))


if __name__ == "__main__":
    main()
