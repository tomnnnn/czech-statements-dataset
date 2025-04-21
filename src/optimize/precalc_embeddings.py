import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio
import logging
from dataset_manager import Dataset
from sklearn.utils import shuffle
from src.fact_checker.search_functions.bge_remote import RemoteSearchFunction

# mute warnings from mlflow
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)


def configure_logging(log_path):
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
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
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name of the run.",
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

    print("Lengths of the label map:")
    for label, statements in label_map.items():
        print(f"{label}: {len(statements)}")

    count_per_label_train = train_size // len(allowed_labels)
    count_per_label_dev = dev_size // len(allowed_labels)

    print(f"Count per label for train: {count_per_label_train}, dev: {count_per_label_dev}")

    train_set = []
    dev_set = []

    for label, statements in label_map.items():
        # Shuffle the statements
        statements = shuffle(statements, random_state=seed)

        # Split the statements into train and dev sets
        train_set.extend(statements[:count_per_label_train])
        dev_set.extend(statements[count_per_label_train:count_per_label_train + count_per_label_dev])


    return train_set, dev_set


async def main():
    args = parse_args()
    sem = asyncio.Semaphore(100)

    # Get allowed labels based on the mode
    allowed_labels = get_allowed_labels(args.mode)
    print(f"Allowed labels: {allowed_labels}")

    # Load the dataset
    dataset = Dataset(args.dataset)


    # Sample a subset of the dataset for evaluation
    print("Gettig statements from the dataset...")
    statements = dataset.get_statements(min_evidence_count=5)

    # Create examples for evaluation
    print(f"Sampling {args.num_train + args.num_dev} statements from the dataset...")
    train_statements, dev_statements = split_sample(
        statements, allowed_labels, args.num_train, args.num_dev, seed=args.seed
    )

    async def search_limited(statement):
        async with sem:
            return await retriever.search_async(statement.statement, args.num_hops, statement.id)

    sample = train_statements + dev_statements
    retriever = RemoteSearchFunction()
    coroutines = [search_limited(statement) for statement in sample]

    print("Pre-calculating embeddings...")
    results = await tqdm_asyncio.gather(*coroutines)


if __name__ == "__main__":
    asyncio.run(main())
