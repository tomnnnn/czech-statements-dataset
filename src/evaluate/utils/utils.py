import json
import os
import random
from typing import Any, Literal
import dspy
from sklearn.utils import shuffle
import mlflow
import pandas as pd
from dataset_manager import Dataset
from dataset_manager.models import Statement
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import dspy

def setup_dspy(model: str = "hosted_vllm/Qwen/Qwen2.5-32B-Instruct-AWQ", api_base: str = "http://localhost:8000/v1"):
    lm = dspy.LM(
        model,
        api_base=api_base,
        api_key=os.environ.get("API_KEY"),
        temperature=0.0,
        max_tokens=3000,
        rpm=60
    )
    dspy.configure(lm=lm)


def setup_mlflow(experiment_name: str = "thesis", run_name: str = "dspy"):
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
    mlflow.set_experiment(experiment_name)


def sample_statements(sample_type: Literal["balanced", "filtered", "random"], dataset_path: str = "datasets/dataset_demagog.sqlite", portion: float = 0.2, allowed_labels: list[str] = ["pravda", "nepravda"]) -> list[Statement]:
    mlflow.log_param("sample_type", sample_type)
    mlflow.log_param("portion", portion)
    mlflow.log_param("allowed_labels", allowed_labels)

    # TODO: Remove hardcoded paths
    random.seed(42)
    dataset = Dataset(dataset_path)
    statements = dataset.get_statements(allowed_labels=allowed_labels)

    if sample_type == "balanced":
        label_map = {
            label: [s for s in statements if s.label.lower() == label]
            for label in allowed_labels
        }
        num_samples = int(len(statements)*portion)
        num_samples_per_label = num_samples // len(label_map)

        samples = []

        for subset in label_map.values():
            statements = shuffle(statements, random_state=42)
            samples += subset[:num_samples_per_label]

    elif sample_type == "filtered":
        with open("fact-checker-eval/verifiable_statements.json", "r") as f:
            verifiable_statements = json.load(f)
            verifiable_ids = [item['id'] for item in verifiable_statements]

        statements = [s for s in statements if s.id in verifiable_ids]
        _, samples = train_test_split(statements, test_size=portion, stratify=[s.label for s in statements]) if portion < 1 else statements

    elif sample_type == "random":
        _, samples = train_test_split(statements, test_size=portion, stratify=[s.label for s in statements]) if portion < 1 else statements

    mlflow.log_param("num_samples", len(samples))

    # log label distribution
    label_distribution = {label: 0 for label in allowed_labels}
    for s in samples:
        label_distribution[s.label.lower()] += 1

    for label, count in label_distribution.items():
        mlflow.log_param(f"label_distribution_{label}", count)

    return samples


def create_examples(statements: list[Statement]):
    examples = [
        dspy.Example(
            statement=statement,
            label=statement.label,
        ).with_inputs("statement")
        for statement in statements
    ]
    return examples


def evaluate(
    fact_checker: dspy.Module,
    examples: list[dspy.Example],
    output_folder: str,
    allowed_labels: list[str],
    name: str = "evaluation",
) -> tuple[Any, dict]:
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
        num_threads=20,
        max_errors=100,
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
                mlflow.log_metric(f"{name}_{label}_{metric_name}", value)
        else:
            # For accuracy
            mlflow.log_metric(f"{name}_{label}", r)

    report_pd = pd.DataFrame(report).transpose()
    mlflow.log_table(report_pd, artifact_file=f"{name}_classification_report.json")

    output_path = os.path.join(output_folder, name + ".json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return eval_results, report
