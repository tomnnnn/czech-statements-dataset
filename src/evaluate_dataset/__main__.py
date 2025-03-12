import pprint
import argparse
import datetime
import json
import logging
import os
import random
import re
import tldextract

import yaml
from sklearn.model_selection import train_test_split

from .config import CONFIG
from .local_llm import Model
from .utils import calculate_metrics

logger = logging.getLogger(__name__)

def load_prompt_config(config_location) -> tuple:
    """
    Load system prompt and examples from configuration file. If the file does not exist, return empty system prompt and no examples.
    """

    if not os.path.exists(config_location):
        logger.warning(
            f"Prompt configuration file {config_location} not found. Running without system prompt and no examples."
        )
        return "", []

    with open(config_location, "r") as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)

    return prompts.get("system_prompt", ""), prompts.get("examples", [])


def extract_label(response, allowed_labels):
    if response:
        labels_regex = "|".join(allowed_labels)
        label = re.findall(labels_regex, response.lower())
        label = label[-1] if label else "nolabel"
    else:
        label = "nolabel"

    return label


def prompt_builder(statements, evidence_dir=None):
    """
    Build prompts for given statements. Optionally include evidence from evidence_dir.

    Args:
    statements (List): List of statements.
    evidence_dir (str): Path to folder with evidence files.
    """
    max_content_len = 5000
    prompts = []
    for statement in statements:
        statement_str = (
            f"{statement['statement']} - {statement['author']}, {statement['date']}"
        )
        if evidence_dir:
            evidence_path = os.path.join(evidence_dir, f"{statement['id']}.json")
            try:
                with open(evidence_path, "r") as f:
                    evidence = json.load(f)
                    truncated = [
                        {
                            'Titulek': e['title'],
                            'Text': e['content'][:max_content_len] + "..." if len(e['content']) > max_content_len else e['content'],
                            'Zdroj': tldextract.extract(e['url']).domain + "." + tldextract.extract(e['url']).suffix,
                        }
                        for e in evidence
                    ]

                    evidence_str = json.dumps(truncated, indent=2, ensure_ascii=False)

            except FileNotFoundError:
                logger.warning(
                    f"Evidence file for statement {statement['id']} not found."
                )
                evidence_str = ""

            prompts.append(f"{statement_str}\nPodpůrné dokumenty: {evidence_str}")
        else:
            prompts.append(statement_str)

    return prompts


def build_examples(
    statements,
    seed=None,
    allowed_labels=["pravda", "nepravda", "neověřitelné"],
    evidence_dir=None,
    include_explanation=False,
    count_for_each_label=1,
):
    """
    Samples pseudo-randomly examples from statements for each label and uses them as examples for few-shot prompting. Optionally includes evidence and explanation.

    Args:
    statements (List): List of statements.
    seed (int): Seed for random sampling.
    allowed_labels (List): List of allowed labels.
    include_evidence (bool): Include evidence in examples.
    include_explanation (bool): Include explanation in examples.
    count_for_each_label (int): Number of examples for each label.
    """
    if seed:
        random.seed(seed)

    example_statements = []
    for label in allowed_labels:
        filtered_statements = [
            stmt for stmt in statements if stmt["assessment"].lower() == label
        ]
        if len(filtered_statements) < count_for_each_label:
            raise ValueError(
                f"Not enough statements for label '{label}'. Required: {count_for_each_label}, Found: {len(filtered_statements)}"
            )

        # pseudo-randomly sample
        shuffled_items = sorted(filtered_statements, key=lambda x: random.random())  # Shuffle while keeping order fixed
        example_statements.extend(shuffled_items[:count_for_each_label])

    inputs = prompt_builder(example_statements, evidence_dir=evidence_dir)

    explanations = [
        (
            f"Úvaha: {stmt['explanation'] if stmt['explanation_brief'] else stmt['explanation']}"
            if include_explanation
            else ""
        )
        for stmt in example_statements
    ]

    examples = [
        {"input": input_text, "output": explanation + "\nHodnocení: " + stmt["assessment"].lower()}
        for input_text, stmt, explanation in zip(
            inputs, example_statements, explanations
        )
    ]

    return examples


def eval_dataset(
    model_id,
    statements,
    result_dir,
    evidence_dir=None,
    prompt_config_path=None,
    index=0,
    with_explanation=False,
    example_count=0,
    batch_size=1,
    allowed_labels=['pravda', 'nepravda']
):
    """
    Test chosen models accuracy of labeling given statements with zero-shot prompt.
    Results are saved to result_dir.

    Args:
    model_id (str): ID of the model to evaluate.
    statements (List): List of statements to evaluate.
    result_dir (str): Path to directory where results will be saved.
    evidence_dir (str): Path to directory with evidence files.
    prompt_config_path (str): Path to prompt configuration file.
    index (int): Index for parallelization.
    with_explanation (bool): Require explanation in the model output.
    example_count (int): Number of examples for each label to use.
    batch_size (int): Inference batch size.
    allowed_labels (List): List of allowed labels.
    """

    model = Model(model_id, max_tokens=4000 if with_explanation else 10)
    system_prompt, _ = load_prompt_config(prompt_config_path)
    examples = build_examples(
        statements,
        seed=42,
        evidence_dir=evidence_dir,
        include_explanation=with_explanation,
        count_for_each_label=example_count,
        allowed_labels=allowed_labels
    )
    logger.info(f"Loaded system prompt: {system_prompt}")
    logger.info(f"Loaded {len(examples)} examples.")

    # debug
    with open(os.path.join(result_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)

    model.set_system_prompt(system_prompt)
    model.set_examples(examples)

    prompts = prompt_builder(statements, evidence_dir=evidence_dir)

    # generate responses
    logger.info(f"Generating responses for {len(prompts)} prompts.")
    responses = model(prompts, batch_size)
    verdicts = [
        {
            "id": statement["id"],
            "statement": statement["statement"],
            "author": statement["author"],
            "date": statement["date"],
            "response": response,
            "label": extract_label(response, allowed_labels),
        }
        for statement, response in zip(statements, responses)
    ]

    # save responses
    responses_dest = os.path.join(result_dir, f"responses_{index}.json")
    with open(os.path.join(responses_dest), "w") as f:
        json.dump(verdicts, f, indent=4, ensure_ascii=False)
        logger.info(f"Responses saved to {responses_dest}")

    # save prompts
    prompts_dest = os.path.join(result_dir, f"prompts_{index}.json")
    with open(prompts_dest, "w") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
        logger.info(f"Prompts saved to {prompts_dest}")

    if index == 0:
        # calculate and save metrics
        results_dest = os.path.join(result_dir, f"metrics.json")
        with open(results_dest, "w") as f:
            json.dump(
                calculate_metrics(statements, verdicts, allowed_labels),
                f,
                indent=4,
                ensure_ascii=False,
            )

            logger.info(f"Metrics saved to {results_dest}")
    else:
        # parallelization used, dont calculate metrics, it will be aggregated later
        with open(os.path.join(result_dir, f"results_{index}.json"), "w") as f:
            json.dump(
                {"y_pred": verdicts, "y_ref": statements},
                f,
                indent=4,
                ensure_ascii=False,
            )

            logger.info(
                f"Results saved to {os.path.join(result_dir, f'results_{index}.json')}"
            )


def setup_logging(log_dir="logs/baseline"):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = (
        f"{log_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stream_handler)


def load_config():
    """
    Loads configuration from config.yaml file and optionally replaces some values with command line arguments.
    """

    config = CONFIG

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--statements", type=str, default=config["StatementsPath"], help="Path to statements file.",)
    parser.add_argument("-n", "--name", type=str, default=config["Name"], help="Name of the evaluation.")
    parser.add_argument("-o", "--out-folder", type=str, default=config["ResultsFolder"], help="Path to output folder.",)
    parser.add_argument("ModelName", help="Name of the model to evaluate.", type=str, default=config["ModelName"], nargs="?",)
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically overwrite existing files.",)
    parser.add_argument("-c", "--crossval", action="store_true", help="Perform cross-validation.")
    parser.add_argument("-i", "--index", type=int, help="Index for parallelization.", default=0)
    parser.add_argument("-m", "--max", type=int, default = 0, help="Maximum number of parallel processes.")
    parser.add_argument("-e", "--explanation", action="store_true", help="Require explanation in the model output",)
    parser.add_argument("-E", "--evidence", type=str, default=config["EvidenceDir"], help="Evidence folder path")
    parser.add_argument("-p", "--prompt-config", type=str, default=config["PromptConfigPath"], help="Prompt config file path")
    parser.add_argument("--example-count", type=int, default=0, help="Number of examples for each label to use")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("-t", "--test-portion", type=float, default=0.2, help="Portion of dataset to sample for testing")
    parser.add_argument("--allowed-labels", nargs="+", default=['pravda','nepravda'], help="Labels that should be included in the evaluation")

    args = parser.parse_args()

    config["StatementsPath"] = args.statements
    config["Name"] = args.name
    config["ResultsFolder"] = args.out_folder
    config["ModelName"] = args.ModelName
    config["Yes"] = args.yes
    config["CrossValidation"] = args.crossval
    config["Index"] = args.index
    config["Max"] = args.max
    config["WithExplanation"] = args.explanation
    config["EvidenceDir"] = args.evidence
    config["PromptConfigPath"] = args.prompt_config
    config["ExampleCount"] = args.example_count
    config["BatchSize"] = args.batch_size
    config["TestPortion"] = args.test_portion
    config["AllowedLabels"] = args.allowed_labels

    if config["Index"] and not config["Max"]:
        parser.error("--index requires --max.")

    logger.info(f"Loaded configuration: {config}")

    return config


def check_file_overwrite(file_path):
    if os.path.exists(file_path):
        logger.warning(f"File {file_path} already exists.")
        user_input = input("Do you want to overwrite it? (y/n): ")

        if user_input.lower() != "y":
            logger.info("Exiting.")
            exit()
        else:
            logger.info("Overwriting file.")


if __name__ == "__main__":
    config = load_config()
    setup_logging(config["LogPath"])

    result_dir = os.path.join(
        config["ResultsFolder"], config["ModelName"].split("/")[-1]
    )

    os.makedirs(result_dir, exist_ok=True)

    logger.info(f"Starting evaluation of model {config['ModelName']}")

    statements = []
    with open(config["StatementsPath"], "r") as f:
        statements = json.load(f)

    # filter only statements with allowed labels
    filtered_statements = [stmt for stmt in statements if stmt['assessment'].lower() in config["AllowedLabels"]]

    if config["TestPortion"] < 1.0:
        _, test_statements = train_test_split(
            filtered_statements, test_size=config["TestPortion"], random_state=42, stratify=None
        )
    else:
        test_statements = filtered_statements

    # determine index range for parallelization
    lower_index = (
        (config["Index"] - 1) * len(test_statements) // config["Max"]
        if config["Index"] > 0
        else 0
    )
    upper_index = (
        config["Index"] * len(test_statements) // config["Max"] - 1
        if config["Index"] > 0
        else len(test_statements) - 1
    )

    logger.info(
        f"Parallelization index: {config['Index']}, lower index: {lower_index}, upper index: {upper_index}"
    )

    # calculate distribution of labels
    labels = [item["assessment"] for item in test_statements]
    label_counts = {label: labels.count(label) for label in set(labels)}
    logger.info(f"Label distribution in test set: {label_counts}\nTotal: {len(labels)}")


    # evaluate
    eval_dataset(
        model_id = config["ModelName"],
        statements = test_statements[lower_index:upper_index],
        result_dir = result_dir,
        evidence_dir = config["EvidenceDir"],
        prompt_config_path = config["PromptConfigPath"],
        index = config["Index"],
        with_explanation = config["WithExplanation"],
        example_count=config["ExampleCount"],
        batch_size = config["BatchSize"],
        allowed_labels = config["AllowedLabels"] 
    )
