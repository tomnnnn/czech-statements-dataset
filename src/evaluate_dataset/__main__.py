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

from dataset_manager import DemagogDataset
from .config import CONFIG
from .llm_apis import llm_api_factory, llm_api_dict
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


def prompt_builder(statements, dataset: DemagogDataset, max_evidence_len=1500, relevancy_threshold=0.5, relevant_paragraph=False):
    """
    Build prompts for given statements. Optionally include evidence from evidence_dir.

    Args:
    statements (List): List of statements.
    evidence_dir (str): Path to folder with evidence files.
    max_content_len (int): Maximum length of evidence content to include in the prompt.
    """
    prompts = []
    for statement in statements:
        statement_str = ( f"{statement['statement']} - {statement['author']}, {statement['date']}")

        evidence = dataset.get_evidence(statement['id'])[:10]
        evidence = [e for e in evidence if float(e['relevancy']) >= float(relevancy_threshold)]

        if not evidence:
            logger.warning(f"No evidence found for statement {statement['id']}.")
            prompts.append(statement_str + "\nPodpůrné dokumenty: Není dostupný žádný podpůrný dokument.")
            continue

        evidence_contents = [
            e['content'][:max_evidence_len] if not relevant_paragraph else e['description'] + '\n' + e['relevant_paragraph'] 
            for e in evidence
        ]

        truncated = [
            {
                'Titulek': e['title'],
                'Text': text,
                'Zdroj': tldextract.extract(e['url']).domain + "." + tldextract.extract(e['url']).suffix if e['url'] else '',
            }
            for e,text in zip(evidence, evidence_contents)
        ]

        evidence_str = json.dumps(truncated, indent=2, ensure_ascii=False)

        prompts.append(f"{statement_str}\nPodpůrné dokumenty: {evidence_str}")

    return prompts


def split_and_build_examples(
    statements,
    dataset,
    seed=None,
    allowed_labels=["pravda", "nepravda", "neověřitelné"],
    include_explanation=False,
    count_for_each_label=1,
    relevancy_threshold=0.5,
    relevant_paragraph=False
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
        filtered_statements = [ stmt for stmt in statements if stmt["label"].lower() == label ]

        if len(filtered_statements) < count_for_each_label:
            raise ValueError(f"Not enough statements for label '{label}'. Required: {count_for_each_label}, Found: {len(filtered_statements)}")

        # pseudo-randomly sample
        shuffled_items = sorted(filtered_statements, key=lambda _: random.random())  # Shuffle while keeping order fixed
        example_statements.extend(shuffled_items[:count_for_each_label])

    test_statements = [stmt for stmt in statements if stmt not in example_statements]
    inputs = prompt_builder(example_statements,dataset, relevancy_threshold=relevancy_threshold, relevant_paragraph=relevant_paragraph)

    explanations = [
        (
            f"Úvaha: {stmt['explanation'] if stmt['explanation_brief'] else stmt['explanation']}\n"
            if include_explanation
            else ""
        )
        for stmt in example_statements
    ]

    examples = [
        {"input": input_text, "output": explanation + "Hodnocení: " + stmt["label"].lower()}
        for input_text, stmt, explanation in zip(inputs, example_statements, explanations)
    ]

    return examples, test_statements


def eval_dataset(
    config,
    dataset,
    statements,
    example_statements=None,
):
    """
    Test chosen models accuracy of labeling given statements with zero-shot prompt.
    Results are saved to result_dir.

    Args:
    config (dict): Configuration dictionary.
    dataset (DemagogDataset): Dataset object.
    statements (List): List of statements to evaluate.
    example_statements (List): List of example statements to use for few-shot prompting. If None, statements are used.
    """

    result_dir = os.path.join(config["ResultsFolder"], config["ModelName"].split("/")[-1])
    os.makedirs(result_dir, exist_ok=True)

    model = llm_api_factory(config["ModelAPI"], config["ModelName"], filename=config["ModelFile"])
    system_prompt, _ = load_prompt_config(config["PromptConfigPath"])

    # split statements into examples and test statements
    examples, test_statements = split_and_build_examples(
        example_statements if example_statements else statements,
        seed=42,
        include_explanation=config["WithExplanation"],
        count_for_each_label=config["ExampleCount"],
        allowed_labels=config["AllowedLabels"],
        dataset=dataset,
        relevancy_threshold=config["RelevancyThreshold"],
        relevant_paragraph=config["RelevantParagraph"]
    )
    logger.info(f"Loaded system prompt: {system_prompt}")
    logger.info(f"Loaded {len(examples)} examples.")

    # debug
    with open(os.path.join(result_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)

    model.set_system_prompt(system_prompt)
    model.set_generation_prompt("Úvaha: " if config["WithExplanation"] else "Hodnocení: ")
    model.set_examples(examples)

    prompts = prompt_builder(test_statements, dataset, relevancy_threshold=config["RelevancyThreshold"], relevant_paragraph=config["RelevantParagraph"])

    # test statements info
    logger.info(f"Evaluating {len(prompts)} statements.")
    labels = [item["label"] for item in test_statements]
    label_counts = {label: labels.count(label) for label in set(labels)}
    logger.info(f"Label distribution in test set: {label_counts}")

    # generate responses
    logger.info(f"Generating responses for {len(prompts)} prompts.")
    responses = model(prompts, config["BatchSize"], max_new_tokens=3000 if config["WithExplanation"] else 25)

    verdicts = [
        {
            "id": statement["id"],
            "statement": statement["statement"],
            "author": statement["author"],
            "date": statement["date"],
            "response": response,
            "label": extract_label(response, config["AllowedLabels"]),
        }
        for statement, response in zip(test_statements, responses)
    ]

    # save responses
    responses_dest = os.path.join(result_dir, f"responses_{config['Index']}.json")
    with open(os.path.join(responses_dest), "w") as f:
        json.dump(verdicts, f, indent=4, ensure_ascii=False)
        logger.info(f"Responses saved to {responses_dest}")

    # save prompts
    prompts_dest = os.path.join(result_dir, f"prompts_{config['Index']}.json")
    with open(prompts_dest, "w") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
        logger.info(f"Prompts saved to {prompts_dest}")

    if config["Index"] == 0:
        # calculate and save metrics
        results_dest = os.path.join(result_dir, f"metrics.json")
        with open(results_dest, "w") as f:
            json.dump(
                calculate_metrics(test_statements, verdicts, config["AllowedLabels"]),
                f,
                indent=4,
                ensure_ascii=False,
            )

            logger.info(f"Metrics saved to {results_dest}")
    else:
        # parallelization used, dont calculate metrics, it will be aggregated later
        file_path = os.path.join(result_dir, f"results_{config['Index']}.json")
        with open(file_path, "w") as f:
            json.dump(
                {"y_pred": verdicts, "y_ref": test_statements},
                f,
                indent=4,
                ensure_ascii=False,
            )

            logger.info(
                f"Results saved to {file_path}."
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
    parser.add_argument("-o", "--out-folder", type=str, default=config["ResultsFolder"], help="Path to output folder.",)
    parser.add_argument("-i", "--index", type=int, default=0, help="Index for parallelization.")
    parser.add_argument("-m", "--max", type=int, default = 1, help="Maximum number of parallel processes.")
    parser.add_argument("-e", "--explanation", action="store_true", help="Require explanation in the model output",)
    parser.add_argument("-p", "--prompt-config", type=str, default=config["PromptConfigPath"], help="Prompt config file path")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("-r", "--relevancy-threshold", type=int, default=1, help="Sets minimum relevancy score needed to include evidence in listed evidence documents")
    parser.add_argument("-P", "--relevant-paragraph", action="store_true", help="Include only relevant paragraph from evidence")
    parser.add_argument("-t", "--test-portion", type=float, default=None, help="Portion of dataset to sample for testing. If number N >= 1 is supplied, exactly N statements will be sampled for testing.")
    parser.add_argument("-d", "--dataset-path", type=str, default=config.get("DatasetPath", ""), help="Path to dataset file. Accepts [sqlite] database file.")
    parser.add_argument("-a", "--allowed-labels", nargs="+", default=['pravda','nepravda'], help="Labels that should be included in the evaluation")
    parser.add_argument("-A", "--model-api", default="transformers", help="LLM API to use. Avaialble APIs: " + ", ".join(llm_api_dict.keys()))
    parser.add_argument("-c", "--example-count", type=int, default=0, help="Number of examples for each label to use")
    parser.add_argument("-l", "--log-path", type=str, default=config["LogPath"], help="Path to log file.")
    parser.add_argument("-E", "--evidence-source", type=str, default="demagog", help="Source of evidence data, used to determine evidence table in dataset database.")
    parser.add_argument("--model-file", help="Optional path to model file if needed.", type=str, default=None, nargs="?")
    parser.add_argument("ModelName", help="Name of the model to evaluate.", type=str, default=config["ModelName"], nargs="?",)
    parser.add_argument("--stratify", action="store_true", help="Stratify test set by labels.")


    args = parser.parse_args()

    config["ResultsFolder"] = args.out_folder
    config["ModelName"] = args.ModelName
    config["Index"] = args.index
    config["Max"] = args.max
    config["WithExplanation"] = args.explanation
    config["PromptConfigPath"] = args.prompt_config
    config["ExampleCount"] = args.example_count
    config["BatchSize"] = args.batch_size
    config["TestPortion"] = args.test_portion
    config["AllowedLabels"] = args.allowed_labels
    config["ModelAPI"] = args.model_api
    config["DatasetPath"] = args.dataset_path
    config["LogPath"] = args.log_path
    config["EvidenceSource"] = args.evidence_source
    config["ModelFile"] = args.model_file
    config["Stratify"] = args.stratify
    config["RelevancyThreshold"] = args.relevancy_threshold
    config["RelevantParagraph"] = args.relevant_paragraph


    if config["Index"] and not config["Max"]:
        parser.error("--index requires --max.")

    logger.info(f"Loaded configuration: {config}")

    return config


if __name__ == "__main__":
    config = load_config()
    setup_logging(config["LogPath"])
    dataset = DemagogDataset(config["DatasetPath"], config["EvidenceSource"], readonly=True)
    random.seed(42)

    logger.info(f"Starting evaluation of model {config['ModelName']} via {config['ModelAPI']} API")

    # get all statements with allowed labels and at least 5 articles
    statements = dataset.get_all_statements(config["AllowedLabels"], 0)
    labels = [item["label"] for item in statements]

    logger.info(f"Loaded {len(statements)} statements with allowed labels and at least 5 evidence articles.")

    if config["TestPortion"] and config["TestPortion"] != 1:
        if config["TestPortion"] < 1.0:
            _, test_statements = train_test_split(statements, test_size=config["TestPortion"], random_state=42, stratify=labels)
            logger.info(f"Split dataset into {len(test_statements)} test statements ({config['TestPortion']*100} %).")
        else:
            # randomly sample N statements
            shuffled_statements = sorted(statements, key=lambda _: random.random())  # Shuffle while keeping order fixed

            if config["Stratify"]:
                by_label = {
                    label: [stmt for stmt in shuffled_statements if stmt["label"].lower() == label]
                    for label in config["AllowedLabels"]
                }
                test_statements = []
                for label in config["AllowedLabels"]:
                    test_statements.extend(by_label[label][:int(config["TestPortion"]/len(config["AllowedLabels"]))])
            else:
                test_statements = shuffled_statements[:config["TestPortion"]]
            
    else:
        test_statements = statements

    # determine index range for parallelization
    lower_index = max(0, (config["Index"] - 1) * len(test_statements) // config["Max"])
    upper_index = len(test_statements) - 1 if config["Index"] == 0 else config["Index"] * len(test_statements) // config["Max"] - 1
    logger.info(f"Parallelization index: {config['Index']}, lower index: {lower_index}, upper index: {upper_index}")

    # evaluate
    eval_dataset(
        config=config,
        dataset=dataset,
        statements=test_statements[lower_index:upper_index + 1],
    )

    logger.info("Evaluation finished.")
