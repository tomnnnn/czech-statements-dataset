import json
import logging
import os
import random
import re
import tldextract

import yaml
from sklearn.model_selection import train_test_split

from dataset_manager import DemagogDataset
from .config import load_config
from .llm_apis import llm_api_factory
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
        labels = re.findall(labels_regex, response.lower())
        return labels[-1] if labels else "nolabel"

    return "nolabel"

def prompt_builder(statements, dataset: DemagogDataset, max_evidence_len=1500, relevancy_threshold=0.5, relevant_paragraph=False):
    """
    Build prompts for given statements. Optionally include evidence from evidence_dir.

    Args:
    statements (List): List of statements.
    evidence_dir (str): Path to folder with evidence files.
    max_content_len (int): maximum length of evidence content to include in the prompt.
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

    result_dir = os.path.join(config.out_folder, config.model_name.split("/")[-1])
    os.makedirs(result_dir, exist_ok=True)

    model = llm_api_factory(config.model_api, config.model_name, filename=config.model_file)
    system_prompt, _ = load_prompt_config(config.prompt_config)

    # split statements into examples and test statements
    examples, test_statements = split_and_build_examples(
        example_statements if example_statements else statements,
        seed=42,
        include_explanation=config.with_explanation,
        count_for_each_label=config.example_count,
        allowed_labels=config.allowed_labels,
        dataset=dataset,
        relevancy_threshold=config.relevancy_threshold,
        relevant_paragraph=config.relevant_paragraph
    )
    logger.info(f"Loaded system prompt: {system_prompt}")
    logger.info(f"Loaded {len(examples)} examples.")

    # debug
    with open(os.path.join(result_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)

    model.set_system_prompt(system_prompt)
    model.set_generation_prompt("Úvaha: " if config.with_explanation else "Hodnocení: ")
    model.set_examples(examples)

    prompts = prompt_builder(test_statements, dataset, relevancy_threshold=config.relevancy_threshold, relevant_paragraph=config.relevant_paragraph)

    # test statements info
    logger.info(f"Evaluating {len(prompts)} statements.")
    labels = [item["label"] for item in test_statements]
    label_counts = {label: labels.count(label) for label in set(labels)}
    logger.info(f"Label distribution in test set: {label_counts}")

    # generate responses
    logger.info(f"Generating responses for {len(prompts)} prompts.")
    responses = model(prompts, config.batch_size, max_new_tokens=3000 if config.with_explanation else 25)

    verdicts = [
        {
            "id": statement["id"],
            "statement": statement["statement"],
            "author": statement["author"],
            "date": statement["date"],
            "response": response,
            "label": extract_label(response, config.allowed_labels),
        }
        for statement, response in zip(test_statements, responses)
    ]

    # save responses
    responses_dest = os.path.join(result_dir, f"responses_{config.index}.json")
    with open(os.path.join(responses_dest), "w") as f:
        json.dump(verdicts, f, indent=4, ensure_ascii=False)
        logger.info(f"Responses saved to {responses_dest}")

    # save prompts
    prompts_dest = os.path.join(result_dir, f"prompts_{config.index}.json")
    with open(prompts_dest, "w") as f:
        json.dump(prompts, f, indent=4, ensure_ascii=False)
        logger.info(f"Prompts saved to {prompts_dest}")

    if config.index == 0:
        # calculate and save metrics
        results_dest = os.path.join(result_dir, f"metrics.json")
        with open(results_dest, "w") as f:
            json.dump(
                calculate_metrics(test_statements, verdicts, config.allowed_labels),
                f,
                indent=4,
                ensure_ascii=False,
            )

            logger.info(f"Metrics saved to {results_dest}")
    else:
        # parallelization used, dont calculate metrics, it will be aggregated later
        file_path = os.path.join(result_dir, f"results_{config.index}.json")
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


def sample_dataset(dataset, config):
    random.seed(42)
    statements = dataset.get_all_statements(config.allowed_labels, config.min_evidence_count)

    logger.info(f"Loaded {len(statements)} statements with allowed labels and at least {config.min_evidence_count} evidence articles.")

    labels = [item["label"] for item in statements]

    if config.test_portion and config.test_portion != 1:
        if config.test_portion < 1.0:
            _, test_statements = train_test_split(statements, test_size=config.test_portion, random_state=42, stratify=labels)

            logger.info(f"Split dataset into {len(test_statements)} test statements ({config.test_portion*100} %).")
        else:
            # randomly sample N statements
            shuffled_statements = sorted(statements, key=lambda _: random.random())

            if config.stratify:
                by_label = {
                    label: [stmt for stmt in shuffled_statements if stmt["label"].lower() == label]
                    for label in config.allowed_labels
                }
                test_statements = []
                for label in config.allowed_labels:
                    test_statements.extend(by_label[label][:int(config.test_portion/len(config.allowed_labels))])
            else:
                test_statements = shuffled_statements[:int(config.test_portion)]
            
    else:
        test_statements = statements

    return test_statements
    

def cut_for_parallelization(config, statements):
    """
    Determines range of statements to evaluate based on parallelization index and max processes.

    Args:
    config (dict): Configuration dictionary.
    statements (List): List of statements to evaluate.

    Returns:
    List: Subset of statements to evaluate.
    """
    lower_index = max(0, (config.index - 1) * len(statements) // config.max)
    upper_index = len(statements) - 1 if config.index == 0 else config.index * len(statements) // config.max - 1
    logger.info(f"Parallelization index: {config.index}, lower index: {lower_index}, upper index: {upper_index}")

    return statements[lower_index:upper_index + 1]

if __name__ == "__main__":
    config = load_config()

    dataset = DemagogDataset(config.dataset_path, config.evidence_source, readonly=True)
    statements = sample_dataset(dataset, config)
    statements = cut_for_parallelization(config, statements)

    logger.info(f"Starting evaluation of model {config.model_name} via {config.model_api} API")

    eval_dataset(
        config=config,
        dataset=dataset,
        statements=statements,
    )

    logger.info("Evaluation finished.")
