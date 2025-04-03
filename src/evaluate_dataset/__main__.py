import json
from typing import Dict
from bs4 import BeautifulSoup
import logging
import operator
import os
import random
import re
from dataclasses import asdict
from typing import List

import tldextract
import yaml
from dataset_manager.orm import *
from sklearn.model_selection import train_test_split
from sqlalchemy import func
from sqlalchemy.orm import Session
from evidence_retriever.hop_retriever import HopRetriever

from .config import Config, load_config
from .llm_apis import llm_api_factory
from .utils import calculate_metrics

logger = logging.getLogger(__name__)

ops = {
    "eq": operator.eq,
    "ge": operator.ge,
    "le": operator.le,
}

def load_prompt_config(config_location) -> tuple:
    """
    Load system prompt and generation prompt from configuration file. If the file does not exist, return empty values
    """

    if not os.path.exists(config_location):
        logger.warning(
            f"Prompt configuration file {config_location} not found. Running without system prompt and no examples."
        )
        return "", []

    with open(config_location, "r") as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)

    return prompts.get("system_prompt", ""), prompts.get("generation_prompt", "") 


def extract_label(response, allowed_labels):
    if response:
        labels_regex = "|".join(allowed_labels)
        labels = re.findall(labels_regex, response.lower())
        return labels[-1] if labels else "nolabel"

    return "nolabel"


def html_to_text(html):
    """
    Convert HTML content to plain text
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Decompose all <img> and <figure> tags in a single loop
    for tag in soup.find_all(['img', 'figure']):
        tag.decompose()

    # Remove <p> tags containing 'cookie' in a single pass
    for tag in soup.find_all('p'):
        if 'cookie' in tag.text.lower():
            tag.decompose()

    # Collect text efficiently using a generator
    text = '\n'.join(
        tag.text.strip()
        for tag in soup.find_all('p')
        if len(tag.text.strip()) > 100
    )

    return text


def format_statement(statement) -> str:
    """Formats the statement string."""
    return f"{statement.statement} - {statement.author}, {statement.date}"


def extract_evidence(statement, config, max_num_articles=10) -> List:
    """Extracts evidence based on config settings."""
    if config.relevant_paragraph:
        return statement.segments
    return [
        Article(
            id=e.id, 
            title=e.title, 
            url=e.url, 
            content=html_to_text(e.content) if config.html_article else e.content
        ) 
        for e in statement.articles[:max_num_articles]
    ]


def format_evidence(evidence, config) -> List[Dict[str, str]]:
    """Formats and truncates evidence content for JSON output."""
    evidence_dicts = []
    
    for e in evidence:
        url = e.url if not config.relevant_paragraph else e.article.url
        source = f"{tldextract.extract(str(url)).domain}.{tldextract.extract(str(url)).suffix}" or ""
        title = e.title if not config.relevant_paragraph else e.article.title
        content = e.text if config.relevant_paragraph else e.content[:3000]
        
        evidence_dicts.append({
            'Titulek': title,
            'Text': content,
            'Zdroj': source,
        })
    
    return evidence_dicts

def prompt_builder(statements: List[Statement], dataset: Session, config: Config) -> List[str]:
    """
    Build prompts for given statements. Optionally include evidence from evidence_dir.
    """
    prompts = []

    for statement in statements:
        statement_str = format_statement(statement)
        evidence = extract_evidence(statement, config)

        if not evidence:
            logger.warning(f"No evidence found for statement {statement.id}.")
            prompts.append(f"{statement_str}\nPodpůrné dokumenty: Není dostupný žádný podpůrný dokument.")
            continue

        truncated_evidence = format_evidence(evidence, config)
        evidence_str = json.dumps(truncated_evidence, ensure_ascii=False)

        prompts.append(f"{statement_str}\nPodpůrné dokumenty: {evidence_str}")

    return prompts

def split_and_build_examples(
    statements: List[Statement],
    dataset: Session,
    config: Config,
    generation_prompt: str = "",
    seed=None,
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
    for label in config.allowed_labels:
        filtered_statements = [ stmt for stmt in statements if stmt.label.lower() == label ]

        if len(filtered_statements) < config.example_count:
            raise ValueError(f"Not enough statements for label '{label}'. Required: {config.example_count}, Found: {len(filtered_statements)}")

        # pseudo-randomly sample
        shuffled_items = sorted(filtered_statements, key=lambda _: random.random())  # Shuffle while keeping order fixed
        example_statements.extend(shuffled_items[:config.example_count])

    test_statements = [stmt for stmt in statements if stmt not in example_statements]
    inputs = prompt_builder(example_statements,dataset, config)

    explanations = [
        (
            f"Úvaha: {stmt.explanation or stmt.explanation_brief}\n"
            if config.with_explanation
            else ""
        )
        for stmt in example_statements
    ]

    examples = [
        {"input": input_text, "output": explanation + generation_prompt + stmt.label.lower()}
        for input_text, stmt, explanation in zip(inputs, example_statements, explanations)
    ]

    return examples, test_statements


def eval_dataset(
    config: Config,
    dataset: Session,
    statements: List[Statement],
    example_statements=None,
):
    """
    Test chosen models accuracy of labeling given statements with zero-shot prompt.
    Results are saved to result_dir.

    Args:
    config (dict): Configuration dictionary.
    dataset (): Dataset SQL session object.
    statements (List): List of statements to evaluate.
    example_statements (List): List of example statements to use for few-shot prompting. If None, statements are used.
    """

    result_dir = os.path.join(config.out_folder, config.model_name.split("/")[-1] if config.model_name else config.model_file.split(".")[0])
    os.makedirs(result_dir, exist_ok=True)

    model = llm_api_factory(config.model_api, config.model_name, **asdict(config))
    system_prompt, generation_prompt = load_prompt_config(config.prompt_config)

    # split statements into examples and test statements
    examples, test_statements = split_and_build_examples(
        example_statements if example_statements else statements,
        dataset,
        config,
        generation_prompt
    )
    logger.info(f"Loaded system prompt: {system_prompt}")
    logger.info(f"Loaded {len(examples)} examples.")

    # debug
    with open(os.path.join(result_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)

    model.set_system_prompt(system_prompt)

    print("Generation Prompt:", generation_prompt)
    if generation_prompt: 
        model.set_generation_prompt(generation_prompt)

    model.set_examples(examples)

    prompts = prompt_builder(test_statements, dataset, config)

    # test statements info
    logger.info(f"Evaluating {len(prompts)} statements.")
    labels = [item.label for item in test_statements]
    label_counts = {label: labels.count(label) for label in set(labels)}
    logger.info(f"Label distribution in test set: {label_counts}")

    # generate responses
    logger.info(f"Generating responses for {len(prompts)} prompts.")
    max_new_tokens = config.max_tokens if config.max_tokens else (3000 if config.with_explanation else 25)
    responses = model(prompts, config.batch_size, max_new_tokens = max_new_tokens)

    verdicts = [
        {
            "id": statement.id,
            "statement": statement.statement,
            "author": statement.author,
            "date": statement.date,
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

    test_statements_dict = rows2dict(test_statements)

    if config.index == 0:
        # calculate and save metrics
        results_dest = os.path.join(result_dir, f"metrics.json")
 
        with open(results_dest, "w") as f:
            json.dump(
                calculate_metrics(test_statements_dict, verdicts, config.allowed_labels),
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
                {"y_pred": verdicts, "y_ref": test_statements_dict},
                f,
                indent=4,
                ensure_ascii=False,
            )

            logger.info(
                f"Results saved to {file_path}."
            )


def sample_dataset(dataset, config: Config):
    random.seed(42)
    statements = (dataset.query(Statement)
        .filter(func.lower(Statement.label).in_(config.allowed_labels))
        .join(ArticleRelevance)
        .group_by(Statement.id)
        .having(func.count(ArticleRelevance.article_id) >= config.min_evidence_count)
        .all()
      )

    logger.info(f"Loaded {len(statements)} statements with allowed labels and at least {config.min_evidence_count} evidence articles.")

    labels = [item.label for item in statements]

    if config.test_portion and config.test_portion != 1:
        if config.test_portion < 1.0:
            _, test_statements = train_test_split(statements, test_size=config.test_portion, random_state=42, stratify=labels)

            logger.info(f"Split dataset into {len(test_statements)} test statements ({config.test_portion*100} %).")
        else:
            # randomly sample N statements
            shuffled_statements = sorted(statements, key=lambda _: random.random())

            if config.stratify:
                by_label = {
                    label: [stmt for stmt in shuffled_statements if stmt.label.lower() == label]
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
    dataset = init_db(config.dataset_path)
    statements = sample_dataset(dataset, config)
    statements = cut_for_parallelization(config, statements)

    logger.info(f"Starting evaluation of model {config.model_name} via {config.model_api} API")

    eval_dataset(
        config=config,
        dataset=dataset,
        statements=statements,
    )

    logger.info("Evaluation finished.")
