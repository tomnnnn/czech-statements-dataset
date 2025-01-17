import os
import json
import re
import asyncio
import itertools
import logging
import datetime
from local_llm import Model
import config

from utils.baseline_utils import find_by_id

CONFIG = config.load_config("baseline_config.yaml")

SEM = asyncio.Semaphore(1)
STATEMENTS_FILE = CONFIG["StatementsPath"]

WITH_NOI = CONFIG["NOI"]
CNTER = itertools.count(1)
RPM = CONFIG["RPM"]

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_ZEROSHOT = f"""Tvým úkolem je ohodnotit zadaný výrok českého politika známkou PRAVDA|NEPRAVDA{"|NEOVĚŘITELNÉ" if WITH_NOI else ""}. {"Výrok je neověřitelný, pokud pro jeho ověření nemáš dostatek informací." if WITH_NOI else ""} Ověřuješ obsah daného výroku, ne jestli byl řečen daným člověkem. Ve vysvětlení důvodu hodnocení popiš proč tento fakt potvrzuješ nebo vyvracíš. Ve zdrojích uveď zdroje, na kterých byl tvůj úsudek postaven.

Formát odpovědi:
VERDIKT: <PRAVDA|NEPRAVDA|NEOVĚŘITELNÉ>
VYSVĚTLENÍ: <Vysvětlení důvodu hodnocení>
ZDROJE: <Zdroje, na kterých je hodnocení založeno>
"""

SYSTEM_PROMPT_ONESHOT = f"""Tvým úkolem je ohodnotit zadaný výrok českého politika známkou PRAVDA|NEPRAVDA{"|NEOVĚŘITELNÉ" if WITH_NOI else ""}. {"Výrok je neověřitelný, pokud pro jeho ověření nemáš dostatek informací." if WITH_NOI else ""} Ověřuješ obsah daného výroku, ne jestli byl řečen daným člověkem. Ve vysvětlení důvodu hodnocení popiš proč tento fakt potvrzuješ nebo vyvracíš. Ve zdrojích uveď zdroje, na kterých byl tvůj úsudek postaven.

Formát odpovědi:
VERDIKT: <PRAVDA|NEPRAVDA|NEOVĚŘITELNÉ>
VYSVĚTLENÍ: <Vysvětlení důvodu hodnocení>
ZDROJE: <Zdroje, na kterých je hodnocení založeno>
"""


def process_response(response, prompt, id):
    label = "NOLABEL"

    if response:
        first_line = response.split("\n")[0]
        label = re.findall(r"NEOVĚŘITELNÉ|PRAVDA|NEPRAVDA", first_line)
        label = label[0] if label else "NOLABEL"
    else:
        response = "NORESPONSE"

    statement = {
        "id": id,
        "response": response,
        "label": label,
        "prompt": prompt,
    }

    return statement

async def eval_dataset(model_id, response_file, result_file):
    """
    Test chosen models accuracy of labeling statements in STATEMENTS_FILE with zero-shot prompt.
    The results are saved to response_file and result_file.

    Args:
    model_id (str): Model name.
    response_file (str): Path to output file for responses.
    """

    with open(STATEMENTS_FILE, "r") as f:
        statements = json.load(f)

    model = Model(model_id)
    model.set_system_prompt(SYSTEM_PROMPT_ZEROSHOT)

    prompts = [
        statement["statement"]
        + " - "
        + statement["author"]
        + ", "
        + statement["date"]
        for statement in statements]

    responses = model(prompts)
    processed = []
    for raw, prompt, statement in zip(responses, prompts, statements):
        processed += [process_response(raw, prompt, statement["id"])]

    with open(response_file, "w") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)

    calculate_accuracy(STATEMENTS_FILE, processed, result_file)

    print(f"Written to LLM responses to {response_file}")
    print(f"Written to LLM test results to {result_file}")
    logger.info(f"Written to LLM responses to {response_file}")
    logger.info(f"Written to LLM test results to {result_file}")


def calculate_accuracy(reference_path, responses, output_path):
    """
    Calculate accuracy of responses in with_responses_path compared to reference in reference_path.
    The results are saved to output_path.

    Args:
    reference_path (str): Path to reference file. The statements in this files define the correct labels.
    responses (List): List of responses to evaluate.
    output_path (str): Path to output file.
    """
    reference = []
    incorrect = []
    correct_cnt = 0

    with open(reference_path) as f:
        reference = json.load(f)

    for res in responses:
        ref = find_by_id(res["id"], reference)
        if ref is None:
            f.write(f"ERROR: Could not find reference for id {res['id']}\n")
            logger.error(f"Could not find reference for id {res['id']}")
            continue

        ref_label = ref["assessment"].lower().strip()
        res_label = res["label"].replace('.','').replace('\n', ' ').split(' ')[0].lower().strip()

        if ref_label == res_label:
            correct_cnt += 1
        else:
            incorrect.append({
                "id": res["id"],
                "statement": ref["statement"],
                "reference_label": ref_label,
                "label": res_label,
                "response": res["response"]
            })

    with open(output_path, "w") as f:
        json.dump({
            "accuracy": correct_cnt / len(responses),
            "correct": correct_cnt,
            "total": len(responses),
            "incorrect": incorrect,
        }, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # set up logging
    logging.basicConfig(filename=f"logs/baseline/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log", level=logging.INFO)
    os.makedirs("logs/baseline", exist_ok=True)
    os.makedirs(CONFIG["ResultsFolder"], exist_ok=True)

    for i, model in enumerate(CONFIG["ModelName"]):
        CNTER = itertools.count(1)
        responses_file = f"{CONFIG['ResultsFolder']}/{model.split('/')[-1]}_responses.json"
        result_file = f"{CONFIG['ResultsFolder']}/{model.split('/')[-1]}_results.json"

        logger.info(f"Starting evaluation of model {model}")
        print(f"Starting evaluation of model {model}")

        asyncio.run(eval_dataset(model, responses_file, result_file))

        print("\n")
