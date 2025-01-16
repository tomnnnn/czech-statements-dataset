import openai
import envyaml
import json
import re
import asyncio
import itertools
import logging
import datetime
import transformers
import torch

from utils.baseline_utils import track_progress, find_by_id

CONFIG = envyaml.EnvYAML("config.yaml", strict=False)["default"]

SEM = asyncio.Semaphore(1)
STATEMENTS_FILE = CONFIG["StatementsPath"]

API_KEY = CONFIG["APIKey"]
API_BASE_URL = CONFIG["APIBaseURL"]
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



async def eval_statement(i, statement, model):
    # Rate limiter for API requests
    await asyncio.sleep(i * 60 / RPM)
    client = openai.AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    prompt = (
        statement["statement"]
        + " - "
        + statement["author_name"]
        + ", "
        + statement["date"]
    )

    logger.info(f"Evaluating statement {i}: {prompt}")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ZEROSHOT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )

    response = response.choices[0].message.content
    label = ""

    if response:
        first_line = response.split("\n")[0]
        label = re.findall(r"NEOVĚŘITELNÉ|PRAVDA|NEPRAVDA", first_line)
        label = label[0] if label else "NOLABEL"
    else:
        response = "NORESPONSE"

    statement["response"] = response
    statement["label"] = label
    statement["prompt"] = prompt

    logger.info(f"Statement {i} evaluated with label: {label}")
    return statement

async def eval_dataset(model, response_file, result_file):
    """
    Test chosen models accuracy of labeling statements in STATEMENTS_FILE with zero-shot prompt.
    The results are saved to response_file and result_file.
    """

    with open(STATEMENTS_FILE, "r") as f:
        statements = json.load(f)

    res_coros = [
        track_progress(
            eval_statement(i, statement, model), CNTER, len(statements), "statement"
        )
        for i, statement in enumerate(statements)
    ]
    responses = await asyncio.gather(*res_coros)

    with open(response_file, "w") as f:
        json.dump(responses, f, indent=4, ensure_ascii=False)

    calculate_accuracy(STATEMENTS_FILE, response_file, result_file)

    print(f"Written to LLM responses to {response_file}")
    print(f"Written to LLM test results to {result_file}")
    logger.info(f"Written to LLM responses to {response_file}")
    logger.info(f"Written to LLM test results to {result_file}")


def calculate_accuracy(reference_path, with_responses_path, output_path):
    reference = []
    responses = []

    incorrect = []
    correct_cnt = 0

    with open(reference_path) as f:
        reference = json.load(f)

    with open(with_responses_path) as f:
        responses = json.load(f)


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

    out_folder = (
        CONFIG["ResultsFolder"]
        if CONFIG["ResultsFolder"][-1] == "/"
        else CONFIG["ResultsFolder"] + "/"
    )

    for i, model in enumerate(CONFIG["ModelAPINames"]):
        CNTER = itertools.count(1)
        responses_file = out_folder + CONFIG["ModelName"][i] + "_responses" + ".json"
        result_file = out_folder + CONFIG["ModelName"][i] + "_results" + ".json"

        logger.info(f"Starting evaluation of model {model}")
        print(f"Starting evaluation of model {model}")

        asyncio.run(eval_dataset(model, responses_file, result_file))

        print("\n")
