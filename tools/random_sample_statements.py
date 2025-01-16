"""
Randomly samples NUM_SAMPLES statements from a dataset of statements and saves them to a new JSON file.
"""

import random
import json
import datetime
import locale

locale.setlocale(locale.LC_TIME, "cs_CZ.UTF-8")

RANGE = [1,10000]
NUM_SAMPLES = 100
STATEMENTS_FILE = "../datasets/without_evidence/statements_2024.json"
OUTPUT_FILE = "../datasets/samples/statements_after_9_2024_sample.json"
INCLUDE_NOI = True

def find_by_id(id, statements):
    for statement in statements:
        if statement["id"] == id:
            if (not INCLUDE_NOI and statement["assessment"] == "Neověřitelné") or statement["assessment"] == "Zavádějící":
                return find_by_id(random.randint(RANGE[0], RANGE[1]), statements)

            parsed_date = datetime.datetime.strptime(statement['date'], "%YYY-%mm-%dd")
            if(parsed_date < datetime.datetime(2024, 9, 30)):
                # include only statements after 30.9.2024
                return find_by_id(random.randint(RANGE[0], RANGE[1]), statements)

            return statement

    # if no statement found, generate new random ID
    return find_by_id(random.randint(RANGE[0], RANGE[1]), statements)

statements = {}
with open(STATEMENTS_FILE, "r") as f:
    statements = json.load(f)

sample_ids = random.sample(range(RANGE[0], RANGE[1]), NUM_SAMPLES)
sample_statements = [find_by_id(id, statements) for id in sample_ids]

with open(OUTPUT_FILE, "w") as f:
    json.dump(sample_statements, f, indent=4, ensure_ascii=False)

print(f"Random sample of {NUM_SAMPLES} statements saved to {OUTPUT_FILE}.")
