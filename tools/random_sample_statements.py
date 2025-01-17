"""
Randomly samples NUM_SAMPLES statements from a dataset of statements and saves them to a new JSON file.
"""

import random
import json
import datetime
import locale

locale.setlocale(locale.LC_TIME, "cs_CZ.UTF-8")

NUM_SAMPLES = 1000
STATEMENTS_FILE = "../datasets/with_evidence/bing/statements.json"
OUTPUT_FILE = "../datasets/samples/random_sample_statements_1000.json"
ALLOWED_LABELS = ["pravda", "nepravda", "neověřitelné"]

def find_by_id(id, statements):
    for statement in statements:
        if statement["id"] == id and statement["assessment"].lower() in ALLOWED_LABELS:
            parsed_date = datetime.datetime.strptime(statement["date"], "%Y-%m-%d")
            if(parsed_date > datetime.datetime(2023, 12, 31)):
                # include only statements after 31.12.2023
                return find_by_id(random.randint(RANGE[0], RANGE[1]), statements)

            return statement

    # if no statement found, generate new random ID
    return find_by_id(random.randint(RANGE[0], RANGE[1]), statements)

statements = {}
with open(STATEMENTS_FILE, "r") as f:
    statements = json.load(f)

# get highest id
ids = [statement["id"] for statement in statements]
ids.sort()

RANGE = (0, ids[-1])

# split dataset by labels
true_statements = [statement for statement in statements if statement["assessment"].lower() == "pravda"]
false_statements = [statement for statement in statements if statement["assessment"].lower() == "nepravda"]
unverifiable_statements = [statement for statement in statements if statement["assessment"].lower() == "neověřitelné"]

# sample evenly from each label
sample_ids = []
sample_statements = []

for i in range(NUM_SAMPLES):
    if i % 3 == 0:
        sample_ids.append(random.choice(true_statements)["id"])
    elif i % 3 == 1:
        sample_ids.append(random.choice(false_statements)["id"])
    else:
        sample_ids.append(random.choice(unverifiable_statements)["id"])

for id in sample_ids:
    sample_statements.append(find_by_id(id, statements))

# calculate distribution
true_count = len([statement for statement in sample_statements if statement["assessment"].lower() == "pravda"])
false_count = len([statement for statement in sample_statements if statement["assessment"].lower() == "nepravda"])
unverifiable_count = len([statement for statement in sample_statements if statement["assessment"].lower() == "neověřitelné"])

# save to file
with open(OUTPUT_FILE, "w") as f:
    sample = {
        "count": NUM_SAMPLES,
        "true": true_count,
        "false": false_count,
        "unverifiable": unverifiable_count,
        "statements": sample_statements
    }
    json.dump(sample, f, indent=4, ensure_ascii=False)

print(f"Created random sample with following distribution: \n{true_count} true\n{false_count} false\n{unverifiable_count} unverifiable\nLocation: {OUTPUT_FILE}")

