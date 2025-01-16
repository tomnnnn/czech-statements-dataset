import json
import os
import csv

CSV_PATH = "../quality_check/criteria.csv"
CONTEXT_PATH = "../datasets/criteria/context/"
OUTPUT_PATH = "filled.csv"

# Load the data
with open(CSV_PATH) as table:
    reader = csv.DictReader(table, delimiter=";")
    if not reader.fieldnames:
        raise ValueError("No header")

    data = list(reader)

    with open(OUTPUT_PATH, "w+") as new:
        writer = csv.DictWriter(new, fieldnames=reader.fieldnames) 
        writer.writeheader()

        for row in data:
            if not row["id"]:
                print("No id")
                print(row)
                continue

            if os.path.exists(CONTEXT_PATH + row["id"] + ".json"):
                with open(CONTEXT_PATH + row["id"] + ".json") as file:
                    context = json.load(file)
                    row["# of articles"] = len(context)
                    writer.writerow(row)

            else:
                print("No context")
                print(row)
                row["# of articles"] = 0
                writer.writerow(row)
