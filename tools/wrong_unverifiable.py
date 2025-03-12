#!/usr/bin/env python

import json
import os
import argparse

def find_by_id(id, data):
    for item in data:
        if item["id"] == id:
            return item

    return None


parser = argparse.ArgumentParser()
parser.add_argument(
    "path",
    type=str,
    default=os.getcwd(),
    help="Path to results file.",
)

args = parser.parse_args()
path = args.path

with open(os.path.join(path, 'aggregated.json'), "r") as f:
    data = json.load(f)
    y_pred = data["y_pred"]
    y_ref = data["y_ref"]

    items = [item for item in y_pred if item["label"].lower() == "neověřitelné" and find_by_id(item["id"], y_ref)["assessment"] != "neověřitelné"]

    with open("unverfiable_responses.json", "w") as f2:
        json.dump({
            "count": len(items),
            "items": items
        }, f2, indent=4,ensure_ascii=False)
    
    print(f"Wrong unverifiable count: {len(items)} ({len(items)/len(data['y_ref'])})")

