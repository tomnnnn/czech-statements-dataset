import json
from collections import defaultdict

with open("./missing_urls.json", "r") as f:
    data = json.load(f)

url_map = defaultdict(set)

for entry in data:
    url = entry["url"]
    statement_id = entry["statement_id"]

    url_map[url].add(statement_id)

with open("./url_map.json", "w") as f:
    json.dump({
        key: list(value)
        for key, value in url_map.items()
    }, f, indent=4)
