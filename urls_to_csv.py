import json
import csv

with open("./url_map.json", "r") as file:
    data = json.load(file)

with open("./url_map.csv", "w", newline="") as csvfile:
    for url in data.keys():
        writer = csv.writer(csvfile)
        writer.writerow([url])
