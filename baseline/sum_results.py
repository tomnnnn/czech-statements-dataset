import json
import os

FOLDER = './after_9_2024/'
FILES = os.listdir(FOLDER)
OUTPUT = 'results-after_9_2024-with-noi.txt'

results = []
for file in FILES:
    if file.endswith('results.json'):
        with open(FOLDER + file) as f:
            data = json.load(f)
            results.append({
                "model": file.replace('_results.json', ''),
                "total": data['total'],
                "correct": data['correct'],
                "accuracy": data['accuracy']
            })

with open(OUTPUT, 'w') as o:
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    for result in results:
        o.write(f"{result['model']} - {round(result['accuracy']*100, 3)} %\n")

