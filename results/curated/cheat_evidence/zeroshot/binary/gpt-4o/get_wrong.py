from dataset_manager import DemagogDataset
import json

dataset = DemagogDataset("../../../../../datasets/curated.sqlite")

wrong = []
with open('responses_0.json', 'r') as f:
    responses = json.load(f)

    for r in responses:
        predicted = r['label']
        true = dataset.get_statement(r['id'])['label']

        if predicted.lower() != true.lower():
            wrong.append(r)


with open('wrong.json', 'w') as f:
    json.dump(wrong, f, indent=4 ,ensure_ascii=False)
