import random
import csv
import json
import os

STATEMENTS_DIR = '../datasets/criteria'
STATEMENTS_PATH = f'{STATEMENTS_DIR}/statements.json'
OUTPUT_PATH = 'assessment_template.csv'

# Generate 300 random numbers between 1 and 10000, unique
id_sample = random.sample(range(1, 10001), 300)

columns = ['id','date', 'author', 'statement', '# of context articles', '# of relevant articles','notes', 'misleading sources']

# For each id, get statement from statements json file
with open(STATEMENTS_PATH, 'r') as f:
    # Load statements from json file
    statements = json.load(f)

    with open('OUTPUT_PATH', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for id in id_sample:
            statement = next((item for item in statements if item['id'] == id), None)
            if statement:
                if os.path.isfile(f'{STATEMENTS_DIR}/context/{id}.json'):
                    with open(f'{STATEMENTS_DIR}/context/{id}.json', 'r') as f:
                        context = json.load(f)
                        num_articles = len(context)
                else:
                    num_articles = ''

                writer.writerow([id,statements['date'], statements['author_name'], statement['statement'], num_articles, '', ''])
