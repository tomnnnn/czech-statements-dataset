"""
Loads old statements json and changes its fields
"""

import json

OLD_JSON = '../datasets/no_context/statements.json'
NEW_JSON = '../demagog_statements.json'
OUT_JSON = 'demagog_transformed_statements.json'

def find_by_field(statements, field, value):
    for statement in statements:
        if statement[field] == value:
            return statement

with open(OLD_JSON, 'r') as old:
    with open(NEW_JSON, 'r') as new:
        print(new)
        data_old = json.load(old)
        data_new = json.load(new)

        new_statements = []

        for statement in data_old:
            new_statement = find_by_field(data_new, 'statement', statement['statement'])
            if new_statement:
                new_statement['id'] = statement['id']
                new_statements.append(new_statement)

        with open(OUT_JSON, 'w') as out:
            json.dump(new_statements, out, indent=4, ensure_ascii=False)

print(f"Transformed {len(new_statements)} statements")
