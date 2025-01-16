import json
import csv

STATEMENTS = '../datasets/no_context/statements.json'
OUTPUT = 'bing.csv'
INPUT = '../quality_check/bing.csv'

header = "id;author;date;statement;# of articles;# of relevant articles;# of broken articles;contains original source of statement;note;misleading sources"

def fint_stmt_by_id(id, statements):
    for stmt in statements:
        if int(stmt['id']) == int(id):
            return stmt

with open(INPUT, 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    rows = list(reader)

    # open csv file to write
    with open(OUTPUT, 'w') as out:
        with open(STATEMENTS, 'r') as f:
            statements = json.load(f)
            csv = csv.writer(out)
            csv.writerow(header.split(';'))

            for row in rows:
                stmt = fint_stmt_by_id(row['id'], statements)
                if stmt:
                    print(stmt)
                    csv.writerow([row['id'], stmt['author_name'], stmt['date'], row['statement'], row['# of articles'],row['# of relevant articles'],row['# of broken articles'],row['contains original source of statement'], row['note'], row['misleading sources']])
