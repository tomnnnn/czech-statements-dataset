import csv
import argparse

argparser = argparse.ArgumentParser(description="Write out ids as json files")
argparser.add_argument("file", type=str, help="File to extract ids from")
argparser.add_argument("start_row", type=int, help="Row to start extracting ids from")

FILE = argparser.parse_args().file
START_ROW = argparser.parse_args().start_row

with open(FILE, "r") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]
    for row in rows[START_ROW:]:
        print(f"{row[0]}.json")
