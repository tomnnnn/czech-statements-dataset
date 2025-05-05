import mlflow
import csv
from fact_checker import FactChecker 
import os
from .utils import create_examples, evaluate, sample_statements, setup_dspy, setup_mlflow

# create examples
statements = sample_statements("random", portion=0.05)
examples = create_examples(statements)

# prepare output folder
output_folder = "results/dspy/num_hops"
os.makedirs(output_folder, exist_ok=True)

setup_mlflow("Find Best num_hops")
setup_dspy()

k = 4
num_hops_range = range(1,10)
macro_f1 = []

# ============= Unfiltered ===============

for num_hops in num_hops_range:
    with mlflow.start_run(run_name=str(num_hops)) as run:
        print(f"Running num_hops: {num_hops}")
        fact_checker = FactChecker(search_endpoint="http://localhost:4242/search", retrieval_hops=num_hops, per_hop_documents=k, mode="binary")
        results, report = evaluate(fact_checker, examples, output_folder, allowed_labels=["pravda", "nepravda"])
        macro_f1.append(report["macro avg"]["f1-score"])


with open(os.path.join(output_folder, "results.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["num_hops", "macro_f1"])
    for num_hops, f1 in zip(num_hops_range, macro_f1):
        writer.writerow([num_hops, f1]) 

# ============== Filtered ===============

print("============= Running Filtered ==============")

# create examples
statements = sample_statements("filtered", portion=0.5)
examples = create_examples(statements)

k = 4
num_hops_range = range(1,10)
macro_f1 = []

setup_mlflow("Find Best num_hops (filtered)")
setup_dspy()

for num_hops in num_hops_range:
    with mlflow.start_run(run_name=str(num_hops)) as run:
        print(f"Running num_hops: {num_hops}")
        fact_checker = FactChecker(search_endpoint="http://localhost:4242/search", retrieval_hops=num_hops, per_hop_documents=k, mode="binary")
        results, report = evaluate(fact_checker, examples, output_folder, allowed_labels=["pravda", "nepravda"])
        macro_f1.append(report["macro avg"]["f1-score"])


with open(os.path.join(output_folder, "results_filtered.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["num_hops", "macro_f1"])
    for num_hops, f1 in zip(num_hops_range, macro_f1):
        writer.writerow([num_hops, f1]) 
