import os
import json
import argparse
import dspy
import datetime
from dataset_manager.orm import *
from .hop_retriever import HopRetriever
import logging
from .search_functions import search_function_factory
from evaluate_dataset.fact_checker import FactChecker
from evaluate_dataset.llm_apis import openai_api

to_label = []

# save to logs/ folder
logging.basicConfig(filename=f"logs/retriever_{str(datetime.datetime.now()).replace(' ', '-')}.log", level=logging.INFO)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

llm = openai_api.OpenAI_API("")
fc = FactChecker("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")


def eval_metric(example, pred, trace=None):
    to_label.extend([{"statement": example['statement'], "statement_id": example['statement_id'], "id": segment['id'], "text": segment['text']} for segment in pred.segments])

    reference_segments = [rs['id'] for rs in example['relevant_segments']]
    predicted_segments = [segment['id'] for segment in pred.segments]

    recall = len(set(reference_segments) & set(predicted_segments)) / len(reference_segments)
    precision = len(set(reference_segments) & set(predicted_segments)) / len(predicted_segments) if len(predicted_segments) > 0 else 0

    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0

    # If we're just doing inference, just measure the recall.
    return recall if EVAL_METRIC == "recall" else precision

def eval_end2end(example, pred, trace=None)



def load_segments(dataset: Session) -> list[dict]:
    """
    Loads all segments from dataset
    """

    return as_dict_list(dataset.query(Segment).all())


def load_statements(dataset: Session) -> list[dict]:
    """
    Loads statements with their annotated relevant segments from the dataset.
    """
    statements = (
        dataset.query(Statement)
        .join(Statement.segments)  # Join through the segments relationship
        .join(SegmentRelevance)  # Join the segment_relevance table to filter by relevance
        .filter(SegmentRelevance.relevance == 1)  # Filter for relevance 1
        .all()  # Get all the results
    )

    statements_dict = as_dict_list(statements)

    for statement_dict, statement in zip(statements_dict, statements):
        statement_dict["segments"] = as_dict_list(statement.segments)

    return statements_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model-path", type=str, default="hosted_vllm/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--optimized-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="retriever-train-results")
    parser.add_argument("--num-threads", type=int, default=48)
    parser.add_argument("--search-function", type=str, default="bge-m3")
    parser.add_argument("--max-tokens", type=int, default=3000)
    # parser.add_argument("--save-index", action="store_true")
    # parser.add_argument("--index-path", type=str, default="")
    parser.add_argument("--eval-metric", type=str, default="recall")

    return parser.parse_args()


if __name__ == "__main__":
    global EVAL_METRIC

    args = parse_args()
    EVAL_METRIC = args.eval_metric

    # lm = dspy.LM(args.model_path, max_tokens=3000)
    lm = dspy.LM(args.model_path, api_base=args.api_base, max_tokens=args.max_tokens)
    dspy.configure(lm=lm, provide_traceback=True)

    dataset = init_db("datasets/curated.sqlite")
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = load_segments(dataset)
    statements = load_statements(dataset)

    # Build examples
    examples = [dspy.Example(statement=f"{s['statement']} - {s['author']}, {s['date']}", relevant_segments=s['segments'], statement_id=s['id']).with_inputs('statement') for s in statements]
    logger.info("Examples built")

    hop_retriever = HopRetriever(args.search_function, corpus, num_docs=2)

    if args.optimized_path:
        print("Loading optimized model")
        hop_retriever.load(args.optimized_path)

    if not args.train:
        evaluate = dspy.Evaluate(devset=examples, metric=eval_metric, num_threads=args.num_threads, display_progress=True, display_table=30, provide_traceback=True)
        evaluate(hop_retriever)
    else:
        # Train the model
        tp = dspy.MIPROv2(metric=eval_metric, auto="light", num_threads=args.num_threads, prompt_model=lm)

        kwargs = dict(minibatch_size=40, minibatch_full_eval_steps=4, requires_permission_to_run=False)
        optimized = tp.compile(hop_retriever, trainset=examples, max_bootstrapped_demos=4, max_labeled_demos=4, **kwargs)

        optimized.save(os.path.join(args.output_dir, "optimized_model.pkl"))

    with open(os.path.join(args.output_dir, "predicted_segments.json"), "w") as f:
        # unique predicted segment dicts
        json.dump(to_label, f, indent=2, ensure_ascii=False)
