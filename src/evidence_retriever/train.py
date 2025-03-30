import bm25s
import json
import argparse
import dspy
import simplemma
from dataset_manager.orm import *
from .hop_retriever import HopRetriever
import numpy as np

to_label = []

def stemmer(words):
    return [simplemma.lemmatize(word, "cs") for word in words]

def search_bm25(query: str, k: int):
    tokens = bm25s.tokenize(query, stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = [
        {"id": corpus[idx].id, "score": score, "text": corpus[idx].text}
        for idx, score in zip(results[0], scores[0])
    ]
    return run

def eval_metric(example, pred, trace=None):
    to_label.extend([{"statement": example.statement, "statement_id": example.statement_id, "id": segment['id'], "text": segment['text']} for segment in pred.segments])

    reference_segments = [rs.id for rs in example.relevant_segments]
    predicted_segments = [segment['id'] for segment in pred.segments]

    recall = len(set(reference_segments) & set(predicted_segments)) / len(reference_segments)

    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0

    # If we're just doing inference, just measure the recall.
    return recall


def load_segments(dataset: Session) -> list[Segment]:
    """
    Loads all segments from dataset
    """

    return dataset.query(Segment).all()


def load_statements(dataset: Session) -> list[Statement]:
    """
    Loads statements with their annotated relevant segments from the dataset.
    """
    statements_with_segments = (
        dataset.query(Statement)
        .join(Statement.segments)  # Join through the segments relationship
        .join(SegmentRelevance)  # Join the segment_relevance table to filter by relevance
        .filter(SegmentRelevance.relevance == 1)  # Filter for relevance 1
        .all()  # Get all the results
    )

    return statements_with_segments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model-path", type=str, default="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    parser.add_argument("--optimized-path", type=str, default=None, help="Path to the JSON optimized prompt")
    parser.add_argument("--output-path", type=str, default="optimized_hop.json")
    parser.add_argument("--output-predictions", type=str, default="predicted_segments.json")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # lm = dspy.LM(args.model_path, max_tokens=3000)
    lm = dspy.LM("hosted_vllm/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", api_base="http://0.0.0.0:8000/v1")
    dspy.configure(lm=lm, provide_traceback=True)

    dataset = init_db()

    corpus = load_segments(dataset)
    statements = load_statements(dataset)

    # Index the corpus for retrieval
    corpus_tokens = bm25s.tokenize([item.text for item in corpus], stemmer=stemmer)
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)

    # Build examples
    examples = [dspy.Example(statement=f"{s.statement} - {s.author}, {s.date}", relevant_segments=s.segments, statement_id=s.id).with_inputs('statement') for s in statements]

    hop_retriever = HopRetriever(search_bm25)
    if args.optimized_path:
        print("Loading optimized model")
        hop_retriever.load(args.optimized_path)

    if not args.train:
        evaluate = dspy.Evaluate(devset=examples, metric=eval_metric, num_threads=16, display_progress=True, display_table=30)
        evaluate(hop_retriever)
    else:
        # Train the model
        tp = dspy.MIPROv2(metric=eval_metric, auto="medium", num_threads=16, prompt_model=lm)

        kwargs = dict(minibatch_size=40, minibatch_full_eval_steps=4, requires_permission_to_run=False)
        optimized = tp.compile(hop_retriever, trainset=examples, max_bootstrapped_demos=4, max_labeled_demos=4, **kwargs)
        optimized.save(args.output_predictions)

    with open(args.output_path, "w") as f:
        # unique predicted segment dicts
        json.dump(to_label, f, indent=2, ensure_ascii=False)
