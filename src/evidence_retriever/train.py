import bm25s
import os
import json
import argparse
import dspy
import simplemma
import datetime
from dataset_manager.orm import *
from .hop_retriever import HopRetriever
import logging

to_label = []
# save to logs/ folder
logging.basicConfig(filename=f'logs/retriever_{str(datetime.datetime.now()).replace(' ', '-')}.log', level=logging.INFO)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def stemmer(words):
    return [simplemma.lemmatize(word, "cs") for word in words]

def search_bm25(query: str, k: int):
    tokens = bm25s.tokenize(query, stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=10, show_progress=False)
    run = [
        {"id": corpus[idx].id, "score": score, "text": corpus[idx].text} # type: ignore
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
    parser.add_argument("--model-path", type=str, default="hosted_vllm/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--optimized-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="retriever-train-results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # lm = dspy.LM(args.model_path, max_tokens=3000)
    lm = dspy.LM(args.model_path, api_base=args.api_base)
    dspy.configure(lm=lm, provide_traceback=True)

    dataset = init_db()
    os.makedirs(args.output_dir, exist_ok=True)

    corpus = load_segments(dataset)
    statements = load_statements(dataset)

    # Index the corpus for retrieval
    corpus_tokens = bm25s.tokenize([item.text for item in corpus], stemmer=stemmer) # type: ignore
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)
    logger.info("Corpus indexed")

    # Build examples
    examples = [dspy.Example(statement=f"{s.statement} - {s.author}, {s.date}", relevant_segments=s.segments, statement_id=s.id).with_inputs('statement') for s in statements]
    logger.info("Examples built")

    hop_retriever = HopRetriever(search_bm25)

    if args.optimized_path:
        print("Loading optimized model")
        hop_retriever.load(args.optimized_path)

    if not args.train:
        evaluate = dspy.Evaluate(devset=examples, metric=eval_metric, num_threads=48, display_progress=True, display_table=30)
        evaluate(hop_retriever)
    else:
        # Train the model
        tp = dspy.MIPROv2(metric=eval_metric, auto="light", num_threads=48, prompt_model=lm)

        kwargs = dict(minibatch_size=40, minibatch_full_eval_steps=4, requires_permission_to_run=False)
        optimized = tp.compile(hop_retriever, trainset=examples, max_bootstrapped_demos=4, max_labeled_demos=4, **kwargs)

        optimized.save(os.path.join(args.output_dir, "optimized_model.pkl"))

    with open(os.path.join(args.output_dir, "predicted_segments.json"), "w") as f:
        # unique predicted segment dicts
        json.dump(to_label, f, indent=2, ensure_ascii=False)
