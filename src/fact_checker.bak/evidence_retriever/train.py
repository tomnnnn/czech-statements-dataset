import argparse
import asyncio
import time
import requests
import datetime
import json
import logging
import os
import pprint
import subprocess
from sklearn.model_selection import train_test_split

import dspy
from tqdm.asyncio import tqdm_asyncio
from sentence_transformers import SentenceTransformer

from dataset_manager import Dataset
from src.dataset_manager.models import Statement
from utils.llm_apis import openai_api

from ..veracity_predictor import BasicPredictor
from .retrievers import HopRetriever
from .search_functions import BGE_M3, BM25

to_label = []
dataset = None

# save to logs/ folder
logging.basicConfig(
    filename=f"logs/retriever_{str(datetime.datetime.now()).replace(' ', '-')}.log",
    level=logging.INFO,
)

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

llm = openai_api.OpenAI_API("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", show_progress=False, api_base_url="http://0.0.0.0:8000/v1")
predictor = BasicPredictor(llm, "prompts/unveri.yaml")

async def create_search_functions(dataset, statements: list[Statement]):
    segments = dataset.get_segments_by_statements([s.id for s in statements])

    # Filter statements with segments
    statements = [s for s in statements if s.id in segments]

    search_functions = {}

    semaphore = asyncio.Semaphore(10)

    # Load model
    model = SentenceTransformer("BAAI/BGE-M3")

    async def build_search_fn(statement: Statement):
        async with semaphore:
            segment_list = segments[statement.id]
            index_path = f"indexes/{statement.id}.faiss"
            load_index = os.path.exists(index_path)

            search_fn = BGE_M3(
                segment_list,
                save_index=not load_index,
                load_index=load_index,
                index_path=index_path,
                model=model,
            )
            await search_fn.index_async()
            return statement.id, search_fn

        # search_fn = BM25(segment_list)
        # return statement.id, search_fn

    # Run throttled async tasks
    results = await tqdm_asyncio.gather(
        *(build_search_fn(s) for s in statements),
        desc="Building evidence document indices",
        unit="indices",
    )

    for statement_id, search_fn in results:
        search_functions[statement_id] = search_fn

    return search_functions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        default="hosted_vllm/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        # default="openai/gpt-4o"
    )
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--optimized-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="retriever-train-results")
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--search-function", type=str, default="bge-m3")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--dataset-path", type=str, default="datasets/dataset_demagog.sqlite")
    # parser.add_argument("--save-index", action="sore_true")
    # parser.add_argument("--index-path", type=str, default="")
    parser.add_argument("--eval-metric", type=str, default="recall")

    return parser.parse_args()

def start_vllm_server():
    subprocess.Popen( ["vllm", "serve", "--enable-chunked-prefill", "true", "--gpu-memory-utilization", "0.85", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"])

    print("Waiting for server to start...")
    while True:
        try:
            response = requests.get("http://0.0.0.0:8000/health")
            if response.status_code == 200:
                print("Server is up and running!")
                break
        except requests.exceptions.RequestException:
            time.sleep(5)


def eval_metric(example, pred, trace=None):
    segments = pred.segments

    evidence = [
        {
            "text": segment.text[:3000],
        }
        for segment in segments
    ]

    # Run the async function synchronously using asyncio.run()
    label, _ = predictor.predict_sync(example.statement, evidence)

    if label.lower() != example.label.lower():
        return 0
    else:
        return 1


async def main():
    global dataset
    args = parse_args()

    dataset = Dataset(args.dataset_path, read_only=True)

    lm = dspy.LM(
        args.model_path,
        api_base="http://0.0.0.0:8000/v1",
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    dspy.configure(lm=lm, provide_traceback=True)

    os.makedirs(args.output_dir, exist_ok=True)

    statements = dataset.get_statements()
    # labels = [s.label for s in statements]
    # _, statements = train_test_split(statements, test_size=1, random_state=42, stratify=labels)

    print("Creating search functions")
    search_functions = await create_search_functions(dataset, statements)

    start_vllm_server()
    
    print("Building examples")
    # Build examples
    examples = [
        dspy.Example(
            statement=f"{s.statement} - {s.author}, {s.date}",
            search_func=search_functions[s.id],
            label=s.label,
            statement_id=s.id,
        ).with_inputs("statement", "search_func")
        for s in statements
    ]
    hop_retriever = HopRetriever(num_hops=4, num_docs=3)

    if args.optimized_path:
        print("Loading optimized model")
        hop_retriever.load(args.optimized_path)

    if not args.train:
        evaluate = dspy.Evaluate(
            devset=examples,
            metric=eval_metric,
            num_threads=args.num_threads,
            display_progress=True,
            return_all_scores=True,
            provide_traceback=True
        )
        result = evaluate(hop_retriever, return_all_scores=True)
        print(result)
    else:
        # Train the model
        print("Training the model")
        tp = dspy.MIPROv2(
            metric=eval_metric,
            auto="light",
            num_threads=args.num_threads,
            prompt_model=lm,
        )

        kwargs = dict(
            minibatch_size=40,
            minibatch_full_eval_steps=4,
            requires_permission_to_run=False,
        )


        optimized = tp.compile(
            hop_retriever,
            trainset=examples,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            **kwargs,
        )

        optimized.save(os.path.join(args.output_dir, "optimized_model.pkl"))

    with open(os.path.join(args.output_dir, "predicted_segments.json"), "w") as f:
        # unique predicted segment dicts
        json.dump(to_label, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
