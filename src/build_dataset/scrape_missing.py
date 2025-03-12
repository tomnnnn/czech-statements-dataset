"""
@file: build_dataset.py
@author: Hai Phong Nguyen

This script scrapes statements from Demagog.cz and provides scraped evidence documents for each statement.
"""

import asyncio
import json
import os
import shutil
import sys

from tqdm.asyncio import tqdm

import config
from article_retriever import ArticleRetriever
from demagog_scraper import DemagogScraper
from evidence_retriever import evidence_retriever_factory

CONFIG = config.load_config("scraper_config.yaml")


def prepare_output_dir(output_dir):
    if os.path.exists(output_dir) and not CONFIG["UseExistingStatements"]:
        # confirmation
        print(
            f"Directory {output_dir} already exists. Do you want to overwrite it? (y/n)"
        )
        choice = input().lower()
        if choice != "y":
            print("Exiting...")
            sys.exit(0)

        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/evidence", exist_ok=True)
    return output_dir


async def build_dataset(output_dir):
    """
    Build a dataset from a list of statements
    Saves pruned statements to file and provides evidence for each statement
    Each statement's evidence is saved to a separate file named by the statement's ID
    """

    output_dir = prepare_output_dir(output_dir)

    with open('./datasets/with_evidence/demagog/not_scraped.json') as f:
        evidence_links = json.load(f)


    if CONFIG["ScrapeArticles"]:
        article_retriever = ArticleRetriever(CONFIG["FetchConcurrency"], CONFIG["FetchDelay"])

        filtered_evidence_links = [item for item in evidence_links if f"{item['id']}.json" not in os.listdir(os.path.join(output_dir, "missing_evidence"))]
        print(f"Skipping {len(evidence_links) - len(filtered_evidence_links)} already scraped statements")

        links_batches = [ [ url for url in item["urls"] ] for item in filtered_evidence_links ]
        batch_ids = [item["id"] for item in filtered_evidence_links]
        batch_save_paths = [os.path.join(output_dir, "missing_evidence", f"{id}.json") for id in batch_ids]

        tasks = [article_retriever.batch_retrieve_save(path,links,id, show_progress=False) for path,id,links in zip(batch_save_paths, batch_ids, links_batches)]
        await tqdm.gather(*tasks, desc="Scraping articles", unit="statements", file=sys.stdout)


async def main():
    await build_dataset(CONFIG["OutputDir"])


if __name__ == "__main__":
    asyncio.run(main())
