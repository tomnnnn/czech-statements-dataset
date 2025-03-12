import json
import os
import shutil
import sys

from tqdm.asyncio import tqdm

from .config import CONFIG
from article_retriever import ArticleRetriever
from demagog_scraper import DemagogScraper
from evidence_retriever import evidence_retriever_factory


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


async def build_dataset():
    """
    Build a dataset from a list of statements, which are either scraped from Demagog.cz or loaded from a file.
    Each statement's evidence is saved to a separate file named by the statement's ID
    """

    output_dir = prepare_output_dir(CONFIG["OutputDir"])

    if CONFIG["UseExistingStatements"]:
        # load existing statements
        print("Using existing statements")
        try:
            with open(f"{CONFIG['OutputDir']}/statements.json", "r") as file:
                stmts = json.load(file)
        except FileNotFoundError:
            print("UseExistingStatements is set to True, but no statements.json file was found in the output directory.")
            exit(1)
    else:
        # scrape statements
        scraper = DemagogScraper(CONFIG["FromYear"], CONFIG["ToYear"], CONFIG["FirstNPages"])

        # include demagog evidence links if the evidence retriever is demagog 
        stmts = await scraper.run(CONFIG["EvidenceRetriever"] == "demagog")

        with open(os.path.join(output_dir, 'statements.json'), "w+") as file:
            json.dump(stmts, file, ensure_ascii=False, indent=4)

    if CONFIG["UseExistingEvidenceLinks"]:
        print("Using existing evidence links")
        with open(os.path.join(output_dir, "evidence_links.json"), "r") as file:
            evidence_links = json.load(file)
    else:
        # search for evidence for each statement
        evidence_retriever = evidence_retriever_factory(
            CONFIG["EvidenceRetriever"], CONFIG["EvidenceAPIKey"]
        )
        evidence_retriever.set_fetch_concurrency(CONFIG["FetchConcurrency"])
        evidence_retriever.set_fetch_delay(CONFIG["FetchDelay"])

        evidence_links = await evidence_retriever.batch_retrieve(stmts)

        with open(os.path.join(output_dir, "evidence_links.json"), "w+") as file:
            json.dump(evidence_links, file, ensure_ascii=False, indent=4)

    if CONFIG["ScrapeArticles"]:
        article_retriever = ArticleRetriever()

        # skip already scraped evidence (file exists)
        filtered_evidence_links = list(filter(lambda x: not os.path.exists(os.path.join(output_dir, "evidence", f"{x['id']}.json")), evidence_links))
        print(f"Skipping {len(evidence_links) - len(filtered_evidence_links)} already scraped evidence links")

        # prepare inputs for batch retrieval
        links_batches = [ [ result["url"] for result in item["results"] ] for item in filtered_evidence_links ]
        batch_ids = [item["id"] for item in filtered_evidence_links]
        batch_save_paths = [os.path.join(output_dir, "evidence", f"{id}.json") for id in batch_ids]

        # scrape articles
        tasks = [article_retriever.batch_retrieve_save(path,links,id, show_progress=False) for path,id,links in zip(batch_save_paths, batch_ids, links_batches)]
        await tqdm.gather(*tasks, desc="Scraping articles (total progress)", unit="statements", file=sys.stdout)
