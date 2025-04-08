import json
import os
import shutil
import sys
from tqdm.asyncio import tqdm
from collections import defaultdict

from .config import CONFIG
from .article_scraper import ArticleScraper
from .demagog_scraper import DemagogScraper
from dataset_manager import Dataset
from .article_retriever import article_retriever_factory
from .segmenter import segment_article
from dataset_manager.orm import rows2dict
from dataset_manager.models import Article


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
    return output_dir


async def build_dataset():
    """
    Build a dataset from a list of statements, which are either scraped from Demagog.cz or loaded from a file.
    Each statement's evidence is saved to a separate file named by the statement's ID
    """

    output_dir = prepare_output_dir(CONFIG["OutputDir"])
    dataset = Dataset(os.path.join(output_dir, "dataset.sqlite"))

    if not CONFIG["UseExistingStatements"]:
        # scrape statements
        scraper = DemagogScraper(CONFIG["FromYear"], CONFIG["ToYear"], CONFIG["FirstNPages"])
        stmts = await scraper.run(include_evidence = CONFIG["EvidenceRetriever"] == "demagog")
        dataset.insert_statements(stmts)

    stmts = dataset.get_statements()

    if CONFIG["UseExistingEvidenceLinks"]:
        print("Using existing evidence links")
        with open(os.path.join(output_dir, "evidence_links.json"), "r") as file:
            evidence_links = json.load(file)
    else:
        # search for evidence for each statement
        evidence_retriever = article_retriever_factory(
            CONFIG["EvidenceRetriever"], CONFIG["EvidenceAPIKey"]
        )
        evidence_retriever.set_fetch_concurrency(CONFIG["FetchConcurrency"])
        evidence_retriever.set_fetch_delay(CONFIG["FetchDelay"])

        evidence_links = await evidence_retriever.batch_retrieve(rows2dict(stmts))

        with open(os.path.join(output_dir, "evidence_links.json"), "w+") as file:
            json.dump(evidence_links, file, ensure_ascii=False, indent=4)

    if CONFIG["ScrapeArticles"]:
        # skip already scraped evidence
        scraped_ids = [a.id for a in dataset.get_articles()]

        filtered_evidence_links = [item for item in evidence_links if item["id"] not in scraped_ids]

        print(f"Skipping {len(scraped_ids)} already scraped statements")

        # prepare inputs for batch retrieval
        links_batches = [ [ result["url"] for result in item["results"] ] for item in filtered_evidence_links ]
        batch_ids = [item["id"] for item in filtered_evidence_links]

        # scrape articles
        tasks = [retrieve_and_save(dataset, id, links) for id,links in zip(batch_ids, links_batches)]

        await tqdm.gather(*tasks, desc="Scraping articles (total progress)", unit="statements", file=sys.stdout)


async def retrieve_and_save(dataset: Dataset, statement_id, links):
    article_scraper = ArticleScraper()
    articles = await article_scraper.scrape_extractus(links, show_progress=True)

    inserted_articles = dataset.insert_articles(articles)

    for a in tqdm(inserted_articles, desc=f"Setting article relevance {'and segmenting articles' if CONFIG['SegmentArticles'] else ''}", unit="articles", file=sys.stdout):
        dataset.set_article_relevance(statement_id, a.id)

        if CONFIG["SegmentArticles"]:
            segments = [
                {"article_id": a.id, "text": segment}
                for segment in segment_article(a.content, min_len=25)
            ]

            dataset.insert_segments(segments)

