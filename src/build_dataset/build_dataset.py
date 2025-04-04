import json
import os
import shutil
import sys
from warnings import showwarning

from tqdm.asyncio import tqdm

from .config import CONFIG
from article_retriever import ArticleRetriever
from demagog_scraper import DemagogScraper
from evidence_retriever import evidence_retriever_factory
from dataset_manager.orm import *
from sqlalchemy import insert


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
    dataset = init_db(os.path.join(output_dir, "dataset.sqlite"))

    if CONFIG["UseExistingStatements"]:
        # load existing statements
        print("Using existing statements")
        stmts = dataset.query(Statement).all()
    else:
        # scrape statements
        scraper = DemagogScraper(CONFIG["FromYear"], CONFIG["ToYear"], CONFIG["FirstNPages"])

        # include demagog evidence links if the evidence retriever is demagog 
        stmts = await scraper.run(CONFIG["EvidenceRetriever"] == "demagog")
        sql_stmt = insert(Statement).values(stmts)
        dataset.execute(sql_stmt)
        dataset.commit()


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
        # skip already scraped evidence (file exists)
        scraped_ids = dataset.query(Article.id).distinct(Article.statement_id).all()
        filtered_evidence_links = [item for item in evidence_links if item["id"] not in scraped_ids]

        print(f"Skipping {len(scraped_ids)} already scraped statements")

        # prepare inputs for batch retrieval
        links_batches = [ [ result["url"] for result in item["results"] ] for item in filtered_evidence_links ]
        batch_ids = [item["id"] for item in filtered_evidence_links]

        # scrape articles
        tasks = [retrieve_and_save(dataset, id, links) for id,links in zip(batch_ids, links_batches)]

        await tqdm.gather(*tasks, desc="Scraping articles (total progress)", unit="statements", file=sys.stdout)


async def retrieve_and_save(dataset: Session, statement_id, links):
    article_retriever = ArticleRetriever()
    articles = await article_retriever.batch_retrieve(links, show_progress=False)

    stmt = insert(Article).values(articles)
    stmt_relevance = insert(ArticleRelevance).values([{"statement_id": statement_id, "article_id": article["id"]} for article in articles])

    dataset.execute(stmt)
    dataset.execute(stmt_relevance)
    dataset.commit()
