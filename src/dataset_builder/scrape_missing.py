"""
@file: build_dataset.py
@author: Hai Phong Nguyen

This script scrapes statements from Demagog.cz and provides scraped evidence documents for each statement.
"""

import asyncio
from datetime import datetime
import json
import os
import shutil
import sys
import aiohttp

from tqdm.asyncio import tqdm

from dataset_manager import Dataset
import sqlite3
from .article_scraper import ArticleScraper

def get_missing_urls():
    con = sqlite3.connect("datasets/dataset.sqlite")
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("""
        SELECT * FROM failed_scrapes
    """)

    rows = cur.fetchall()

    row_dicts = [dict(row) for row in rows]

    return row_dicts

async def main():
    scraper = ArticleScraper()
    dataset = Dataset("datasets/dataset_demagog.sqlite")
    articles = dataset.get_articles()
    missing_articles = get_missing_urls()
    scraped_urls = [
        a.url for a in articles
    ]
    
    # filter identical urls
    missing_urls = []

    for article in missing_articles:
        if article.get("url", None) not in scraped_urls:
            missing_urls.append(article)

    with open("missing_urls.json", "w") as f:
        json.dump(missing_urls, f, indent=4, ensure_ascii=False)

    missing_urls = [ i["url"] for i in missing_urls if i["url"] if not i["url"].endswith(".pdf")]
    missing_urls = [i for i in missing_urls if not "195.46.72.16" in i]
    missing_urls = list(set(missing_urls))

    articles,unprocessed = await scraper.scrape_extractus_async(missing_urls)

    for a in articles:
        a["accessed"] = datetime.now().isoformat()

    with open("articles.json", "w") as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)

    with open("unprocessed.json", "w") as f:
        json.dump(unprocessed, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    asyncio.run(main())
