"""
@file: build_dataset.py
@author: Hai Phong Nguyen

This script scrapes statements from Demagog.cz and provides scraped evidence documents for each statement.
"""

import scrape_statements as scraper
import json
import datetime
import bs4
import time
import asyncio as asyncio
import aiofiles
import os
import itertools
from collections import defaultdict
import contextvars
import sys
import shutil
from utils.scraper_utils import fetch, post_request, search_bing, track_progress
import config

SEARCH_SEM = contextvars.ContextVar("search_sem")
FETCH_SEM = contextvars.ContextVar("fetch_sem")
CONFIG = config.load_config("scraper_config.yaml")

async def extract_article(url):
    """
    Extract article content from a given URL
    Heuristitc: Find the parent element with the most <p> children
    """
    if CONFIG["EvidenceLinkOnly"]:
        return {
            "url": url,
        }

    async with FETCH_SEM.get():
        html = await fetch(url)
        if not html:
            return None
        await asyncio.sleep(CONFIG['FetchDelay'])

    MIN_NUM_WORDS = 5
    try:
        soup = bs4.BeautifulSoup(html, "html.parser", parse_only=bs4.SoupStrainer(["title","body"]))
    except Exception as e:
        print(f"Failed to parse HTML for {url}: {e}", file=sys.stderr)
        return None

    p_parents = defaultdict(list)

    ps = soup.find_all("p")
    if len(ps) > 1000:
        # skip articles with too many paragraphs to avoid long processing times
        print(f"Skipping article with too many paragraphs ({len(ps)}): {url}", file=sys.stderr)
        return None

    for p in ps:
        p_parents[p.parent].append(p)

    parents_counts = sorted([(parent, len(ps)) for parent, ps in p_parents.items()], key=lambda v: -v[1])
    if not parents_counts:
        return None

    article_dom = parents_counts[0][0]
    article_text = " ".join(p.get_text().strip() for p in article_dom.find_all("p") if len(p.get_text().split()) > MIN_NUM_WORDS)
    title = soup.find('title')
    title = title.get_text() if title else ""

    return {
        "url": url,
        "title": title,
        "content": article_text
    }

async def scrape_evidence(query):
    """
    Search for evidence articles for a given query and return their content
    Uses Criteria API or Bing API based on config
    """
    if CONFIG["SearchAPI"] == 'criteria':
        data = dict(claim=query)
        async with SEARCH_SEM.get():
            await asyncio.sleep(CONFIG['SearchDelay'])
            results = await post_request("https://lab.idiap.ch/criteria/search_tom", data)

        return [{
            'url': result['url'],
            'title': result['title'],
            'date': datetime.datetime.fromtimestamp(result['date']/1000).strftime('%d-%m-%Y'),
            'score': result['score'],
            'content': result['fulltext'],
            } for result in results
        ] if results else []
    else:
        num_results = CONFIG["EvidenceNumBuffer"]
        if num_results == 0:
            return []

        urls = []

        async with SEARCH_SEM.get():
            if(CONFIG["SearchAPI"] == 'bing'):
                # Bing search
                try:
                    query += " -site:demagog.cz -site:facebook.com -site:reddit.com -site:instagram.com -site:x.com"
                    result = await search_bing(query, CONFIG["BingAPIKey"], num_results)
                    urls = [item['url'] for item in result]
                except KeyError as e:
                    print(f"Please, set BING_API_KEY environment variable to use Bing search API: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Failed to fetch Bing search results: {e}", file=sys.stderr)

            await asyncio.sleep(CONFIG['SearchDelay'])

        evidence = []
        article_coros = [ extract_article(url) for url in urls ]

        articles = await asyncio.gather(*article_coros)
        if not articles:
            print(f"Warning: No articles found for query: {query}", file=sys.stderr)
            return []

        for article in articles:
            if len(evidence) >= CONFIG["EvidenceNum"]:
                break
            elif CONFIG["EvidenceLinkOnly"]:
                evidence.append(article)
            elif article and article['content']:
                evidence.append(article)

        return evidence


async def provide_evidence(stmt):
    """
    Provide evidence for a given statement
    """

    query = stmt['statement']
    evidence = await scrape_evidence(query)

    async with aiofiles.open(f"{CONFIG['OutputDir']}/evidence/{stmt['id']}.json", mode="w+") as file:
            await file.write(json.dumps(evidence, ensure_ascii=False, indent=4))
            await file.close()

def prepare_output_dir():
    out_dir = CONFIG['OutputDir']

    if(os.path.exists(out_dir) and not CONFIG["UseExistingStatements"]):
        # confirmation
        print(f"Directory {out_dir} already exists. Do you want to overwrite it? (y/n)")
        choice = input().lower()
        if choice != 'y':
            print("Exiting...")
            sys.exit(0)

        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/evidence", exist_ok=True)
    return out_dir


async def build_dataset():
    """
    Build a dataset from a list of statements
    Saves pruned statements to file and provides evidence for each statement
    Each statement's evidence is saved to a separate file named by the statement's ID
    """

    out_dir = prepare_output_dir()

    # scrape statements if not using existing statements
    if CONFIG["UseExistingStatements"]:
        print("Using existing statements")
        with open(f"{CONFIG['OutputDir']}/statements.json", "r") as file:
            existiting_evidence = [f for f in os.listdir(f"{CONFIG['OutputDir']}/evidence") if f.endswith(".json")]
            stmts = json.load(file)
            stmts = [stmt for stmt in stmts if f"{stmt['id']}.json" not in existiting_evidence]
    else:
        stmts = await scraper.scrapeByYears(CONFIG["FromYear"], CONFIG["ToYear"], CONFIG["DemagogToPage"])

        # include only fields enabled in CONFIG
        stmts_pruned = [{k:v for k,v in temp.items() if v} for temp in [{
            "id": stmt['id'],
            "statement": stmt['statement'],
            "author": stmt['author'] if CONFIG["IncludeAuthor"] else "",
            "date": stmt['date'] if CONFIG["IncludeDate"] else "",
            "explanation":  stmt['explanation'] if CONFIG["IncludeExplanation"] else "",
            "assessment": stmt['assessment'] if CONFIG["IncludeAssessment"] else "",
        } for stmt in stmts]]

        # write pruned statements to file
        with open(f"./{out_dir}/statements.json", "w+") as file:
            json.dump(stmts_pruned, file, ensure_ascii=False, indent=4)


    if CONFIG["ScrapeWithEvidence"]:
        # provide evidence for each statement
        print("Providing evidence for each statement")

        stmt_cnter = itertools.count(0)
        evidence_coros = [
            track_progress(provide_evidence(stmt), stmt_cnter, len(stmts), "statement")
            for stmt in stmts
        ]
        await asyncio.gather(*evidence_coros)


async def main():
    global total_statements
    SEARCH_SEM.set(asyncio.Semaphore(CONFIG["SearchesPerDelay"]))
    FETCH_SEM.set(asyncio.Semaphore(CONFIG["FetchesPerDelay"]))

    start_time = time.time()

    await build_dataset()

    print(f"Execution time: {time.time() - start_time}")


if __name__ == "__main__":
    asyncio.run(main())

