"""
@file scrape_statements.py
@author: Hai Phong Nguyen (xnguye28@stud.fit.vutbr.cz)

Script to scrape statements from https://demagog.cz/
"""

from bs4 import BeautifulSoup
from dataclasses import dataclass
import time
import sys
from utils import fetch, config, track_progress
import asyncio as asyncio
import itertools

BASE_URL = "https://demagog.cz"
STMTS_URL = "https://demagog.cz/vyroky"


@dataclass
class Statement:
    id: int
    link: str
    date: str
    assessment: str
    statement: str
    speaker: str

async def scrapeStatementsFromPage(url, fetch_sem, filter_func=None, statement_cnter=itertools.count(1)):
    statements = []

    async with fetch_sem:
        html = await fetch(url)
        await asyncio.sleep(config['FetchDelay'])

    if not html:
        return statements

    soup = BeautifulSoup(html, "html.parser")
    statement_divs = soup.find_all("div", class_="s-statement")

    for statement_div in statement_divs:
        try:
            # statement link
            accordion_div = statement_div.find("div", class_="accordion")
            explanation = ""

            if accordion_div:
                explanation = accordion_div.findChildren("div", recursive=False)[2].select(
                    "a"
                )[1]["href"]
            else:
                explanation = ' '.join([p.get_text() for p in statement_div.select(".d-block p")])

            # date
            citation = statement_div.find("cite").get_text(strip=True)
            date = citation.split(",")[1]

            # assessment
            assessment_div = statement_div.find(
                "div", {"data-sentry-component": "StatementAssessment"}
            )
            assessment = assessment_div.findChildren("span", recursive=False)[
                1
            ].get_text(strip=True)

            # statement text
            blockquote_div = statement_div.find("blockquote")
            statement_text = blockquote_div.select("span")[1].get_text(strip=True)
            # remove any words that contain "demagog" in them
            statement_text = ' '.join([word for word in statement_text.split() if "demagog" not in word.lower()])
            

            # speaker url
            speakerLink = statement_div.find(
                "a", {"data-sentry-component": "SpeakerLink"}
            )["href"]

            statement = {
                'id':next(statement_cnter),
                'statement':statement_text,
                'explanation':explanation,
                'date':date,
                'assessment':assessment,
                'speaker':speakerLink,
            }

            # Apply the filter if provided
            if not filter_func or filter_func(statement):
                statements.append(statement)

        except (IndexError, AttributeError) as e:
            print(f"Error processing statement at {url}: {e}", file=sys.stderr)

    return statements


async def scrapeStatements(from_page=1, to_page=300, filterFunc=None, start_index=1):
    start_time = time.time()
    statements = []
    index_counter = itertools.count(start_index)
    statement_tracker = itertools.count(1)
    fetch_sem = asyncio.Semaphore(config["FetchesPerDelay"])

    print(f"Scraping Demagog from page {from_page} to page {to_page}")
    coros = [
        track_progress(scrapeStatementsFromPage(f"{STMTS_URL}?page={page}", fetch_sem, filterFunc, index_counter), statement_tracker, to_page + 1 - from_page, 'page')
        for page in range(from_page, to_page+1)
    ]

    pages = await asyncio.gather(*coros)

    for page in pages:
        statements.extend(page)

    end_time = time.time()
    print(f"Scraped Demagog in {end_time - start_time:.2f} seconds")
    return statements
