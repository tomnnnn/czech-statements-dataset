"""
@file scrape_statements.py
@author: Hai Phong Nguyen (xnguye28@stud.fit.vutbr.cz)

Script to scrape statements from https://demagog.cz/
"""

from bs4 import BeautifulSoup
from dataclasses import dataclass
import time

from gpt4all.gpt4all import sys
from utils import fetch
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

stmtCnter = itertools.count(0)


async def scrapeStatementsFromPage(url, filter_func=None):
    statements = []
    html = await fetch(url)
    if not html:
        return statements
    soup = BeautifulSoup(html, "html.parser")
    statement_divs = soup.find_all("div", class_="s-statement")

    for statement_div in statement_divs:
        try:
            # statement link
            accordion_div = statement_div.find("div", class_="accordion")
            link = ""

            if accordion_div:
                link = accordion_div.findChildren("div", recursive=False)[2].select(
                    "a"
                )[1]["href"]
            else:
                link = "N/A"

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
            statementText = blockquote_div.select("span")[1].get_text(strip=True)

            # speaker url
            speakerLink = statement_div.find(
                "a", {"data-sentry-component": "SpeakerLink"}
            )["href"]

            statement = {
                'id':next(stmtCnter),
                'link':link,
                'date':date,
                'assessment':assessment,
                'statement':statementText,
                'speaker':speakerLink,
            }

            # Apply the filter if provided
            if not filter_func or filter_func(statement):
                statements.append(statement)

        except (IndexError, AttributeError) as e:
            print(f"Error processing statement at {url}: {e}", file=sys.stderr)

    return statements


async def scrapeStatements(from_page=1, to_page=300, filterFunc=None):
    start_time = time.time()
    statements = []

    coros = [
        scrapeStatementsFromPage(f"{statements_url}?page={page}", filterFunc)
        for page in range(from_page, to_page)
    ]

    pages = await asyncio.gather(*coros)


    for page in pages:
        statements.extend(page)

    end_time = time.time()
    print(f"Scraped Demagog in {end_time - start_time:.2f} seconds")
    return statements
