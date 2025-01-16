"""
@file scrape_statements.py
@author: Hai Phong Nguyen (xnguye28@stud.fit.vutbr.cz)

Script to scrape statements from https://demagog.cz/
"""

from bs4 import BeautifulSoup
import json
from dataclasses import dataclass
import time
import sys
import asyncio as asyncio
import itertools
from selenium import webdriver
from selenium.webdriver.common.by import By
import dateparser
import math
import os
from utils.scraper_utils import fetch, track_progress
import config

BASE_URL = "https://demagog.cz"
STMTS_URL = "https://demagog.cz/vyroky"
CONFIG = config.load_config("scraper_config.yaml")


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
        await asyncio.sleep(CONFIG['FetchDelay'])

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
                explanation = accordion_div.findChildren("div", recursive=False)[0].get_text(strip=True)
            else:
                explanation = ' '.join([p.get_text() for p in statement_div.select(".d-block p")])

            # date
            citation = statement_div.find("cite").get_text(strip=True)
            # last part of the citation is the date
            date = citation.split(",")[-1]
            # parse the date from 12. listopadu 2020 to 2020-11-12
            date = dateparser.parse(date, languages=["cs"]).strftime("%Y-%m-%d")

            tags_container = statement_div.select(".row.g-2")
            tags = []
            if tags_container:
                tag_divs = tags_container[0].find_all("div")
                tags = [tag.find("span").get_text(strip=True) for tag in tag_divs]
                # remove duplicates
                tags = list(dict.fromkeys(tags))
                tags = [tag for tag in tags if tag]

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

            # speaker name
            author_name = statement_div.find("h3").get_text(strip=True)

            statement = {
                'id':next(statement_cnter),
                'statement':statement_text,
                'explanation':explanation,
                'date':date,
                'assessment':assessment,
                'author':author_name,
                'tags':tags if tags else []
            }

            # Apply the filter if provided
            if not filter_func or filter_func(statement):
                statements.append(statement)

        except (IndexError, AttributeError) as e:
            print(f"Error processing statement at {url}: {e}", file=sys.stderr)

    return statements


async def scrapeStatements(from_page=1, to_page=300, filterFunc=None, start_index=1, query=""):
    start_time = time.time()
    statements = []
    index_counter = itertools.count(start_index)
    statement_tracker = itertools.count(1)
    fetch_sem = asyncio.Semaphore(CONFIG["FetchesPerDelay"])

    print(f"Scraping Demagog from page {from_page} to page {to_page}")
    coros = [
        track_progress(scrapeStatementsFromPage(f"{STMTS_URL}?{query}&page={page}", fetch_sem, filterFunc, index_counter), statement_tracker, to_page + 1 - from_page, 'page')
        for page in range(from_page, to_page+1)
    ]

    pages = await asyncio.gather(*coros)

    for page in pages:
        statements.extend(page)

    end_time = time.time()
    print(f"Scraped Demagog in {end_time - start_time:.2f} seconds")
    return statements


def scrapeYears():
    driver = webdriver.Chrome()
    driver.get(STMTS_URL)
    filters_button = driver.find_elements(By.CSS_SELECTOR, ".btn.w-100.h-44px")[0]
    filters_button.click()

    # get years filter
    filters = driver.find_elements(By.CSS_SELECTOR, "div.filter")
    year_filter = filters[-1]

    # scroll to the year filter
    driver.execute_script("arguments[0].scrollIntoView();", year_filter)

    actions = webdriver.ActionChains(driver)
    actions.move_to_element(year_filter)
    actions.perform()

    year_filter.click()

    # get the years
    year_divs = year_filter.find_elements(By.CSS_SELECTOR, ".filter-content")[0].find_elements(By.CSS_SELECTOR, ".check-btn")

    years = [{
        "year": year_div.find_elements(By.CSS_SELECTOR, ".small")[0].text.strip(),
        "count": year_div.find_elements(By.CSS_SELECTOR, ".smallest")[0].text.split(' ')[0]
    } for year_div in year_divs]

    driver.quit()
    return years

async def scrapeByYears(from_year=None, to_year=None, from_page=1, to_page=None):
    # needed to determine the number of pages to scrape
    years = scrapeYears()

    if from_year:
        if not to_year:
            to_year = from_year
        years = [year for year in years if int(year['year']) >= from_year and int(year['year']) <= to_year]

    statements = []
    index_counter = 0
    years_processed = itertools.count(1)

    coros = []
    for year in years:
        coros.append(
            track_progress(
                scrapeStatements(from_page, math.ceil(int(year['count'])/10) if not to_page else to_page, query=f"years={year['year']}", start_index=index_counter), 
                years_processed, 
                len(years), 
                'year')
        )

        index_counter += int(year["count"])

    pages = await asyncio.gather(*coros)

    for page in pages:
        statements.extend(page)

    return statements



if __name__ == "__main__":
    statements = asyncio.run(scrapeByYears(CONFIG["FromYear"], CONFIG["ToYear"], CONFIG["DemagogFromPage"], CONFIG["DemagogToPage"]))

    os.makedirs(CONFIG['OutputDir'], exist_ok=True)
    with open(f"{CONFIG['OutputDir']}/statements.json", "w") as f:
        json.dump(statements, f, indent=2)

