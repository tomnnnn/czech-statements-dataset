"""
@file scrape_statements.py
@author: Hai Phong Nguyen (xnguye28@stud.fit.vutbr.cz)

Script to scrape statements from https://demagog.cz/
"""

from bs4 import BeautifulSoup
import json
import time
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


class Statement:
    date: str
    assessment: str
    statement: str
    author: str
    explanation: str
    tags: list
    origin: str

    def __init__(self, statement_div):
        self.statement_div = statement_div
        self.selectors = {
            "citation": "div > div:nth-child(1) > div > div.ps-5 > cite",
            "statement": "div > div:nth-child(2) > div.accordion > div.content.fs-6",
            "author": "div > div:nth-child(1) > div > div.w-100px.min-w-100px > div.mt-2.text-center.w-100 > h3",
            "tags": "div > div.ps-5 > div",
            "assessment": "div > div:nth-child(2) > div.d-flex.align-items-center.mb-2 > span.text-primary.fs-5.text-uppercase.fw-600",
            "explanation": "div > div:nth-child(2) > div.accordion",
            "explanation_alt": "div > div:nth-child(2) > div.d-block",
            "date": "div:nth-child(1) > div > div:nth-child(1) > div > div.ps-5 > cite",
        }
        self.__parse()


    def get_dict(self):
        return {
            "assessment": self.assessment,
            "statement": self.statement,
            "author": self.author,
            "explanation": self.explanation,
            "date": self.date,
            "origin": self.origin,
            "tags": self.tags
        }


    def __parse(self):
       self.__parse_citation()
       self.__parse_statement()
       self.__parse_assessment()
       self.__parse_explanation()
       self.__parse_tags()
       self.__parse_author()


    def __parse_citation(self):
        citation = self.statement_div.select(self.selectors["citation"])[0].get_text(strip=True)
        # last part of the citation is the date
        date = citation.split(",")[-1]
        # the rest is the origin
        origin = ', '.join(citation.split(",")[:-1])

        # parse the date from dd. m yyyy to YYY-mm-dd
        parsed_date = dateparser.parse(date, languages=["cs"])
        parsed_date = parsed_date.strftime("%Y-%m-%d") if parsed_date else "Unknown"

        if(parsed_date == "Unknown"):
            print(f"WARNING: Failed to parse date: {date}")

        self.date = parsed_date
        self.origin = origin

    def __parse_statement(self):
        statement = self.statement_div.select(self.selectors["statement"])[0].get_text(strip=True)
        # remove any words that contain "demagog" in them
        statement = ' '.join([word for word in statement.split() if "demagog" not in word.lower()])
        # notes by demagog are written as (pozn. Demagog), by removing words containing string demagog, 
        # we remove the closing parenthesis, so we need to put it back
        statement = statement.replace("pozn.", "pozn.)")

        self.statement = statement

    def __parse_assessment(self):
        assessment_div = self.statement_div.find( "div", {"data-sentry-component": "StatementAssessment"})
        assessment = assessment_div.findChildren("span", recursive=False)[1].get_text(strip=True)
        self.assessment = assessment

    def __parse_explanation(self):
        explanation_container = self.statement_div.select(self.selectors["explanation"])

        if explanation_container:
            explanation = explanation_container[0].findChildren("div", recursive=False)[0].get_text(strip=True)
        else:
            explanation = ' '.join([p.get_text() for p in self.statement_div.select(self.selectors["explanation_alt"])[0].find_all("p")])
        self.explanation = explanation

    def __parse_tags(self):
        tags_container_selector = "div > div:nth-child(1) > div > div.ps-5 > div > div"

        tags_container = self.statement_div.select(tags_container_selector)
        tags = []
        if tags_container:
            tags = [span.get_text(strip=True) for span in tags_container[0].find_all("span")]
        self.tags = tags

    def __parse_author(self):
        author = self.statement_div.select(self.selectors["author"])[0].get_text(strip=True)
        self.author = author


async def scrapeStatementsFromPage(url, fetch_sem, statement_cnter=itertools.count(1)):
    async with fetch_sem:
        html = await fetch(url)
        await asyncio.sleep(CONFIG['FetchDelay'])

    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    statement_divs = soup.find_all("div", class_="s-statement")
    statements = [{"id":next(statement_cnter), **Statement(statement_div).get_dict()} for statement_div in statement_divs]

    return statements


async def scrapeStatements(from_page=1, to_page=300, start_index=1, year=None):
    """
    Scrape statements from demagog.cz with given GET filter query

    Arguments:
        from_page (int): The starting page to scrape
        to_page (int): The ending page to scrape
        start_index (int): The starting index to be used for saved statements
        query (str): The GET query string to filter the statements

    Returns:
        list: List of statements scraped
    """
    start_time = time.time()
    statements = []
    index_counter = itertools.count(start_index)
    finished_cnter = itertools.count(1)
    fetch_sem = asyncio.Semaphore(CONFIG["FetchesPerDelay"])
    query = f"years={year}" if year else ""

    print(f"Scraping {year if year else ''} statements from page {from_page} to page {to_page}")
    coros = [
        track_progress(
            scrapeStatementsFromPage(
                f"{STMTS_URL}?{query}&page={page}",
                fetch_sem,
                index_counter
            ), 
            finished_cnter, # counter for statements
            to_page + 1 - from_page, # total number of pages
            'page' # unit of progress 
        )
        for page in range(from_page, to_page+1)
    ]

    pages = await asyncio.gather(*coros)

    for page in pages:
        statements.extend(page)

    end_time = time.time()
    print(f"Scraped Demagog in {end_time - start_time:.2f} seconds")
    return statements


def scrapeYears():
    """
    Scrapes available years from the Demagog page

    Returns:
        list of (year, count) entries for each available year.
    """
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

    # expand years filter
    year_filter.click()

    # get the years
    year_divs = year_filter.find_elements(By.CSS_SELECTOR, ".filter-content")[0].find_elements(By.CSS_SELECTOR, ".check-btn")

    years = [{
        "year": int(year_div.find_elements(By.CSS_SELECTOR, ".small")[0].text.strip()),
        "count": int(
            year_div.find_elements(By.CSS_SELECTOR, ".smallest")[0] # element containing number of statements 
            .text.split(' ')[0] # first part is the number
            .replace(',', '')
        )
    } for year_div in year_divs]

    driver.quit()
    return years

def loadYears():
    """
    Load the years from a file

    Returns:
        list of (year, count) entries for each available year.
    """
    if os.path.exists(f"{CONFIG['OutputDir']}/years.json"):
        with open(f"{CONFIG['OutputDir']}/years.json", "r") as f:
            return json.load(f)
    else:
        years = scrapeYears()
        os.makedirs(CONFIG['OutputDir'], exist_ok=True)
        with open(f"{CONFIG['OutputDir']}/years.json", "w") as f:
            json.dump(years, f, indent=2, ensure_ascii=False)

        return years

async def scrapeByYears(from_year=-1, to_year=-1, first_npages=None):
    """
    Scrape statements from demagog.cz for the given years

    Args:
        from_year (int): The starting year to scrape, if -1 then scrape from the first year
        to_year (int): The ending year to scrape, if -1 then scrape until the last year
        first_npages (int): Number of pages to scrape for each year

    Returns:
        list: List of statements scraped
    """

    # needed to determine the number of pages to scrape
    years = loadYears()
    years.sort(key=lambda x: int(x['year']))

    from_year = from_year if from_year > 0 else int(years[0]['year'])
    to_year = to_year if to_year > 0 else int(years[-1]['year'])

    if(from_year > to_year):
        raise AssertionError("scrapeByYears(): from_year argument cannot be higher than to_year argument")

    years = [year for year in years if int(year['year']) >= from_year and int(year['year']) <= to_year]

    statements = []
    index_counter = 0
    years_processed = itertools.count(1)

    coros = []
    for year in years:
        coros.append(
            track_progress(
                scrapeStatements(1, math.ceil(int(year['count'])/10) if not first_npages else first_npages, year=year['year'], start_index=index_counter), 
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
    output_path = f"{CONFIG['OutputDir']}/statements.json"
    os.makedirs(CONFIG['OutputDir'], exist_ok=True)

    if os.path.exists(output_path):
        # ask user if they want to overwrite the file
        print(f"File {output_path} already exists, do you want to overwrite it? (y/n)")
        response = input()
        if response.lower() != "y":
            print("Exiting without saving")
            exit()

    statements = asyncio.run(scrapeByYears(CONFIG["FromYear"], CONFIG["ToYear"], CONFIG["DemagogToPage"]))

    with open(output_path, "w") as f:
        json.dump(statements, f, indent=2, ensure_ascii=False)

