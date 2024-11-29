"""
@file: build_dataset.py
@author: Hai Phong Nguyen

Build a dataset of statements with their context
This script scrapes statements from Demagog.cz and provides context for each statement.
"""

import scrape_statements as Demagog
import json
import bs4
import time
from utils import *
import asyncio as asyncio
import aiofiles
import os
import itertools
from gpt4all import GPT4All
from collections import defaultdict
import contextvars
import shutil

search_sem = contextvars.ContextVar("search_sem")
fetch_sem = contextvars.ContextVar("fetch_sem")

def generate_context_query(stmtText):
    """
    Generate a context query for a given statement text
    """

    query = config["ContextLLMQuery"] if "ContextLLMQuery" in config else "Vytvoř český google dotaz pomocí klíčových slov pro získání relevantního kontextu, který bude sloužit k ověření následujícího tvrzení:"

    if config["ContextLLM"] == "llama":
        model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        with model.chat_session():
            return model.generate(query + stmtText)
    else:
        return stmtText

async def extract_article(url):
    """
    Extract article content from a given URL
    Heuristitc: Find the parent element with the most <p> children
    """
    if config["ContextLinkOnly"]:
        return {
            "url": url,
        }
    else:
        async with fetch_sem.get():
            html = await fetch(url)
            if not html:
                return None
            await asyncio.sleep(config['FetchDelay'])

        MIN_NUM_WORDS = 5
        try:
            soup = bs4.BeautifulSoup(html, "html.parser", parse_only=bs4.SoupStrainer(["title","body"]))
        except Exception as e:
            print(f"Failed to parse HTML for {url}: {e}", file=sys.stderr)
            return None

        p_parents = defaultdict(list)
        for p in soup.find_all("p"):
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

async def scrape_context(query):
    """
    Search for context articles for a given query and return their content
    Uses Criteria API, Bing API or Google search, based on config
    """
    if config["SearchAPI"] == 'criteria':
        data = dict(claim=query)
        async with search_sem.get():
            await asyncio.sleep(config['SearchDelay'])
            results = await post_request("https://lab.idiap.ch/criteria/search_tom", data)

        return [{
            'url': result['url'],
            'title': result['title'],
            'content': result['fulltext']
            } for result in results
        ]
    else:
        num_results = config["ContextNumBuffer"]
        if num_results == 0:
            return []

        urls = []

        async with search_sem.get():
            if(config["SearchAPI"] == 'google'):
                # Google search
                    search_adjusters = " -filetype:epub -filetype:mobi -filetype:pdf -filetype:csv -filetype:xlsx -filetype:doc -filetype:docx -filetype:ppt -filetype:pptx -filetype:txt -site:demagog.cz -site:facebook.com -site:reddit.com -site:instagram.com -site:x.com"
                    query += search_adjusters
                    urls = await search_google(query, num_results)

            elif(config["SearchAPI"] == 'bing'):
                # Bing search
                    query += " -site:demagog.cz -site:facebook.com -site:reddit.com -site:instagram.com -site:x.com"
                    result = await search_bing(query, config["BingAPIKey"], num_results)
                    urls = [item['url'] for item in result]

            await asyncio.sleep(config['SearchDelay'])

        context = []
        articles_counter = itertools.count(1)
        article_coros = [ track_progress(extract_article(url), articles_counter, config['ContextNum'], 'article') for url in urls]

        print("Extracting grounding context articles...")
        articles = await asyncio.gather(*article_coros)
        if not articles:
            print(f"Warning: No articles found for query: {query}", file=sys.stderr)
            return []

        for article in articles:
            if len(context) >= config["ContextNum"]:
                break
            if article and article['content']:
                context.append(article)

        return context


async def provide_context(stmt):
    """
    Provide context for a given statement
    """

    query = generate_context_query(stmt['statement'])
    context = await scrape_context(query)

    async with aiofiles.open(f"{config['OutputDir']}/context/{stmt['id']}.json", mode="a") as file:
        try:
            await file.write(json.dumps(context, ensure_ascii=False, indent=4) + ",\n")
        except TypeError as e:
            print(f"Failed to write context for statement: {e}", file=sys.stderr)
            for article in context:
                print(type(article))


async def build_dataset(stmts):
    """
    Build a dataset from a list of statements
    Saves pruned statements to file and provides context for each statement
    Each statement's context is saved to a separate file named by the statement's ID
    """

    # create output directory
    out_dir = config['OutputDir']
    if(os.path.exists(out_dir) and config["OverwriteOutput"]):
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        os.makedirs(f"{out_dir}/context", exist_ok=True)

    # include only fields enabled in config
    stmts_pruned = [{k:v for k,v in temp.items() if v} for temp in [{
        "id": stmt['id'],
        "statement": stmt['statement'],
        "author": stmt['speaker'] if config["IncludeAuthor"] else "",
        "date": stmt['date'] if config["IncludeDate"] else "",
        "explanation":  stmt['explanation'] if config["IncludeExplanation"] else "",
        "assessment": stmt['assessment'] if config["IncludeAssessment"] else "",
    } for stmt in stmts]]

    # write pruned statements to file
    with open(f"./{out_dir}/statements.json", "w+") as file:
        json.dump(stmts_pruned, file, ensure_ascii=False, indent=4)

    # provide context for each statement
    stmt_cnter = itertools.count(0)
    context_coros = [
        track_progress(provide_context(stmt), stmt_cnter, total_statements, "statement")
        for stmt in stmts
    ]
    await asyncio.gather(*context_coros)


useExistingStatements = False


async def main():
    global total_statements
    search_sem.set(asyncio.Semaphore(config["SearchesPerDelay"]))
    fetch_sem.set(asyncio.Semaphore(config["FetchesPerDelay"]))

    start_time = time.time()
    stmts = []

    if useExistingStatements:
        with open(f"{config['OutputDir']}/statements.json", "r") as file:
            stmts = json.load(file)
    else:
        stmts = await Demagog.scrapeStatements(from_page=config["DemagogFromPage"],to_page=config["DemagogToPage"])

    total_statements = len(stmts)

    await build_dataset(stmts)

    print(f"Execution time: {time.time() - start_time}")


if __name__ == "__main__":
    asyncio.run(main(), debug=True)

