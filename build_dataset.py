from googlesearch import search, sleep
from requests import post
import scrape_statements as Demagog
import json
import bs4
import time
from utils import fetch, post_request, config
import asyncio as asyncio
import aiofiles
import os
import itertools
from gpt4all import GPT4All
from collections import defaultdict

finished_ctnter = itertools.count(0)

def generateGroundingContextQuery(stmtText):
    query =  config["ContextLLMQuery"] if "ContextLLMQuery" in config else "Vytvoř český google dotaz pomocí klíčových slov pro získání relevantního kontextu, který bude sloužit k ověření následujícího tvrzení:"

    if config["ContextLLM"] == "llama":
        model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        with model.chat_session():
            return model.generate(query + stmtText)
    else:
        return stmtText

async def extractArticleFromWebsite(url):
    if config["ContextLinkOnly"]:
        return {
            "url": url,
        }
    else:
        html = await fetch(url)
        if not html:
            return None

        MIN_NUM_WORDS = 5
        soup = bs4.BeautifulSoup(html, "html.parser", parse_only=bs4.SoupStrainer(["body"]))
        p_parents = defaultdict(list)
        for p in soup.find_all("p"):
            p_parents[p.parent].append(p)

        parents_counts = sorted([(parent, len(ps)) for parent, ps in p_parents.items()], key=lambda v: -v[1])
        if not parents_counts:
            return None
        article_dom = parents_counts[0][0]
        article_text = " ".join(p.get_text().strip() for p in article_dom.find_all("p") if len(p.get_text().split()) > MIN_NUM_WORDS)
        title = soup.find('title')

        return {
            "url": url,
            "title": title,
            "content": article_text
        }

async def searchContext(query):
    if config["SearchAPI"] == 'criteria':
        data = dict(claim=query)
        results = await post_request("https://lab.idiap.ch/criteria/search_tom", data)
        return [{
            'url': result['url'],
            'title': result['title'],
            'content': result['fulltext']
            } for result in results
        ]
    else:
        searchAdjusters = " -filetype:epub -filetype:mobi -filetype:pdf -filetype:csv -filetype:xlsx -filetype:doc -filetype:docx -filetype:ppt -filetype:pptx -filetype:txt -site:demagog.cz"
        query += searchAdjusters

        urls = search(query, config["ContextNum"]+5, sleep_interval=config["GoogleSearchSleep"])

        groundingContexts = []
        articlesCoros = [ extractArticleFromWebsite(url) for url in urls]

        emptyArticles = 0
        for article in await asyncio.gather(*articlesCoros):
            if article:
                groundingContexts.append(article)
            else:
                emptyArticles = emptyArticles + 1

        for _ in range(config["ContextNum"], config["ContextNum"] + emptyArticles):
            articlesCoros = [
                extractArticleFromWebsite(url)
                for url in  urls           ]



        return groundingContexts

async def provideGroundingContext(stmt):
    query = generateGroundingContextQuery(stmt['statement'])
    context = await searchContext(query)

    async with aiofiles.open(f"{config['OutputDir']}/context/{stmt['id']}.json", mode="a") as file:
        await file.write(json.dumps(context, ensure_ascii=True) + ",\n")

async def buildDataset(stmts, outputFile="dataset.json"):
    outputDir = config['OutputDir']
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(f"{outputDir}/context", exist_ok=True)

    stmts_pruned = [{k:v for k,v in temp.items() if v} for temp in [{
        "statement": stmt['statement'],
        "author": stmt['speaker'] if config["IncludeAuthor"] else "",
        "date": stmt['date'] if config["IncludeDate"] else "",
        "source":  stmt['link'] if config["IncludeSource"] else "",
        "assessment": stmt['assessment'] if config["IncludeAssessment"] else "",
    } for stmt in stmts]]

    with open(f"./{outputDir}/statements.json", "w+") as file:
        json.dump(stmts_pruned, file, ensure_ascii=False, indent=4)

    groundingContextCoros = [provideGroundingContext(stmt) for stmt in stmts]
    stmts = await asyncio.gather(*groundingContextCoros)

    print(f"{next(finished_ctnter)}/{total_statements}", flush=True)

    with open(outputFile, "w") as file:
        json.dump(stmts, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    global total_statements

    start_time = time.time()
    stmts = asyncio.run(Demagog.scrapeStatements(to_page=2))
    total_statements = len(stmts)

    print(f"0/{total_statements}", flush=True)
    asyncio.run(buildDataset(stmts))

    print(f"Execution time: {time.time() - start_time}")


