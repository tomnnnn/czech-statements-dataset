from googlesearch import search
import scrape_statements as Demagog
import json
import bs4
import time
from utils import fetch, config
import asyncio as asyncio
import aiofiles


async def provideGroundingContext(stmt, outputFile):
    print(f"Processing statement: {stmt.statement}")
    statementExtended = {
        "statement": stmt.statement,
        "context": [],
        "author": config["IncludeAuthor"] if stmt.speaker else "",
        "date": config["IncludeDate"] if stmt.date else "",
        "source": config["IncludeSource"] if stmt.link else "",
        "assessment": config["IncludeAssessment"] if stmt.assessment else "",
    }

    # exclude demagog.cz from search results
    searchQuery = f"{stmt.statement} -filetype:epub -filetype:mobi -filetype:pdf -filetype:csv -filetype:xlsx -filetype:doc -filetype:docx -filetype:ppt -filetype:pptx -filetype:txt -site:demagog.cz"

    async def processGroundingContext(url):
        if config["ContextLinkOnly"]:
            return {
                "url": url,
            }
        else:
            html = await fetch(url)
            if not html:
                return None
            strainer = bs4.SoupStrainer(
                ["title", "p", "h1", "h2", "h3", "h4", "h5", "h6"]
            )
            soup = bs4.BeautifulSoup(html, "html.parser", parse_only=strainer)
            titleEl = soup.find("title")
            title = titleEl.get_text().strip() if titleEl else ""

            return {
                "url": url,
                "title": title,
                "text": "\n".join([p.get_text().strip() for p in soup.find_all("p")]),
            }

    groundingContexts = []
    coros = [
        processGroundingContext(url)
        for url in search(searchQuery, config["ContextNum"], sleep_interval=1)
    ]

    for groundingContext in await asyncio.gather(*coros):
        if groundingContext:
            groundingContexts.append(groundingContext)

    statementExtended["context"] = groundingContexts

    # remove empty fields
    statementExtended = {k: v for k, v in statementExtended.items() if v}

    # async with aiofiles.open(outputFile, mode="a") as file:
    #     await file.write(json.dumps(statementExtended, ensure_ascii=True) + ",\n")

    return statementExtended


async def buildDataset(stmts, outputFile="dataset.json"):
    coros = [provideGroundingContext(stmt, outputFile) for stmt in stmts]
    stmts = await asyncio.gather(*coros)

    with open(outputFile, "w") as file:
        json.dump(stmts, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    start_time = time.time()

    stmts = asyncio.run(Demagog.scrapeStatements(to_page=2))
    dataset = asyncio.run(buildDataset(stmts))

    print(f"Execution time: {time.time() - start_time}")
