import os
import time
from dataset_builder.segmenter import segment_article
from fastapi import FastAPI
from pydantic import BaseModel
from dataset_builder.article_scraper import ArticleScraper
from fact_checker.search_functions import HybridSearch
from dataset_manager.models import Article, Segment
from typing import Literal
from aiolimiter import AsyncLimiter
from fact_checker.search_functions.google import GoogleSearch

# === Init ===
app = FastAPI()

# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k: int = 5
    type: Literal["dense", "hybrid"] = "hybrid"

# === Initialize global vars ===
segment_retriever: HybridSearch
document_retriever: GoogleSearch
limiter = AsyncLimiter(5,1)

@app.on_event("startup")
async def startup_event():
    global segment_retriever
    scratch_dir = os.environ.get("SCRATCHDIR", "./")
    dataset_path = os.path.join(scratch_dir, "dataset_demagog.sqlite")
    segment_retriever = HybridSearch(dataset_path, "indices_hybrid_new")
    await segment_retriever.load_indices()

# === Route ===
@app.post("/search")
async def search(req: SearchRequest):
    if req.type == "hybrid":
        results = await segment_retriever.search(req.query, statement_id=req.statement_id, k=req.k)
    else:
        results = await segment_retriever.search_dense(req.query, statement_id=req.statement_id, k=req.k)

    serialized = [r.to_dict(include_relationships=True) for r in results]

    return {"results": serialized}


@app.post("/search-online")
async def search_online(req: SearchRequest):
    """
    For a given query, searcher for k documents using google api. After that, retrieves relevant segments from the document
    """

    cleaned_query = req.query.replace('AND', '').replace('"', '')
    # Get the documents
    # NOTE: hard-coded 10 documents for now
    search_results = await document_retriever.search_async(cleaned_query, 10)

    if not search_results:
        return {
            "query": req.query,
            "k": req.k,
            "results": [],
        }
        
        
    links = [res["link"] for res in search_results]

    # Scrape the documents
    start = time.time()
    try:
        async with limiter:
            articles = await ArticleScraper.scrape_extractus_async(links)

        articles = [article for article in articles if article]
    except RuntimeError as e:
        # Handle the error
        print("Error scraping articles:", e)
        return {"results": []}

    print("Scraping took", time.time() - start, "seconds")

    segments = []
    for article in articles:
        article_obj = Article(
            title=article['title'],
            url=article['url'],
            content="", # leave out content
            published=article['published'],
            author=article['author'],
            source=article['source'],
            description=article['description'],
        )
        article_segments = segment_article(article['content'], min_len=100)

        for segment in article_segments:
            segment_dict = {
                "article": article_obj,
                "text": segment,
            }
            segment_obj = Segment(**segment_dict)
            segments.append(segment_obj)

    # Search for relevant segments
    retrieved_segments = await segment_retriever.search_on_the_fly(
        query=req.query,
        corpus=segments,
        k=req.k,
    )

    # Convert back to dict
    retrieved_segment_dicts = [
            segment.to_dict(True) for segment in retrieved_segments
    ]

    # Return the results
    return {
        "query": req.query,
        "k": req.k,
        "results": retrieved_segment_dicts,
    }


