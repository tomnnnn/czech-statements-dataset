from dataset_builder.segmenter import segment_article
from dataset_builder.article_scraper import ArticleScraper
from dataset_manager.models import Article, Segment
from fact_checker.search_functions.google import GoogleSearch
from fact_checker.search_functions.bge_m3 import BGE_M3
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import time
import torch
from aiolimiter import AsyncLimiter


# === Init ===
app = FastAPI()
model = None
segment_retriever: BGE_M3
document_retriever: GoogleSearch
limiter = AsyncLimiter(5,1)


@app.on_event("startup")
async def load_model():
    global segment_retriever
    global document_retriever

    segment_retriever = BGE_M3(64,20,20)
    document_retriever = GoogleSearch()


# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k: int = 10

# === Retrievers ===

# === Route ===
@app.post("/search")
async def search(req: SearchRequest):
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

    # Index the segments
    await segment_retriever.add_index(segments, save=False, load_if_exists=False, key=req.statement_id, metric="ip")

    # Search for relevant segments
    retrieved_segments = await segment_retriever.search_async(req.query, k=req.k, key=req.statement_id, num_neighbors=2)

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


@app.post("/search_l2")
async def search_l2(req: SearchRequest):
    """
    For a given query, searcher for k documents using google api. After that, retrieves relevant segments from the document
    """
    # Get the documents
    # NOTE: hard-coded 10 documents for now
    search_results = await document_retriever.search_async(req.query, 10)

    if not search_results:
        return {
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

    # Index the segments
    await segment_retriever.add_index(segments, save=False, load_if_exists=False, key=req.statement_id, metric="l2")

    # Search for relevant segments
    retrieved_segments = await segment_retriever.search_async(req.query, k=req.k, key=req.statement_id)

    # Convert back to dict
    retrieved_segment_dicts = [
            segment.to_dict(True) for segment in retrieved_segments
    ]


    # Return the results
    return {
        "results": retrieved_segment_dicts,
    }
