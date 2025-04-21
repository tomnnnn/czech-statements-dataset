from dataset_builder.segmenter import segment_article
from dataset_builder.article_scraper import ArticleScraper
from dataset_manager.models import Article, Segment
from fact_checker.search_functions.google import GoogleSearch
from fact_checker.search_functions.bge_m3 import BGE_M3
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# === Init ===
app = FastAPI()
model = SentenceTransformer("BAAI/BGE-M3")  # or any other embedding model

# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k_docs: int = 10
    k_segments: int = 5

# === Retrievers ===
segment_retriever = BGE_M3(model=model)
document_retriever = GoogleSearch()

# === Route ===
@app.post("/search")
@app.get("/search")
async def search(req: SearchRequest):
    """
    For a given query, searcher for k documents using google api. After that, retrieves relevant segments from the document
    """
    # Get the documents
    search_results = await document_retriever.search_async(req.query, req.k_docs)
    links = [res["link"] for res in search_results] # TODO: adjust

    # Scrape the documents
    try:
        articles = await ArticleScraper.scrape_extractus_async(links)
        articles = [article for article in articles if article]
    except RuntimeError as e:
        # Handle the error
        return {"error": str(e)}



    # TODO: replace segments with retrieved segments
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
        article_segments = segment_article(article['content'])

        for segment in article_segments:
            segment_dict = {
                "article": article_obj,
                "text": segment,
            }
            segment_obj = Segment(**segment_dict)
            segments.append(segment_obj)

    # Convert segments to Segment objects
    segment_dicts = [segment.to_dict(True) for segment in segments]

    # Index the segments
    await segment_retriever.add_index(segments, save=False, load_if_exists=False, key=req.statement_id)

    # Search for relevant segments
    retrieved_segments = await segment_retriever.search_async(req.query, k=req.k_segments, key=req.statement_id)

    # Return the results
    return {
        "results": retrieved_segments,
    }
