import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from dataset_manager import Dataset
from fact_checker.search_functions import HybridSearch
from typing import Literal
from contextlib import asynccontextmanager

# === Init ===
app = FastAPI()

# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k: int = 5
    type: Literal["dense", "hybrid"] = "hybrid"

# === Initialize global vars ===
index_manager: HybridSearch

@app.on_event("startup")
async def startup_event():
    global index_manager
    scratch_dir = os.environ.get("SCRATCHDIR", "./")
    dataset_path = os.path.join(scratch_dir, "dataset_demagog.sqlite")
    index_manager = HybridSearch(dataset_path, "indices_hybrid_new")
    await index_manager.load_indices()

# === Route ===
@app.post("/search")
async def search(req: SearchRequest):
    if req.type == "hybrid":
        results = await index_manager.search(req.query, statement_id=req.statement_id, k=req.k)
    elif req.type == "dense":
        results = await index_manager.search_dense(req.query, statement_id=req.statement_id, k=req.k)

    serialized = [r.to_dict(include_relationships=True) for r in results]

    return {"results": serialized}
