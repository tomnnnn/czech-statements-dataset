import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dataset_manager import Dataset
from fact_checker.search_functions import BGE_M3

# === Init ===
app = FastAPI()
model = SentenceTransformer("BAAI/BGE-M3")  # or any other embedding model
model.half()
scratch_dir = os.environ.get("SCRATCHDIR", "./")
dataset = Dataset(os.path.join(scratch_dir, "dataset.sqlite"))

# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k: int = 5

class UnloadRequest(BaseModel):
    statement_id: int

# Initialize shared memory for caching
app.state.index_cache = {}
app.state.data_map = {}
index_manager = BGE_M3(model=model)

# === Route ===
@app.post("/search")
async def search(req: SearchRequest):
    if not index_manager.key_exists(req.statement_id):
        start = time.time()
        segments = dataset.get_segments_by_statement(req.statement_id)
        print("Getting segments took:", time.time() - start, " seconds")
        await index_manager.add_index(segments, f"indices/{req.statement_id}.faiss",load_if_exists=True,save=True, key=req.statement_id)

    
    results = await index_manager.search_async(req.query, k=req.k, key=req.statement_id)
    serialized = [r.to_dict(include_relationships=True) for r in results]

    index_manager.unload_index(req.statement_id)
    return {"results": serialized}
