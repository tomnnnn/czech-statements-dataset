import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dataset_manager import Dataset
from fact_checker.search_functions import BGE_M3

# === Init ===
app = FastAPI()
model = SentenceTransformer("BAAI/BGE-M3")  # or any other embedding model
scratch_dir = os.environ.get("SCRATCHDIR", "./")
dataset = Dataset(os.path.join(scratch_dir, "dataset.sqlite"))

# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k: int = 5

# GPU resource (initialized once for reuse)
# gpu_resource = faiss.StandardGpuResources()

# Initialize shared memory for caching
app.state.index_cache = {}
app.state.data_map = {}
index_manager = BGE_M3(model=model)

# === Route ===
@app.post("/search")
async def search(req: SearchRequest):
    if not index_manager.key_exists(req.statement_id):
        segments = dataset.get_segments_by_statement(req.statement_id)
        await index_manager.add_index(segments, f"indexes/{req.statement_id}.faiss",load_if_exists=True,save=True, key=req.statement_id)

    
    results = await index_manager.search_async(req.query, k=req.k, key=req.statement_id)

    return {"results": results}


@app.post("/unload")
async def unload_index(statement_id: int):
    if index_manager.key_exists(statement_id):
        index_manager.unload_index(statement_id)
        return {"message": f"Index for statement {statement_id} unloaded."}
    else:
        return {"message": f"No index found for statement {statement_id}."}
