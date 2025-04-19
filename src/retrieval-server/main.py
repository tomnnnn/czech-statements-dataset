import os
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dataset_manager import Dataset

# === Init ===
app = FastAPI()
model = SentenceTransformer("BAAI/BGE-M3")  # or any other embedding model
dataset = Dataset(os.path.join(os.environ.get("SCRATCHDIR"), "dataset.sqlite"))

# === Request schema ===
class SearchRequest(BaseModel):
    query: str
    statement_id: int
    k: int = 5

# GPU resource (initialized once for reuse)
# gpu_resource = faiss.StandardGpuResources()

async def lazy_load_index(statement_id: int):
    index_path = f"indexes/{statement_id}.faiss"
    segments = dataset.get_segments_by_statement(statement_id)

    # Try to load the index if it exists
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            # gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load index: {e}")
    else:
        # Create a new index and store it
        texts = [segment.text for segment in segments]
        embeddings = model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]

        index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
        # gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        # gpu_index.add(embeddings)
        index.add(embeddings)

        try:
            faiss.write_index(gpu_index, index_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save index: {e}")

    # return gpu_index, segments
    return index, segments

# Initialize shared memory for caching
app.state.index_cache = {}
app.state.data_map = {}

# === Route ===
@app.post("/search")
async def search(req: SearchRequest):
    # Access the cache
    index_cache = app.state.index_cache
    data_map = app.state.data_map

    # If index is not already loaded, load it
    if req.statement_id not in index_cache:
        try:
            index_cache[req.statement_id], data_map[req.statement_id] = await lazy_load_index(req.statement_id)
        except HTTPException as e:
            raise e

    # Retrieve index and data
    index = index_cache[req.statement_id]
    data = data_map[req.statement_id]

    # Encode the query and search the index
    query_vec = model.encode([req.query], convert_to_numpy=True)
    distances, result_indices = index.search(query_vec, req.k)

    # Prepare the results
    results = [
        {"score": float(distances[0][i]), "data": data[idx]}
        for i, idx in enumerate(result_indices[0]) if idx != -1
    ]

    return {"results": results}

