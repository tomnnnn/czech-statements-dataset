import asyncio
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
from src.dataset_manager.models import Segment
from .bge_m3 import BGE_M3
from sklearn.model_selection import train_test_split
from dataset_manager import Dataset


dataset = Dataset("datasets/dataset_demagog.sqlite")

all_statements = dataset.get_statements()
labels = [statement.label for statement in all_statements]

_, statements = train_test_split(all_statements, test_size=0.1, random_state=42, stratify=labels)

print(f"Loaded {len(statements)} statements")

segments = {
    statement.id: [segment for article in statement.articles for segment in dataset.get_segments(article.id)]
    for statement in statements
}

print(f"Prepared segments for {len(segments)} statements")

os.makedirs("indexes", exist_ok=True)

# This function must be at the top-level for multiprocessing to work correctly
def run_bge_m3(statement_id, corpus):
    print(f"[PID:{os.getpid()}] Precalculating embeddings for statement id: {statement_id}")
    BGE_M3(corpus, save_index=True, index_path=f"indexes/{statement_id}.faiss")


async def main():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, run_bge_m3, sid, corpus)
            for sid, corpus in segments.items()
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    import os
    asyncio.run(main())
