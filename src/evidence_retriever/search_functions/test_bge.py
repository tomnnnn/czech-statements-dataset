from .bge_m3 import BGE_M3
import asyncio
from dataset_manager import Dataset

dataset = Dataset("datasets/curated.sqlite")

corpus = dataset.get_segments()[:100]
bge = BGE_M3(corpus)

async def search(query):
    results = bge.search(query, k=5)
    
    return results

async def main():
    coroutines = [
        search("What is the capital of France?"),
        search("Who wrote 'To Kill a Mockingbird'?"),
        search("What is the largest mammal?"),
        search("What is the speed of light?"),
        search("What is the capital of Japan?")
    ]

    results = await asyncio.gather(*coroutines)
    for result in results:
        print(result)

asyncio.run(main())
