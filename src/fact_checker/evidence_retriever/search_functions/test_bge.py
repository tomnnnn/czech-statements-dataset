from .bge_m3 import BGE_M3
import asyncio
from dataset_manager import Dataset

dataset = Dataset("datasets/curated.sqlite")

corpus = dataset.get_segments()
bge = BGE_M3(corpus)

async def search(query):
    results = bge.search(query, k=5)
    
    return results

async def main():
    coroutines = [
        search("Kdo ze STANu je obvineny v kauze Dozimetr?"),
        search("Fatah a Hamas"),
        search("Lisabonska smlouva, jednani v parlamentu"),
    ]

    results = await asyncio.gather(*coroutines)
    for result in results:
        texts = [r.text for r in result]
        print(texts)

asyncio.run(main())
