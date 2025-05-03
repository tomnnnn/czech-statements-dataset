from .hybrid import HybridSearch
import asyncio
import os

async def main():
    retriever = HybridSearch(os.path.join(os.environ.get("SCRATCHDIR"), "dataset_demagog.sqlite"), "indices_hybrid_new")
    await retriever.create_indices()
    await retriever.load_indices()

    query = "STAN obvineni politici v kauze Dozimetr"
    results = await retriever.search(query, 100)
    results = [i.text for i in results]

    for r in results:
        print(r, end="\n\n")

asyncio.run(main())
