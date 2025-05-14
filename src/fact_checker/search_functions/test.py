from .hybrid import HybridSearch
import asyncio
import os

async def main():
    retriever = HybridSearch(os.path.join(os.environ.get("SCRATCHDIR"), "dataset_demagog.sqlite"), "index_merged")
    # await retriever.create_indices()
    await retriever.load_indices()

    query = "STAN obvineni politici v kauze Dozimetr"
    results = await retriever.search(query, "merged")
    results = [i.text for i in results]

    while True:
        # Accept user input from stdin
        query = input("Enter your query (or type 'exit' to quit): ").strip()

        if query.lower() == 'exit':
            break

        # Perform the search
        results = await retriever.search(query, "merged")
        results = [i.text for i in results]

        # Print the results
        if results:
            for r in results:
                print(r, end="\n\n")
        else:
            print("No results found.\n")

        for r in results:
            print(r, end="\n\n")

asyncio.run(main())
