from dataset_manager import DemagogDataset
import csv
import json
from .segment import segment_article

if __name__ == "__main__":
    dataset = DemagogDataset("datasets/curated.sqlite", "demagog")
    evidence = dataset.get_all_evidence()

    segments = []
    for article in evidence:
        segments.extend(
            {"article_id": article["id"], **segment} 
            for segment in segment_article(article["content"]) if segment is not None
        )

    segments = [segment for segment in segments if segment["text"]]

    # Save segments to json
    with open("segments.json", "w") as f:
        json.dump(segments, f, indent=4, ensure_ascii=False)

    # save segments to csv
    with open("segments.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["article_id", "tag", "text", "raw"])
        writer.writeheader()
        writer.writerows(segments)
