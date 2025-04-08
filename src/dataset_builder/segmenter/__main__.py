from dataset_manager import Dataset
from .segment import segment_article
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment articles into smaller segments.")
    parser.add_argument("-d", "--dataset_path", type=str, help="Path to the dataset file")
    args = parser.parse_args()

    dataset = Dataset(args.d)
    articles = dataset.get_articles()

    segments = []
    for article in articles:
        segments.extend(
            {"article_id": article.id, "text": segment} 
            for segment in segment_article(article.content) if len(segment) > 25
        )

    dataset.insert_segments(segments)
