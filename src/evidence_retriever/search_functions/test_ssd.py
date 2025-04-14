from .bge_m3 import BGE_M3
import time
import argparse
from dataset_manager import Dataset

parser = argparse.ArgumentParser(description="Test SSD")
parser.add_argument("path" , type=str, help="Path to the corpus")

args = parser.parse_args()

dataset = Dataset("datasets/dataset_demagog.sqlite")

start = time.time()
print("Getting segments from dataset")
corpus = dataset.get_segments()
print(f"Got {len(corpus)} segments in {time.time() - start:.2f} seconds")

start_2 = time.time()
print("Initializing BGE_M3")
search_engine = BGE_M3(corpus, save_index=True, index_path=args.path)
print(f"Initialized BGE_M3 in {time.time() - start_2:.2f} seconds")

query = "Dozimetr STAN Hlubocek souvislosti"

start_3 = time.time()
print(search_engine.search(query))
print(f"Search took {time.time() - start_3:.2f} seconds")



print("Times:")
print(f"Dataset loading time: {time.time() - start:.2f} seconds")
print(f"BGE_M3 initialization time: {time.time() - start_2:.2f} seconds")
print(f"Total time: {time.time() - start:.2f} seconds")
