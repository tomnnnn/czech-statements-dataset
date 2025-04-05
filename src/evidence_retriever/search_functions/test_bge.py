from .bge_m3 import BGE_M3
from .bm25 import BM25
from dataset_manager.orm import *
import pprint

dataset = init_db()

corpus = rows2dict(dataset.query(Segment).all())[:20]
query = "Co je kauza Dozimetr?"

print("Loading BGE_M3...")
search_engine = BGE_M3(corpus, vector_type="lexical_weights")
print("Loaded BGE_M3")

results = search_engine.search(query, k=3)
pprint.pp(results)
