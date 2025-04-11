from .bge_m3 import BGE_M3
from dataset_manager import Dataset

dataset = Dataset("datasets/dataset_demagog.sqlite")

corpus = dataset.get_segments()
corpus = [i for i in corpus if "demagog.cz" not in i.article.url.lower()]


search_engine = BGE_M3(corpus, save_index=True, index_path="demagog_index.faiss")

print("Precalculated embeddings and index")
