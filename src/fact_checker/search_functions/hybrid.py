import os
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from dataset_manager import Dataset
from pymilvus import (
    MilvusClient,
    DataType,
    Collection,
)

class HybridSearch():
    col: Collection

    def __init__(self, db_uri: str, col_name: str):
        self.ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
        self.rf = BGERerankFunction(
            model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
            device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        )

        self.client = MilvusClient(uri=db_uri)
        self.col_name = col_name

    def create_index(self):
        dense_dim = self.ef.dim["dense"]
        schema = self.client.create_schema()
        schema.add_field("pk", DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field("statement_id", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=10000)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        
        self.client.create_collection(self.col_name, schema=schema)
        self.client.create_index(self.col_name, index_params=index_params)

    def insert_segments(self):
        dataset = Dataset(os.path.join(os.environ.get("SCRATCHDIR", "datasets"), "dataset.db"))
        entries = dataset.get_segments_with_statement_ids()

        statement_ids = [e[0] for e in entries]
        segments = [e[1] for e in entries]
        segment_texts = [seg.text for seg in segments]

        docs_embeddings = self.ef(segment_texts)

        for i in range(0, 20, 20):
            batched_entities = [
                {
                    "statement_id": statement_id,
                    "text": text,
                    "sparse_vector": sparse,
                    "dense_vector": dense,
                }
                for statement_id, text, sparse, dense in zip(
                    statement_ids[i : i + 50],
                    segment_texts[i : i + 50],
                    docs_embeddings["sparse"][i : i + 50],
                    docs_embeddings["dense"][i : i + 50],
                )
            ]

            self.client.insert(self.col_name, batched_entities)

    def search(
        self,
        query,
        limit=10,
        k=10
    ):
        query_embedding = self.ef.encode_queries([query])

        dense_results = self.client.search(
            self.col_name,
            data=query_embedding["dense"],
            limit=limit,
            params={"metric_type": "IP"},
            output_fields=["text"]
        )[0]

        sparse_results = self.client.search(
            self.col_name,
            data=query_embedding["sparse"],
            limit=limit,
            params={"metric_type": "IP"},
            output_fields=["text"]
        )[0]

        # Combine the results
        results = []
        for dense,sparse in zip(dense_results, sparse_results):
            results.append(dense["text"])
            results.append(sparse["text"])

        # Combine the results using the BGERerankFunction
        reranked_results = self.rf(
            query=query,
            documents=results,
            top_k=k,
        )

        reranked_texts = [
            doc.text for doc in reranked_results
        ]
        return reranked_texts
