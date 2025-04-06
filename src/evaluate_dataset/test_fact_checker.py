from dataset_manager.models import Statement
from dataset_manager import Dataset
from .fact_checker import FactChecker
from .evaluator import FactCheckingEvaluator
from .llm_apis.mock_api import MockLanguageModelAPI
from evidence_retriever.retrievers import MockRetriever
from .config import Config
import pprint


corpus = [
    {"id": 1, "text": "This is a test document."},
    {"id": 2, "text": "This is another test document."},
    {"id": 3, "text": "This is a third test document."},
    {"id": 4, "text": "This is a fourth test document."},
    {"id": 5, "text": "This is a fifth test document."},
    {"id": 6, "text": "This is a sixth test document."},
    {"id": 7, "text": "This is a seventh test document."},
    {"id": 8, "text": "This is an eighth test document."},
    {"id": 9, "text": "This is a ninth test document."},
    {"id": 10, "text": "This is a tenth test document."},
]

template = """
{statement} - {author}, {date}

Evidence:
{evidence}
"""

config = Config("test","ahoj",0,1,False,"prompts/unveri.yaml",0,1,1, min_evidence_count=0)

llm = MockLanguageModelAPI("mock")
retriever = MockRetriever('bge-m3', corpus, 3)
fc = FactChecker(llm, retriever, config, template)

stmt =  {
    "id": 1,
    "statement": "This is a test statement.",
    "author": "John Doe",
    "date": "2023-10-01",
    "label": "pravda",
}

dataset = Dataset(":memory:")
dataset.insert_statement(stmt)

evaluator = FactCheckingEvaluator(dataset, fc, config)

results = evaluator.run()
