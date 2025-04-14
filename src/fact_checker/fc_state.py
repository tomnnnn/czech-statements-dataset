from typing import TypedDict
from dataset_manager.models import Statement

class FactCheckingState(TypedDict):
    statement: Statement
    evidence: list[dict]
    label: str
    metadata: dict
