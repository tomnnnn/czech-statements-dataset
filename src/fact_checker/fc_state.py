from typing import TypedDict

class FactCheckingResult(TypedDict):
    statement_id: int
    statement: str
    author: str
    date: str
    evidence: list[dict]
    label: str
    metadata: dict
