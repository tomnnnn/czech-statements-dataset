from typing import TypedDict

class ContextualizedStatement(TypedDict):
    id: int
    statement: str
    evidence: list[dict]

