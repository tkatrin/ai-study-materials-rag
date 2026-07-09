from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Chunk]
