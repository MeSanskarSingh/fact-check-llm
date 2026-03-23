from pydantic import BaseModel
from typing import Optional


class ProcessResponse(BaseModel):
    verdict: str
    confidence: float
    extractedText: str
    explanation: str
    cluster: Optional[int] = None