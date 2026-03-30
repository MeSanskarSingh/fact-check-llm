from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal


class ProcessResponse(BaseModel):
    verdict: Literal["True", "False", "Uncertain"]
    confidence: float = Field(ge=0, le=1)
    extractedText: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    cluster: Optional[int] = None
