import os
from dotenv import load_dotenv
from typing import Literal

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# ✅ Step 1: Strict Schema
class ValidationResult(BaseModel):
    verdict: Literal["True", "False", "Uncertain"] = Field(
        description="Final classification of the claim. Must be exactly one of: True, False, or Uncertain."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score between 0 and 1 indicating how certain the model is."
    )
    explanation: str = Field(
        description="Short reasoning explaining why the claim is classified this way."
    )


class ValidationService:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=ValidationResult)

        self.llm = ChatMistralAI()

        self.prompt = ChatPromptTemplate.from_template("""
You are a fact-checking AI. Use your own intelligence and sources to analyze the following claim and determine:
1. Verdict: Real / Fake / Uncertain
2. Confidence: number between 0 and 1
3. Explanation: short reasoning

Instructions:
Use real/fake when you are sure and uncertain when you have no credibility over the source.

Return STRICT JSON format:

IMPORTANT:
- Do NOT return anything except valid JSON
- Do NOT add extra text
- Ensure verdict is EXACTLY one of: True, False, Uncertain

Claim:
{input}
""")

        self.chain = self.prompt | self.llm | self.parser

    def validate_text(self, text: str):
        try:
            response: ValidationResult = self.chain.invoke({
                "input": text,
                "format_instructions": self.parser.get_format_instructions()
            })

            return {
                "verdict": response.verdict,
                "confidence": response.confidence,
                "explanation": response.explanation
            }

        except Exception as e:
            print("Validation error:", e)

            return {
                "verdict": "Uncertain",
                "confidence": 0.5,
                "explanation": "Validation failed, fallback used"
            }


validation_service = None

def get_validation_service():
    global validation_service
    if validation_service is None:
        validation_service = ValidationService()
    return validation_service
