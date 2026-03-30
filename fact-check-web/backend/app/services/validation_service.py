import os
from dotenv import load_dotenv
from typing import Literal

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

import json
import re

load_dotenv()


# =========================
# ✅ Schema
# =========================
class ValidationResult(BaseModel):
    verdict: Literal["True", "False", "Uncertain"] = Field(
        description="Final classification of the claim"
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score between 0 and 1"
    )
    explanation: str = Field(
        description="Short reasoning explaining the decision"
    )


# =========================
# ✅ Service
# =========================
class ValidationService:
    def __init__(self):
        self.llm = ChatMistralAI(temperature=0)

        self.prompt = ChatPromptTemplate.from_template("""
You are a fact-checking AI.

Analyze the claim and determine:
1. Verdict: True / False / Uncertain
2. Confidence: number between 0 and 1
3. Explanation: short reasoning

IMPORTANT:
- Return ONLY raw JSON
- DO NOT wrap response in ```json or ```
- DO NOT add any extra text
- Use EXACT keys:
  verdict, confidence, explanation
- verdict MUST be exactly one of: True, False, Uncertain

Claim:
{input}
""")

        self.chain = self.prompt | self.llm

    # =========================
    # 🔥 Core Function
    # =========================
    def validate_text(self, text: str):
        try:
            # 🔹 Call LLM
            raw_response = self.chain.invoke({"input": text})
            content = raw_response.content.strip()

            print("RAW LLM OUTPUT:", content)

            # =========================
            # 🔥 Extract JSON safely
            # =========================
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in LLM response")

            json_str = match.group(0)

            # =========================
            # 🔥 Parse JSON
            # =========================
            data = json.loads(json_str)

            # =========================
            # 🔥 Normalize keys
            # =========================
            data = {k.lower(): v for k, v in data.items()}

            # =========================
            # 🔥 Normalize values
            # =========================
            if isinstance(data.get("verdict"), str):
                data["verdict"] = data["verdict"].strip().capitalize()

            if "confidence" in data:
                data["confidence"] = float(data["confidence"])

            # =========================
            # 🔥 Validate schema
            # =========================
            parsed = ValidationResult(**data)

            return {
                "verdict": parsed.verdict,
                "confidence": parsed.confidence,
                "explanation": parsed.explanation
            }

        except Exception as e:
            import traceback
            traceback.print_exc()

            return {
                "verdict": "Uncertain",
                "confidence": 0.5,
                "explanation": "Validation failed, fallback used"
            }


# =========================
# ✅ Singleton
# =========================
validation_service = None

def get_validation_service():
    global validation_service
    if validation_service is None:
        validation_service = ValidationService()
    return validation_service