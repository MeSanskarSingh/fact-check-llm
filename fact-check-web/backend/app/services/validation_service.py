import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ValidationService:
    def __init__(self):
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are a fact-checking AI. Use your own intelligence and sources to analyze the following claim and determine:
1. Verdict: Real / Fake / Uncertain
2. Confidence: number between 0 and 1
3. Explanation: short reasoning

Instructions:
Use real/fake when you are sure and uncertain when you have no credibility over the source.

Return STRICT JSON format:

{{
  "verdict": "...",
  "confidence": ...,
  "explanation": "..."
}}

Claim:
{input}
""")

        self.chain = self.prompt | self.llm | StrOutputParser()

    def validate_text(self, text: str):
        try:
            response = self.chain.invoke({"input": text})

            # Try parsing JSON safely
            import json
            data = json.loads(response)

            return {
                "verdict": data.get("verdict", "Uncertain"),
                "confidence": float(data.get("confidence", 0.5)),
                "explanation": data.get("explanation", "No explanation provided")
            }

        except Exception as e:
            print("Validation error:", e)

            # fallback (important)
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
