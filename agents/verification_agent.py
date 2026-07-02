from typing import Dict, List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from config.settings import settings

class VerificationAgent:
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

Instructions:
- Verify the answer against the provided context.
- Check direct or indirect factual support, unsupported claims, contradictions, and relevance.
- Provide additional details where relevant.
- Respond in exactly this format, without unrelated information:

Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Answer:
{answer}

Context:
{context}

Return only the required verification format."""
            ),
        ]
    )

    def __init__(self):
        """
        Initialize the verification agent with OpenAI.
        """
        self.model = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=300,
        )
        

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    @staticmethod
    def empty_verification(details: str) -> Dict:
        """Create a consistent negative verification result."""
        return {
            "Supported": "NO",
            "Unsupported Claims": [],
            "Contradictions": [],
            "Relevant": "NO",
            "Additional Details": details,
        }

    def generate_prompt(self, answer: str, context: str) -> List[BaseMessage]:
        """
        Generate structured chat messages for answer verification.
        """
        return self.prompt_template.format_prompt(
            answer=answer,
            context=context,
        ).to_messages()

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            field_names = {
                name.casefold(): name
                for name in [
                    "Supported",
                    "Unsupported Claims",
                    "Contradictions",
                    "Relevant",
                    "Additional Details",
                ]
            }
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = field_names.get(key.strip().casefold())
                    value = value.strip()
                    if key:
                        if key in {"Unsupported Claims", "Contradictions"}:
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                                verification[key] = items
                            else:
                                verification[key] = []
                        elif key == "Additional Details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()
            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

            return verification
        except Exception:
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        if unsupported_claims:
            report += f"**Unsupported Claims:** {', '.join(unsupported_claims)}\n"
        else:
            report += f"**Unsupported Claims:** None\n"

        if contradictions:
            report += f"**Contradictions:** {', '.join(contradictions)}\n"
        else:
            report += f"**Contradictions:** None\n"

        report += f"**Relevant:** {relevant}\n"

        if additional_details:
            report += f"**Additional Details:** {additional_details}\n"
        else:
            report += f"**Additional Details:** None\n"

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        context = "\n\n".join([doc.page_content for doc in documents])
        messages = self.generate_prompt(answer, context)
        try:
            response = self.model.invoke(messages)
        except Exception as e:
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        try:
            llm_response = response.content.strip()
        except (AttributeError, TypeError):
            verification_report = self.empty_verification(
                "Invalid response structure from the model."
            )
            verification_report_formatted = self.format_verification_report(verification_report)
            return {
                "verification_report": verification_report_formatted,
                "context_used": context
            }

        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized_response:
            verification_report = self.empty_verification("Empty response from the model.")
        else:
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                verification_report = self.empty_verification(
                    "Failed to parse the model's response."
                )

        verification_report_formatted = self.format_verification_report(verification_report)

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }
