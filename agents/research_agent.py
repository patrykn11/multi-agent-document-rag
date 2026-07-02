from typing import Dict, List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from config.settings import settings
from langchain_openai import ChatOpenAI


class ResearchAgent:
    NO_ANSWER_RESPONSE = "I cannot answer this question based on the provided documents."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """You are an AI assistant designed to provide precise and factual answers based on the given context.

Instructions:
- Answer the question using only the provided context.
- Be clear, concise, and factual.
- Return as much relevant information as you can get from the context."""
            ),
            HumanMessagePromptTemplate.from_template(
                """Question:
{question}

Context:
{context}

Provide your answer below:"""
            ),
        ]
    )

    def __init__(self):
        """
        Initialize the research agent with OpenAI.
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

    def generate_prompt(self, question: str, context: str) -> List[BaseMessage]:
        """
        Generate structured chat messages for the LLM.
        """
        return self.prompt_template.format_prompt(
            question=question,
            context=context,
        ).to_messages()

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        context = "\n\n".join(doc.page_content for doc in documents)
        messages = self.generate_prompt(question, context)
        try:
            response = self.model.invoke(messages)
        except Exception as e:
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        try:
            llm_response = response.content.strip()
        except (AttributeError, TypeError):
            llm_response = self.NO_ANSWER_RESPONSE

        draft_answer = self.sanitize_response(llm_response) if llm_response else self.NO_ANSWER_RESPONSE

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
