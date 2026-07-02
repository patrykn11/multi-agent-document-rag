from config.settings import settings
import logging
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from typing import List

logger = logging.getLogger(__name__)

class RelevanceChecker:
    VALID_LABELS = frozenset({"CAN_ANSWER", "PARTIAL", "NO_MATCH"})

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """You are an AI relevance checker between a user's question and provided document content.

Classify how well the document content addresses the question. Respond with only one label:
- CAN_ANSWER: The passages contain enough explicit information to fully answer the question.
- PARTIAL: The passages mention or discuss the topic but do not provide all details needed for a complete answer.
- NO_MATCH: The passages do not discuss or mention the topic at all.

If the passages mention the topic or timeframe in any way, even incompletely, use PARTIAL instead of NO_MATCH.
Do not include any explanation or additional text."""
            ),
            HumanMessagePromptTemplate.from_template(
                """Question:
{question}

Passages:
{document_content}

Label:"""
            ),
        ]
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=30,
        )

    def generate_prompt(self, question: str, document_content: str) -> List[BaseMessage]:
        """Generate structured chat messages for relevance classification."""
        return self.prompt_template.format_prompt(
            question=question,
            document_content=document_content,
        ).to_messages()

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve the top-k document chunks from the global retriever.
        2. Combine them into a single text string.
        3. Pass that text + question to the LLM for classification.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """

        logger.debug("RelevanceChecker.check called with question=%r and k=%s", question, k)

        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        messages = self.generate_prompt(question, document_content)

        try:
            response = self.model.invoke(messages)
        except Exception:
            return "NO_MATCH"

        try:
            llm_response = response.content.strip().upper()
            logger.debug("LLM response: %s", llm_response)
        except (AttributeError, TypeError):
            return "NO_MATCH"

        if llm_response not in self.VALID_LABELS:
            logger.debug("LLM did not respond with a valid label. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"
        else:
            logger.debug("Classification recognized as %r.", llm_response)
            classification = llm_response

        return classification
