from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
from langchain_openai import ChatOpenAI


class ResearchAgent:
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

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)
        try:
            response = self.model.invoke(
                [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        except Exception as e:
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        try:
            llm_response = response.content.strip()
        except (AttributeError, TypeError):
            llm_response = "I cannot answer this question based on the provided documents."

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
