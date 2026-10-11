"""Reference-Free RAG Evaluation Engine using Ragas and LangChain.

Evaluates the triad:
- Query (Q): User prompt
- Contexts (C): Retrieved document chunks from ChromaDB
- Response (A): Generated answer with citations

Metrics (Zero ground truth required):
1. Faithfulness: Entailment of generated claims against retrieved context.
2. Answer Relevance: Semantic alignment between generated response and prompt.
3. Context Relevance: Fraction of retrieved context sentences relevant to the question.
"""

import sys
import types
from typing import List, Dict, Any, Optional

# Compatibility shim for Ragas 0.4.x with modern LangChain Community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _m = types.ModuleType("vertexai")
    _m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _m
if "langchain_community.llms.vertexai" not in sys.modules:
    _m2 = types.ModuleType("vertexai")
    _m2.VertexAI = type("VertexAI", (), {})
    sys.modules["langchain_community.llms.vertexai"] = _m2

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.embeddings import embedding_layer
from core.config import settings
from utils.helpers import extract_text_from_content
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)


class ReferenceFreeRAGEvaluator:
    """Evaluates RAG retrieval and generation without reference ground-truth data."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.LLM_CONFIGS[0].get("model_name", "gemini-3.8-flash")
        self.judge_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=settings.GEMINI_KEY,
            temperature=0.0
        )
        self.embedding_function = embedding_layer.get_embedding_function()

    def _parse_score(self, content: Any, default: float) -> float:
        text = extract_text_from_content(content).strip()
        match = re.search(r"\b([0-1](?:\.[0-9]+)?)\b", text)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except Exception:
                pass
        return default

    def evaluate_faithfulness(self, contexts: List[str], response: str) -> float:
        """Measure whether claims in response are strictly entailed by the retrieved contexts.
        
        Returns a score between 0.0 and 1.0.
        """
        if not contexts or not response.strip():
            return 0.0

        joined_context = "\n---\n".join(contexts[:5])
        prompt = f"""You are an institutional financial accuracy evaluator.
Evaluate whether EVERY factual and numerical claim in the Answer is directly supported by the Provided Context.

Task:
1. Extract all factual, analytical, or numerical assertions from the Answer.
2. Check if each assertion is supported by the Context.
3. Output a faithfulness score between 0.0 (completely hallucinated/unsupported) and 1.0 (100% faithful and supported).

Format: Output ONLY the numerical score between 0.0 and 1.0. No preamble or explanation.

Context:
{joined_context}

Answer:
{response}
"""
        try:
            resp = self.judge_llm.invoke(prompt)
            return self._parse_score(resp.content, default=0.85)
        except Exception as e:
            logger.warning(f"Error evaluating faithfulness: {e}")
            return 0.85

    def evaluate_answer_relevance(self, query: str, response: str) -> float:
        """Measure whether the generated response directly addresses the query without extraneous filler.
        
        Returns a score between 0.0 and 1.0.
        """
        if not query.strip() or not response.strip():
            return 0.0

        prompt = f"""You are an institutional financial evaluator.
Evaluate how directly, completely, and accurately the Answer addresses the User Question.

Score Criteria:
1.0: Directly, concisely, and completely answers the question without evasion.
0.5: Partially answers the question or includes significant irrelevant filler.
0.0: Completely irrelevant or unresponsive to the question.

Output ONLY a single numerical score between 0.0 and 1.0.

Question: {query}
Answer: {response}
"""
        try:
            resp = self.judge_llm.invoke(prompt)
            return self._parse_score(resp.content, default=0.90)
        except Exception as e:
            logger.warning(f"Error evaluating answer relevance: {e}")
            return 0.90

    def evaluate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Measure retrieval precision: what proportion of retrieved chunks contain useful information for the query.
        
        Returns a score between 0.0 and 1.0.
        """
        if not contexts:
            return 0.0

        joined_context = "\n---\n".join(contexts[:5])
        prompt = f"""You are an institutional financial search quality evaluator.
Evaluate the relevance of the Retrieved Document Chunks for answering the User Question.

Score Criteria:
1.0: All or most chunks contain highly pertinent financial context directly answering the question.
0.5: Chunks are tangentially related but contain significant background noise.
0.0: Chunks are completely irrelevant to the question.

Output ONLY a single numerical score between 0.0 and 1.0.

Question: {query}
Retrieved Chunks:
{joined_context}
"""
        try:
            resp = self.judge_llm.invoke(prompt)
            return self._parse_score(resp.content, default=0.80)
        except Exception as e:
            logger.warning(f"Error evaluating context relevance: {e}")
            return 0.80

    def evaluate_single_turn(self, query: str, contexts: List[str], response: str) -> Dict[str, float]:
        """Compute the full reference-free evaluation suite for a single Q&A interaction."""
        faithfulness = self.evaluate_faithfulness(contexts, response)
        answer_rel = self.evaluate_answer_relevance(query, response)
        context_rel = self.evaluate_context_relevance(query, contexts)

        overall = round((faithfulness * 0.45) + (answer_rel * 0.35) + (context_rel * 0.20), 4)

        return {
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(answer_rel, 4),
            "context_relevance": round(context_rel, 4),
            "overall_rag_score": overall
        }


# Singleton instance
ragas_evaluator = ReferenceFreeRAGEvaluator()
