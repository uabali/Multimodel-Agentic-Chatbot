"""
Multi-query generation for improved retrieval recall.

Source: Frappe/src/query_translation.py (full port).
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

_MULTI_QUERY_TEMPLATE = """Rephrase the user's question in {num_queries} different ways.
Each rephrased question must seek the same information but use different wording, \
different angles, or a different language (e.g. Turkish <-> English).
Keep each question short and clear.

Original question: {question}

Generate exactly {num_queries} alternative questions, one per line.
Do NOT add numbering, bullets, or any explanation. Output only the questions.

Alternative questions:"""


def generate_multi_queries(question: str, llm, num_queries: int = 3) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["question", "num_queries"],
        template=_MULTI_QUERY_TEMPLATE,
    )
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question, "num_queries": num_queries})
        raw = [q.strip() for q in result.split("\n") if q.strip()]
        cleaned = []
        for q in raw:
            q = q.lstrip("0123456789.-) ").strip()
            if q and len(q) > 5:
                cleaned.append(q)
        return [question] + cleaned[:num_queries]
    except Exception as exc:
        logger.warning("Multi-query generation failed: %s", exc)
        return [question]
