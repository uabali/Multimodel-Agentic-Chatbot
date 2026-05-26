import logging
from src.config import settings

logger = logging.getLogger(__name__)

from src.agent.nodes.base import (
    reset_nodes_llm_cache,
    select_recent_history,
    RerankerRegistry as _RerankerRegistry,
    observe_node as _observe_node,
    get_rag_llm as _get_rag_llm,
)

# Mark base wrappers to prevent infinite recursion
_observe_node._is_original = True
_get_rag_llm._is_original = True

from src.agent.nodes.router import router_node
from src.agent.nodes.rewriter import rewriter_node
from src.agent.nodes.retriever import (
    retriever_node,
    _build_source_filter,
    _retriever_score_lookup_enabled,
    _fetch_document_overview_chunks,
)
from src.agent.nodes.grader import (
    grader_node,
    _parse_grader_payload,
    _parse_grader_reason,
)
from src.agent.nodes.vision import vision_node, vision_rag_node, vision_search_node
from src.agent.nodes.generator import generator_node, append_used_sources, assemble_rag_context
from src.agent.nodes.web_search import (
    web_search_node,
    _build_contextual_web_query,
    _build_web_search_queries,
    _compact_web_query,
    _docs_from_explicit_urls,
    web_docs_from_result as _web_docs_from_result,
    _web_fallback_answer,
    fetch_public_url_text,
)

fetch_public_url_text._is_original = True

from src.agent.nodes.direct import direct_response_node

__all__ = [
    "settings",
    "logger",
    "reset_nodes_llm_cache",
    "select_recent_history",
    "_RerankerRegistry",
    "_observe_node",
    "_get_rag_llm",
    "router_node",
    "rewriter_node",
    "retriever_node",
    "_build_source_filter",
    "_retriever_score_lookup_enabled",
    "_fetch_document_overview_chunks",
    "grader_node",
    "_parse_grader_payload",
    "_parse_grader_reason",
    "vision_node",
    "vision_rag_node",
    "vision_search_node",
    "generator_node",
    "web_search_node",
    "direct_response_node",
    "append_used_sources",
    "assemble_rag_context",
    "_build_contextual_web_query",
    "_build_web_search_queries",
    "_compact_web_query",
    "_docs_from_explicit_urls",
    "_web_docs_from_result",
    "_web_fallback_answer",
    "fetch_public_url_text",
]
