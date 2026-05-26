"""
Direct response node module.

Provides fast-path bypasses for direct greetings, real-time date queries,
and local AST-based math evaluations. Launches the multi-tool ReAct agent on fallback.
"""

from __future__ import annotations

import ast
import datetime
import logging
import operator
import re
import time
from typing import Any

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.routing import is_direct_support_query, is_weather_query, is_web_query
from src.agent.web_search import WebResultFormatter
from src.agent.prompts import build_generator_prompt
from src.agent.state import AgentState
from src.config import settings
from src.agent.nodes.base import get_chat_llm, get_agent_llm, select_recent_history
from src.agent.nodes.generator import _final_answer_fields

logger = logging.getLogger(__name__)

_MAX_ABS_VALUE = 10**12
_MAX_POWER_EXPONENT = 10_000

_PLAIN_DIRECT_TOOL_RE = re.compile(
    r"("
    r"hesapla|calculate|kaç eder|yüzde|percent|kdv|vat|"
    r"dosya|belge|pdf|upload|yüklediğim|oku|read_uploaded_file|"
    r"github|gitlab|repo|repository|commit|pull request|branch|issue|gist|"
    r"takvim|calendar|email gönder|send email|toplantı ayarla|schedule meeting"
    r")",
    re.IGNORECASE | re.UNICODE,
)
_PLAIN_DIRECT_ARITH_RE = re.compile(r"^\s*[\d\s+\-*/().,^%]+\s*$")
_DATE_QUERY_RE = re.compile(
    r"(bug[üu]n(ün)?\s*(tarih|g[üu]n|g[üu]nl[üu]k|hangi|ne|kaç[ıi]nc[ıi])|"
    r"tarih\s*(nedir|ne|kaç|bugün)|"
    r"bug[üu]n\s*ne\s*g[üu]n[üu]?|"
    r"hangi\s*g[üu]n[üu]?|"
    r"g[üu]n[üu]n\s*tarihi|"
    r"bu\s*g[üu]n\s*(g[üu]nlerden|ne\s*g[üu]n|hangi))",
    re.IGNORECASE | re.UNICODE,
)
_MATH_WORD_RE = re.compile(
    r"(asal|prime|fibonacci|fakt[öo]riyel|factorial|mutlak\s+fark|"
    r"basamakl[ıi]|toplam[ıi]?|çarp[ıi]m[ıi]?|carp[ıi]m[ıi]?|kaç\s+eder|kac\s+eder)",
    re.IGNORECASE | re.UNICODE,
)
_PLAIN_DIRECT_CHAT_RE = re.compile(
    r"^\s*("
    r"merhaba|selam|hey|hi|hello|"
    r"nas[ıi]ls[ıi]n|naber|g[üu]nayd[ıi]n|iyi\s+(g[üu]nler|ak[şs]amlar)|"
    r"te[şs]ekk[üu]r|sa[ğg]ol|tamam|ok|eyvallah|"
    r"sen\s+kimsin|ad[ıi]n\s+ne|ne\s+yapabilirsin"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

_SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_math_expr(expression: str) -> str:
    """Safe evaluation of purely arithmetic calculations using Python's AST."""
    def _eval(node: Any) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in _SAFE_MATH_OPS:
                raise ValueError(f"Unsupported operator: {op.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            if op is ast.Pow:
                if abs(right) > _MAX_POWER_EXPONENT:
                    raise ValueError(f"Exponent too large (max {_MAX_POWER_EXPONENT}).")
                if abs(left) > _MAX_ABS_VALUE:
                    raise ValueError(f"Base too large (max {_MAX_ABS_VALUE}).")
            result = _SAFE_MATH_OPS[op](left, right)
            if isinstance(result, (int, float)) and abs(result) > _MAX_ABS_VALUE:
                raise ValueError(f"Result too large (max {_MAX_ABS_VALUE}).")
            return result
        if isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in _SAFE_MATH_OPS:
                raise ValueError(f"Unsupported operator: {op.__name__}")
            return _SAFE_MATH_OPS[op](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    normalized = expression.replace("^", "**").replace(",", ".")
    result = _eval(ast.parse(normalized, mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression.strip()} = {result}"


def _should_use_plain_direct_llm(question: str) -> bool:
    """Skip tool templates and ReAct reasoning chains for quick greetings or simple chatter."""
    q = question.strip()
    if not q:
        return True
    if is_web_query(q) or re.search(r"(github|gitlab|repo|mcp|toplantı|calendar|email)", q, re.I):
        return False
    if _PLAIN_DIRECT_ARITH_RE.fullmatch(q) or _PLAIN_DIRECT_TOOL_RE.search(q):
        return False
    if _PLAIN_DIRECT_CHAT_RE.search(q):
        return True
    return len(q) <= 80 and "\n" not in q


def _should_use_math_direct_llm(question: str) -> bool:
    """Bypass ReAct agent flow for math word problems; let chat LLM resolve directly."""
    q = question.strip()
    has_mcp = re.search(r"(github|gitlab|repo|mcp|toplantı|calendar|email)", q, re.I)
    return bool(_MATH_WORD_RE.search(q)) and not is_web_query(q) and not has_mcp


def _dedupe_tools(tools: list) -> list:
    """Deduplicate tools by their registered name key."""
    seen: set[str] = set()
    result = []
    for t in tools:
        name = getattr(t, "name", "") or ""
        if name and name not in seen:
            seen.add(name)
            result.append(t)
    return result


def _get_deduped_tools_cached(mcp_tools: list, base_tools: list) -> list:
    """Cache the merged MCP + built-in tool suite within the Chainlit session context."""
    try:
        mcp_names = tuple(getattr(t, "name", "") for t in mcp_tools)
        cached = cl.user_session.get("_deduped_tools_cache")
        cached_key = cl.user_session.get("_deduped_tools_key")
        if cached is not None and cached_key == mcp_names:
            return cached
        result = _dedupe_tools(mcp_tools + base_tools)
        cl.user_session.set("_deduped_tools_cache", result)
        cl.user_session.set("_deduped_tools_key", mcp_names)
        return result
    except Exception:
        return _dedupe_tools(mcp_tools + base_tools)


async def direct_response_node(state: AgentState) -> AgentState:
    """Produce direct chat, mathematical, or ReAct-based agent outputs.

    If live web queries are requested, uses the Tavily search service directly.
    """
    t0 = time.perf_counter()
    question = state["question"]
    prior_messages = list(state.get("messages", []))
    direct_history = select_recent_history(prior_messages, mode="direct")

    # Fast Path 1: Live real-time Web Search Query
    if is_web_query(question):
        from src.agent.nodes.web_search import get_web_search_service, _build_web_search_queries, _search_web_queries, _is_pure_weather_query, _fast_web_summarize, _web_fallback_answer, web_docs_from_result

        service = get_web_search_service()
        search_queries = _build_web_search_queries(question, prior_messages)
        search_query = " | ".join(search_queries)
        logger.debug(
            "Direct: web_fast [queries=%d, query_chars=%d, prior=%d]",
            len(search_queries), len(search_query), len(prior_messages),
        )
        t_search = time.perf_counter()
        web_result = await _search_web_queries(service, search_queries)
        if web_result:
            logger.debug(
                "Direct: web_result [provider=%s, chars=%d, search_t=%.3fs]",
                web_result.provider, len(web_result.text),
                time.perf_counter() - t_search,
            )
            t_sum = time.perf_counter()
            if settings.weather_specialization_enabled and is_weather_query(question) and _is_pure_weather_query(question):
                answer = WebResultFormatter.format_weather(question, web_result.text)
                logger.debug("Direct: weather_format [ans_len=%dch, t=%.3fs]", len(answer), time.perf_counter() - t_sum)
            else:
                answer = await _fast_web_summarize(
                    question,
                    web_result.text,
                    direct_history,
                    search_query=search_query,
                )
                if not answer.strip():
                    answer = _web_fallback_answer(question, web_docs_from_result(web_result, query=search_query))
                logger.debug(
                    "Direct: web_summarize [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
                    len(answer), time.perf_counter() - t_sum, time.perf_counter() - t0,
                )

            new_messages = [
                *prior_messages,
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            return {
                **state,
                "generation": answer,
                "messages": new_messages,
                **_final_answer_fields(state, answer, t0=t0, mode="direct_response")
            }
        else:
            logger.warning("Direct: web_no_result [search_t=%.3fs]", time.perf_counter() - t_search)
            if service is None:
                answer = (
                    "Canlı web araması şu anda devre dışı çünkü `TAVILY_API_KEY` ayarlanmamış. "
                    "Bu yüzden güncel hava durumu verisini güvenilir şekilde çekemiyorum. "
                    "Web araması için `.env` içine `TAVILY_API_KEY` ekleyip uygulamayı yeniden başlatmalısın."
                )
            else:
                answer = (
                    "Web araması sonuç döndürmedi. Canlı hava durumu gibi güncel bilgiler için "
                    "web sağlayıcısını kontrol edip tekrar deneyebilirsin."
                )
            new_messages = [
                *prior_messages,
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            return {
                **state,
                "generation": answer,
                "messages": new_messages,
                **_final_answer_fields(state, answer, t0=t0, mode="direct_response")
            }

    # Fast Path 2: Rapid Date lookup
    if _DATE_QUERY_RE.search(question):
        _today = datetime.date.today()
        _months_tr = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }
        _days_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
        answer = f"{_today.day} {_months_tr[_today.month]} {_today.year}, {_days_tr[_today.weekday()]}."
        logger.debug("Direct: date_fast [ans='%s', total_t=%.3fs]", answer, time.perf_counter() - t0)
        new_messages = [*prior_messages, HumanMessage(content=question), AIMessage(content=answer)]
        return {
            **state,
            "generation": answer,
            "messages": new_messages,
            **_final_answer_fields(state, answer, t0=t0, mode="direct_response")
        }

    # Fast Path 3: Local Math Expression Evaluator
    if _PLAIN_DIRECT_ARITH_RE.fullmatch(question.strip()):
        try:
            answer = _safe_eval_math_expr(question)
        except Exception as exc:
            answer = f"Hesaplama hatası: {exc}"
        logger.debug("Direct: calc_fast [q_len=%d, total_t=%.3fs]", len(question), time.perf_counter() - t0)
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
        return {
            **state,
            "generation": answer,
            "messages": new_messages,
            **_final_answer_fields(state, answer, t0=t0, mode="direct_response")
        }

    # Fast Path 4: Fast Chat LLM Math word problems
    if _should_use_math_direct_llm(question):
        logger.debug("Direct: math_chat [prior=%d, q_len=%d]", len(prior_messages), len(question))
        system_prompt = (
            "Sen kısa ve doğru matematik çözen bir asistansın.\n"
            "Kullanıcının dilinde yanıt ver. Gereken ara adımları kısa göster.\n"
            "Sonucu net biçimde yaz. Görsel, web veya belge bağlamı yoksa bunlardan bahsetme."
        )
        llm = get_chat_llm(max_tokens=int(state.get("max_tokens") or settings.chat_max_tokens))
        t_math = time.perf_counter()
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
        generation = getattr(response, "content", "") or ""
        logger.debug(
            "Direct: math_done [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
            len(generation), time.perf_counter() - t_math, time.perf_counter() - t0,
        )
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "messages": new_messages,
            **_final_answer_fields(state, generation, t0=t0, mode="direct_response")
        }

    # Fast Path 5: Fast Chat LLM Plain conversational greeting
    if _should_use_plain_direct_llm(question):
        logger.debug("Direct: plain_chat [prior=%d, q_len=%d]", len(prior_messages), len(question))
        _today = datetime.date.today()
        _months_tr = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
        }
        _days_tr = {0: "Pazartesi", 1: "Salı", 2: "Çarşamba", 3: "Perşembe", 4: "Cuma", 5: "Cumartesi", 6: "Pazar"}
        _date_str = f"{_today.day} {_months_tr[_today.month]} {_today.year}, {_days_tr[_today.weekday()]}"
        system_prompt = (
            f"GÜNCEL TARİH: {_date_str}.\n"
            "Sen bir yapay zeka asistanısın. Adın Frappe'dir (başka ismin yok).\n"
            "İsim sorulursa sadece 'Frappe' de; 'Sen Frappe' veya 'Ben Sen' yazma.\n"
            "Kullanıcının diliyle yanıt ver. Türkçe soru → Türkçe yanıt.\n"
            "Kısa ama samimi ol. Selamlamalara sıcak ve doğal yanıt ver (1-2 cümle).\n"
            "Soruyu başta tekrar etme; emoji kullanma."
        )
        if is_direct_support_query(question):
            system_prompt += (
                "\nKullanıcı devam etmeni, yarım kalan cevabı tamamlamanı veya cevap kesilmesini açıklamanı "
                "istiyorsa web arama yapmadan son asistan cevabından devam et ya da teknik nedeni kısa açıkla."
            )
        llm = get_chat_llm(max_tokens=int(state.get("max_tokens") or settings.chat_max_tokens))
        messages_to_send = [SystemMessage(content=system_prompt)]
        messages_to_send.extend(direct_history)
        messages_to_send.append(HumanMessage(content=question))
        t_plain = time.perf_counter()
        response = await llm.ainvoke(messages_to_send)
        generation = getattr(response, "content", "") or ""
        logger.debug(
            "Direct: plain_done [ans_len=%dch, llm_t=%.3fs, total_t=%.3fs]",
            len(generation), time.perf_counter() - t_plain, time.perf_counter() - t0,
        )
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "messages": new_messages,
            **_final_answer_fields(state, generation, t0=t0, mode="direct_response")
        }

    # Normal Path: Multi-tool ReAct Agent
    from langgraph.prebuilt import create_react_agent
    from src.tools.search import tavily_search
    from src.tools.file_reader import read_uploaded_file
    from src.tools.calculator import calculator
    from src.tools.mcp_bridge import mcp_call
    from src.mcp.mcp_client import get_mcp_tools

    mcp_tools: list = []
    try:
        cached = cl.user_session.get("mcp_langchain_tools")
        if isinstance(cached, list) and cached:
            mcp_tools = cached
    except Exception:
        pass

    if not mcp_tools and re.search(r"(github|gitlab|repo|mcp|toplantı|calendar|email)", question, re.I):
        try:
            mcp_tools = await get_mcp_tools()
        except Exception as exc:
            logger.warning("MCP araçları yüklenemedi: %s", exc)

    base_tools = [tavily_search, calculator, read_uploaded_file, mcp_call]
    all_tools = _get_deduped_tools_cached(mcp_tools, base_tools)

    system_prompt = build_generator_prompt(all_tools)
    llm = get_agent_llm()
    backend = (settings.llm_backend or "").lower().strip()

    if backend in {"llama.cpp", "llamacpp", "llama"} and re.search(r"(github|gitlab|repo|mcp|toplantı|calendar|email)", question, re.I):
        logger.debug("Direct: react_skip [reason=llamacpp_no_tools, prior=%d]", len(prior_messages))
        messages_to_send = [SystemMessage(content=system_prompt)]
        messages_to_send.extend(direct_history)
        messages_to_send.append(HumanMessage(content=question))
        response = await llm.ainvoke(messages_to_send)
        generation = getattr(response, "content", "") or ""
        new_messages = [
            *prior_messages,
            HumanMessage(content=question),
            AIMessage(content=generation),
        ]
        return {
            **state,
            "generation": generation,
            "messages": new_messages,
            **_final_answer_fields(state, generation, t0=t0, mode="direct_response")
        }

    logger.debug(
        "Direct: react_agent [tools=%d, prior=%d, backend=%s]",
        len(all_tools), len(prior_messages), backend,
    )
    t_react = time.perf_counter()

    llm_key = f"{type(llm).__name__}_{getattr(llm, 'model_name', getattr(llm, 'model', ''))}"
    agent_cache_key = (tuple(sorted(getattr(t, "name", "") for t in all_tools)), llm_key)
    try:
        _cached_agent = cl.user_session.get("_react_agent")
        _cached_agent_key = cl.user_session.get("_react_agent_key")
        if _cached_agent is not None and _cached_agent_key == agent_cache_key:
            agent = _cached_agent
        else:
            agent = create_react_agent(llm, all_tools, prompt=system_prompt)
            cl.user_session.set("_react_agent", agent)
            cl.user_session.set("_react_agent_key", agent_cache_key)
    except Exception:
        agent = create_react_agent(llm, all_tools, prompt=system_prompt)

    result = await agent.ainvoke({"messages": direct_history + [HumanMessage(content=question)]})

    generation = result["messages"][-1].content
    logger.debug(
        "Direct: react_done [ans_len=%dch, react_t=%.3fs, total_t=%.3fs]",
        len(generation), time.perf_counter() - t_react, time.perf_counter() - t0,
    )
    new_messages = [
        *prior_messages,
        HumanMessage(content=question),
        AIMessage(content=generation),
    ]
    return {
        **state,
        "generation": generation,
        "messages": new_messages,
        **_final_answer_fields(state, generation, t0=t0, mode="direct_response")
    }
