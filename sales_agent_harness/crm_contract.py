from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


HANDOFF_NEXT_ACTION = "转人工处理法务、安全或商务条款确认"

HANDOFF_SCOPE_BY_FLAG = {
    "LEGAL_OR_SECURITY": "法务、安全或合规材料（DPA、SOC2、安全审计、数据保护）",
    "CONTRACT_OR_PROCUREMENT": "合同或采购条款",
    "CUSTOM_QUOTE": "全球部署或定制报价",
    "COMPLAINT": "投诉或升级请求",
    "HUMAN_REQUEST": "人工跟进请求",
}
HANDOFF_RISK_FLAGS = tuple(HANDOFF_SCOPE_BY_FLAG.keys())
HANDOFF_SOURCE_FACT_PATTERNS = {
    "security_or_legal": r"安全|法务|合规|DPA|SOC2|审计|安全问卷|安全报告|数据保护|GDPR|等保|个保法",
    "contract_or_procurement": r"合同|采购|条款|procurement|contract",
    "custom_quote": r"定制报价|全球部署|大客户|Deal Desk|enterprise|custom quote",
    "technical_review": r"技术|架构|性能|延迟|p99|SLA|部署|私有化|数据存储|训练数据",
    "complaint": r"投诉|complaint",
    "human_request": r"人工|真人|人工跟进|人工回访|human handoff",
}
HANDOFF_FACTS_BY_FLAG = {
    "LEGAL_OR_SECURITY": {"security_or_legal"},
    "CONTRACT_OR_PROCUREMENT": {"contract_or_procurement"},
    "CUSTOM_QUOTE": {"custom_quote"},
    "COMPLAINT": {"complaint"},
    "HUMAN_REQUEST": {"human_request"},
}


def build_handoff_reason(risk_flags: Iterable[str], requested_scope: str = "") -> str:
    scope = handoff_scope(risk_flags, requested_scope)
    owner = _handoff_owner(scope, risk_flags)
    return f"客户询问{scope}，需要{owner}确认"


def build_handoff_payload(
    *,
    lead_id: str,
    qualification_level: str,
    customer_pain: str,
    risk_flags: Iterable[str],
    requested_scope: str = "",
    related_scope: str = "",
    next_action: str = HANDOFF_NEXT_ACTION,
) -> Dict[str, Any]:
    scope = handoff_scope(risk_flags, requested_scope)
    summary = _handoff_summary(scope, related_scope)
    return {
        "lead_id": lead_id,
        "summary": summary,
        "customer_pain": customer_pain,
        "current_stage": "handoff_required",
        "qualification_level": qualification_level,
        "next_action": next_action,
    }


def handoff_scope(risk_flags: Iterable[str], requested_scope: str = "") -> str:
    flags = list(risk_flags or [])
    scopes: List[str] = []
    for flag, scope in HANDOFF_SCOPE_BY_FLAG.items():
        if flag in flags and scope not in scopes:
            scopes.append(scope)
    if scopes:
        return "、".join(scopes)
    cleaned_scope = _clean_requested_scope(requested_scope)
    return cleaned_scope or "法务、安全、采购或高风险商务材料"


def handoff_source_facts(text: str) -> List[str]:
    return [name for name, pattern in HANDOFF_SOURCE_FACT_PATTERNS.items() if re.search(pattern, text or "", re.I)]


def handoff_summary_has_context(text: str) -> bool:
    return bool(handoff_source_facts(text))


def handoff_summary_covers_flags(text: str, risk_flags: Iterable[str]) -> bool:
    facts = set(handoff_source_facts(text))
    expected_facts = set()
    for flag in risk_flags or []:
        expected_facts.update(HANDOFF_FACTS_BY_FLAG.get(flag, set()))
    if not expected_facts:
        return bool(facts)
    return bool(facts.intersection(expected_facts))


def _handoff_summary(scope: str, related_scope: str = "") -> str:
    suffix = "" if scope.endswith("问题") or scope.endswith("请求") else "需求"
    summary = f"客户提出{scope}{suffix}，需人工跟进。"
    if related_scope:
        summary += f" 同时客户也询问{related_scope}。"
    return summary


def _handoff_owner(scope: str, risk_flags: Iterable[str]) -> str:
    flags = set(risk_flags or [])
    if "LEGAL_OR_SECURITY" in flags or any(token in scope for token in ["法务", "安全", "合规", "DPA", "SOC2", "审计"]):
        return "销售或安全同事"
    if "CONTRACT_OR_PROCUREMENT" in flags or any(token in scope for token in ["合同", "采购"]):
        return "销售或法务同事"
    if "COMPLAINT" in flags or "投诉" in scope:
        return "销售负责人"
    if any(token in scope for token in ["技术", "架构", "部署", "性能", "数据"]):
        return "销售或技术同事"
    return "销售同事"


def _clean_requested_scope(requested_scope: str) -> str:
    text = (requested_scope or "").strip(" ，,。；;")
    for prefix in ["客户询问", "客户提出"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    for marker in ["，需要", ",需要", "，需", ",需"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    for suffix in ["需要人工确认", "需要人工跟进", "需人工跟进", "需求"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip(" ，,。；;")
