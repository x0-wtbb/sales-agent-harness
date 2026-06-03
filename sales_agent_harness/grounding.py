from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from pydantic import ValidationError

from .models import AgentState, ClaimType, FacetName, RequestedFacet, ToolResult

logger = logging.getLogger(__name__)

FACET_PATTERNS: Dict[FacetName, List[str]] = {
    "pricing": ["价格", "报价", "多少钱", "费用", "标准版", "price", "pricing", "quote"],
    "customer_case": ["客户案例", "案例", "哪些客户", "合作客户", "标杆客户", "500 强", "customer case", "case study"],
    "metric": ["指标", "roi", "转化率", "提升多少", "效果", "metric"],
    "technical_architecture": ["核心架构", "架构"],
    "modeling_approach": ["rag", "fine-tuned", "fine tuned", "finetune", "微调", "大模型", "模型"],
    "api_latency": ["p99", "sla", "api 延迟", "api延迟", "latency", "响应时间", "延迟"],
    "private_deployment": ["私有化部署", "私有化", "私有部署", "本地部署", "私有云", "on-prem", "on prem"],
    "data_storage": ["数据存哪里", "数据存储", "数据存放", "数据存"],
    "training_data_compliance": ["gdpr", "训练数据", "来源合规", "数据来源", "数据保护", "个保法", "等保", "合规"],
    "security_audit": ["dpa", "soc2", "iso", "gdpr", "安全审计", "安全", "审计", "法务", "合规材料", "等保", "个保法"],
    "lead_scoring": ["自动分层", "分层"],
    "follow_up_reminder": ["跟进提醒", "提醒销售", "提醒跟进", "漏跟进", "线索跟进", "销售跟进"],
    "follow_up_suggestion": ["跟进建议", "下一步建议", "建议生成"],
    "crm_field_sync": ["crm 字段映射", "crm字段映射", "crm 字段同步", "crm字段同步", "crm 同步", "crm同步"],
    "crm_integration": ["crm 对接", "crm 集成", "crm对接", "crm集成", "salesforce", "hubspot"],
    "implementation_review": ["实施评估", "实施周期", "上线评估"],
}

FACET_LABELS = {
    "pricing": "价格或报价",
    "customer_case": "客户案例",
    "metric": "效果指标",
    "technical_architecture": "核心架构",
    "modeling_approach": "RAG / fine-tuned model",
    "api_latency": "API p99 延迟",
    "private_deployment": "私有化部署",
    "data_storage": "数据存储位置",
    "training_data_compliance": "训练数据来源合规",
    "security_audit": "安全审计或合规材料",
    "lead_scoring": "线索自动分层",
    "follow_up_reminder": "销售跟进提醒",
    "follow_up_suggestion": "跟进建议生成",
    "crm_integration": "CRM 集成",
    "crm_field_sync": "CRM 字段同步",
    "implementation_review": "实施评估",
    "product_capability": "产品能力",
}

CLAIM_TYPE_BY_FACET: Dict[str, ClaimType] = {
    "pricing": "price",
    "customer_case": "customer_case",
    "metric": "metric",
    "technical_architecture": "technical_architecture",
    "modeling_approach": "technical_architecture",
    "api_latency": "metric",
    "private_deployment": "deployment",
    "data_storage": "data_compliance",
    "training_data_compliance": "data_compliance",
    "security_audit": "security",
    "lead_scoring": "feature",
    "follow_up_reminder": "feature",
    "follow_up_suggestion": "feature",
    "crm_integration": "feature",
    "crm_field_sync": "feature",
    "implementation_review": "delivery_commitment",
    "product_capability": "feature",
}

TECHNICAL_OR_SECURITY_FACETS = {
    "technical_architecture",
    "modeling_approach",
    "api_latency",
    "private_deployment",
    "data_storage",
    "training_data_compliance",
    "security_audit",
}
TECHNICAL_ARCHITECTURE_TERMS = [
    term
    for facet, terms in FACET_PATTERNS.items()
    if facet in TECHNICAL_OR_SECURITY_FACETS
    for term in terms
]
HIGH_RISK_CLAIM_TYPES = {
    "price",
    "customer_case",
    "metric",
    "feature",
    "technical_architecture",
    "deployment",
    "security",
    "data_compliance",
    "delivery_commitment",
}

RequestedFacetLike = Union[str, Dict[str, Any], RequestedFacet]


def first_match(text: str, terms: Sequence[str]) -> Optional[str]:
    lowered = (text or "").lower()
    for term in terms:
        if term and term.lower() in lowered:
            return term
    return None


def has_technical_or_security_terms(text: str) -> bool:
    return first_match(text or "", TECHNICAL_ARCHITECTURE_TERMS) is not None


def extract_requested_facets(text: str) -> List[RequestedFacet]:
    facets: List[RequestedFacet] = []
    for facet, terms in FACET_PATTERNS.items():
        phrase = first_match(text or "", terms)
        if phrase is not None:
            facets.append(RequestedFacet(facet=facet, phrase=phrase))
    return facets


def normalize_requested_facets(raw_facets: Sequence[RequestedFacetLike]) -> List[RequestedFacet]:
    """Compatibility shim for planner metadata; new code should emit RequestedFacet."""

    facets: List[RequestedFacet] = []
    for item in raw_facets or []:
        facet: Optional[str] = None
        phrase = ""
        if isinstance(item, RequestedFacet):
            facet = item.facet
            phrase = item.phrase
        elif isinstance(item, dict):
            facet = item.get("facet")
            phrase = str(item.get("phrase") or facet or "")
        elif isinstance(item, str):
            facet = item
            phrase = item
            logger.warning("string requested facet is deprecated; emit RequestedFacet at the source", extra={"facet": item})
        else:
            logger.warning("invalid requested facet shape ignored", extra={"item": repr(item)})
            continue
        if not facet:
            logger.warning("requested facet without facet name ignored", extra={"item": repr(item)})
            continue
        try:
            requested = RequestedFacet(facet=facet, phrase=phrase or facet)
        except ValidationError as exc:
            logger.warning("invalid requested facet ignored", extra={"item": repr(item), "error": str(exc)})
            continue
        if not any(existing.facet == requested.facet for existing in facets):
            facets.append(requested)
    return facets


def serialize_requested_facets(facets: Sequence[RequestedFacetLike]) -> List[Dict[str, str]]:
    return [facet.model_dump(mode="json") for facet in normalize_requested_facets(facets)]


def requested_facet_ids(facets: Sequence[RequestedFacetLike]) -> List[str]:
    ids: List[str] = []
    for item in facets or []:
        if isinstance(item, str):
            facet = item
        elif isinstance(item, RequestedFacet):
            facet = item.facet
        elif isinstance(item, dict):
            facet = item.get("facet")
        else:
            continue
        if facet and facet not in ids:
            ids.append(facet)
    return ids


def docs_from_tool_results(tool_results: Sequence[ToolResult]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for result in tool_results:
        if result.tool_name != "search_knowledge_base" or not result.success:
            continue
        for doc in result.data.get("documents", []):
            if isinstance(doc, dict):
                docs.append(doc)
    return docs


def last_kb_query(tool_results: Sequence[ToolResult]) -> str:
    for result in reversed(tool_results):
        if result.tool_name == "search_knowledge_base":
            return str(result.arguments.get("query") or result.data.get("query") or "")
    return ""


def find_evidence_for_facet(facet: str, docs: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for doc in docs:
        supported = doc.get("supported_facets") or []
        if facet in supported:
            return doc
    return None


def evidence_covers_facet(state: AgentState, facet: Optional[str], evidence_ids: Iterable[str]) -> bool:
    if not facet:
        return False
    supported_ids = set(state.evidence_policy.supported_facets.get(facet, []))
    return bool(supported_ids.intersection(set(evidence_ids)))


def claim_type_for_facet(facet: str) -> ClaimType:
    return CLAIM_TYPE_BY_FACET.get(facet, "other")


def format_facets(facets: Sequence[RequestedFacetLike]) -> str:
    return "、".join(FACET_LABELS.get(facet, facet) for facet in requested_facet_ids(facets))


def has_technical_or_security_facets(facets: Sequence[RequestedFacetLike]) -> bool:
    return any(facet in TECHNICAL_OR_SECURITY_FACETS for facet in requested_facet_ids(facets))
