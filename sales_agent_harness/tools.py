from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .crm_contract import HANDOFF_SOURCE_FACT_PATTERNS, handoff_summary_has_context
from .models import CRMNote, KBDocument, OutboxEvent, ToolResult
from .store import InMemoryStore


LEAD_DB: Dict[str, Dict[str, Any]] = {
    "L123": {
        "company_name": "Demo Manufacturing",
        "previous_notes": [],
        "known_email": None,
        "known_timezone": None,
    },
    "L_EXISTING": {
        "company_name": "Existing Manufacturing",
        "previous_notes": ["300人制造业公司，痛点是销售跟进不及时。"],
        "company_size": "300",
        "industry": "制造业",
        "pain_point": "销售跟进不及时",
        "known_email": None,
        "known_timezone": None,
    },
}

KB_UPDATED_AT = "2026-06-02T00:00:00Z"


def _kb(
    doc_id: str,
    topic: str,
    title: str,
    content: str,
    tags: List[str],
    policy: Dict[str, Any],
    supported_facets: Optional[List[str]] = None,
    unsupported_facets: Optional[List[str]] = None,
    visibility: str = "public",
    confidence: str = "high",
) -> KBDocument:
    return KBDocument(
        id=doc_id,
        topic=topic,
        title=title,
        content=content,
        tags=tags,
        policy=policy,
        supported_facets=supported_facets or [],
        unsupported_facets=unsupported_facets or [],
        visibility=visibility,
        confidence=confidence,
        updated_at=KB_UPDATED_AT,
    )


KB_DOCS: List[KBDocument] = [
    _kb(
        "kb_feature_001",
        "lead_follow_up",
        "线索自动分层",
        "产品支持对线索进行自动分层、提醒销售跟进、生成跟进建议，并可与 CRM 集成。",
        ["线索", "自动分层", "跟进提醒", "跟进建议", "销售跟进", "漏跟进"],
        {"feature_policy": "feature_supported"},
        ["lead_scoring", "follow_up_reminder", "follow_up_suggestion", "crm_integration"],
    ),
    _kb(
        "kb_feature_002",
        "lead_follow_up",
        "跟进提醒和待办",
        "线索跟进场景会结合线索来源、最近互动、销售动作和未处理提醒生成下一步建议。",
        ["线索来源", "销售动作", "下一步建议", "提醒", "跟进慢"],
        {"feature_policy": "feature_supported"},
        ["follow_up_reminder", "follow_up_suggestion"],
    ),
    _kb(
        "kb_feature_003",
        "lead_follow_up",
        "不替代销售成交",
        "系统不承诺自动成交，也不完全替代销售判断；销售仍需确认客户意向、预算和决策流程。",
        ["自动成交", "替代销售", "销售判断", "预算", "决策"],
        {"feature_policy": "needs_confirmation", "unsupported_claims": ["automatic_close", "replace_sales"]},
        [],
        ["auto_close_deals", "replace_sales"],
        confidence="medium",
    ),
    _kb(
        "kb_crm_001",
        "crm_integration",
        "常见 CRM 集成",
        "产品支持与常见 CRM 系统集成，具体集成方式需销售或实施同事确认。",
        ["CRM", "crm", "集成", "对接", "实施"],
        {"feature_policy": "feature_supported"},
        ["crm_integration"],
    ),
    _kb(
        "kb_crm_002",
        "crm_integration",
        "CRM 字段同步范围",
        "CRM 同步通常覆盖线索字段、跟进记录、任务状态和预约结果，字段映射需按客户系统确认。",
        ["字段映射", "同步", "跟进记录", "任务状态", "预约结果"],
        {"feature_policy": "feature_supported"},
        ["crm_field_sync"],
    ),
    _kb(
        "kb_crm_003",
        "crm_integration",
        "私有系统不保证即插即用",
        "Salesforce、HubSpot 等常见系统可作为评估入口；非标准或私有 CRM 需要实施同事评估 API 能力，不能承诺即插即用。",
        ["Salesforce", "HubSpot", "API", "私有系统", "即插即用"],
        {"feature_policy": "needs_confirmation", "integration_policy": "implementation_review_required"},
        ["crm_integration", "implementation_review"],
        ["instant_private_crm_integration"],
        confidence="medium",
    ),
    _kb(
        "kb_pricing_001",
        "pricing",
        "公开知识库不提供固定报价",
        "公开知识库暂时不会给固定报价。具体报价需要销售根据客户规模、使用场景和部署方式确认。",
        ["价格", "报价", "多少钱", "费用", "标准版", "企业版", "最低价"],
        {"price_policy": "no_public_price"},
        ["pricing"],
        ["exact_price", "discount", "custom_quote"],
    ),
    _kb(
        "kb_pricing_002",
        "pricing",
        "报价评估维度",
        "报价评估会参考席位规模、线索量、CRM 集成复杂度、部署区域和支持级别。",
        ["席位", "线索量", "部署", "支持级别", "价格依据"],
        {"price_policy": "no_public_price"},
        ["pricing"],
        ["exact_price", "discount", "custom_quote"],
    ),
    _kb(
        "kb_pricing_003",
        "pricing",
        "定制报价需要人工评估",
        "大客户、全球部署或定制报价需要人工销售或 Deal Desk 评估，不能由自动流程直接承诺固定 5 万或 20 万报价。",
        ["大客户", "全球部署", "定制报价", "Deal Desk", "5万", "20万", "承诺"],
        {"price_policy": "no_public_price", "custom_quote_required": True},
        ["pricing"],
        ["exact_price", "discount", "custom_quote", "delivery_commitment"],
    ),
    _kb(
        "kb_security_001",
        "security",
        "安全材料需要转人工",
        "安全审计、DPA、SOC2、合规材料需要由销售或安全同事提供。",
        ["DPA", "SOC2", "安全", "审计", "法务", "合规"],
        {"feature_policy": "needs_confirmation", "handoff_required": True},
        ["security_audit"],
        ["security_report_details"],
    ),
    _kb(
        "kb_security_002",
        "security",
        "安全问卷和采购条款",
        "涉及数据处理协议、采购条款、审计报告或安全问卷时，应转给人工同事处理。",
        ["数据处理", "采购条款", "审计报告", "安全问卷", "人工"],
        {"feature_policy": "needs_confirmation", "handoff_required": True},
        ["security_audit"],
        ["security_report_details"],
    ),
    _kb(
        "kb_security_003",
        "security",
        "受限安全报告内容",
        "SOC2 报告原文、审计报告细节和内部安全材料属于受限内容，不应由公开自动回复披露。",
        ["SOC2报告原文", "审计报告细节", "安全报告下载", "内部安全材料"],
        {"feature_policy": "needs_confirmation", "handoff_required": True},
        ["security_audit"],
        ["security_report_details"],
        visibility="restricted",
    ),
    _kb(
        "kb_case_001",
        "customer_case",
        "公开知识库无客户名称",
        "当前公开知识库没有可公开客户名称。",
        ["客户案例", "客户名称", "客户名", "500", "500强", "名字"],
        {"customer_case_policy": "no_public_customer_name"},
        ["customer_case"],
        ["public_customer_names"],
    ),
    _kb(
        "kb_case_002",
        "customer_case",
        "案例材料需要销售提供",
        "如客户需要案例材料，应记录行业、规模和关注场景，请销售提供可分享版本。",
        ["案例材料", "行业", "规模", "可分享", "客户案例"],
        {"customer_case_policy": "no_public_customer_name", "case_material_needs_sales": True},
        ["customer_case"],
        ["public_customer_names"],
    ),
    _kb(
        "kb_case_003",
        "customer_case",
        "可分享匿名案例",
        "可分享匿名化案例的场景摘要，但不能在公开自动回复中列出客户真实名称。",
        ["匿名案例", "匿名化", "场景摘要", "真实名称"],
        {"customer_case_policy": "no_public_customer_name", "anonymized_case_available": True},
        ["customer_case"],
        ["public_customer_names"],
    ),
    _kb(
        "kb_metric_001",
        "metric",
        "不承诺固定效果比例",
        "效果取决于客户线索质量、销售流程、团队执行和系统集成情况，公开知识库不承诺固定比例。",
        ["转化率", "提升", "效果", "承诺", "保证", "比例"],
        {"metric_policy": "no_guaranteed_metric"},
        ["metric"],
        ["exact_metric", "guaranteed_metric"],
    ),
    _kb(
        "kb_metric_002",
        "metric",
        "效果评估依赖工作流",
        "评估效果时建议先确认当前基线、线索来源、响应时长、销售跟进纪律和 CRM 数据完整度。",
        ["基线", "响应时长", "数据完整度", "评估", "工作流"],
        {"metric_policy": "no_guaranteed_metric", "effect_depends_on_workflow": True},
        ["metric"],
        ["exact_metric", "guaranteed_metric"],
    ),
    _kb(
        "kb_metric_003",
        "metric",
        "不能保证 50% 提升",
        "可以设计试点指标，但不能在公开答复中保证具体转化率、收入或提效百分比，例如固定提升 50%。",
        ["试点", "指标", "收入", "提效", "百分比", "50%"],
        {"metric_policy": "no_guaranteed_metric"},
        ["metric"],
        ["exact_metric", "guaranteed_metric"],
    ),
]

CALENDAR_DB: Dict[str, List[Dict[str, Any]]] = {
    "Asia/Singapore": [
        {
            "slot_id": "sg_slot_1",
            "display": "周二 10:00",
            "timezone": "Asia/Singapore",
            "available": True,
        },
        {
            "slot_id": "sg_slot_2",
            "display": "周三 15:00",
            "timezone": "Asia/Singapore",
            "available": True,
        },
        {
            "slot_id": "sg_slot_3",
            "display": "周四 11:00",
            "timezone": "Asia/Singapore",
            "available": False,
        },
        {
            "slot_id": "sg_slot_4",
            "display": "周五 16:00",
            "timezone": "Asia/Singapore",
            "available": True,
        },
        {
            "slot_id": "sg_slot_5",
            "display": "下周一 10:30",
            "timezone": "Asia/Singapore",
            "available": True,
        },
    ],
    "Asia/Shanghai": [
        {
            "slot_id": "cn_slot_1",
            "display": "周二 10:00",
            "timezone": "Asia/Shanghai",
            "available": True,
        },
        {
            "slot_id": "cn_slot_2",
            "display": "周三 14:00",
            "timezone": "Asia/Shanghai",
            "available": True,
        },
        {
            "slot_id": "cn_slot_3",
            "display": "周四 16:00",
            "timezone": "Asia/Shanghai",
            "available": True,
        },
    ],
    "Asia/Tokyo": [
        {
            "slot_id": "jp_slot_1",
            "display": "周二 11:00",
            "timezone": "Asia/Tokyo",
            "available": True,
        },
        {
            "slot_id": "jp_slot_2",
            "display": "周三 16:00",
            "timezone": "Asia/Tokyo",
            "available": True,
        },
        {
            "slot_id": "jp_slot_3",
            "display": "周五 10:00",
            "timezone": "Asia/Tokyo",
            "available": True,
        },
    ],
    "Australia/Sydney": [
        {
            "slot_id": "au_slot_1",
            "display": "周三 14:00",
            "timezone": "Australia/Sydney",
            "available": True,
        },
        {
            "slot_id": "au_slot_2",
            "display": "周四 10:00",
            "timezone": "Australia/Sydney",
            "available": True,
        },
    ],
    "Asia/Kolkata": [
        {
            "slot_id": "in_slot_1",
            "display": "周二 12:00",
            "timezone": "Asia/Kolkata",
            "available": True,
        },
        {
            "slot_id": "in_slot_2",
            "display": "周四 15:00",
            "timezone": "Asia/Kolkata",
            "available": True,
        },
    ],
    "America/New_York": [
        {
            "slot_id": "ny_slot_1",
            "display": "周二 09:00",
            "timezone": "America/New_York",
            "available": True,
        },
        {
            "slot_id": "ny_slot_2",
            "display": "周三 13:00",
            "timezone": "America/New_York",
            "available": True,
        },
        {
            "slot_id": "ny_slot_3",
            "display": "周五 11:00",
            "timezone": "America/New_York",
            "available": True,
        },
    ],
    "America/Los_Angeles": [
        {
            "slot_id": "sf_slot_1",
            "display": "周二 09:00",
            "timezone": "America/Los_Angeles",
            "available": True,
        },
        {
            "slot_id": "sf_slot_2",
            "display": "周三 14:00",
            "timezone": "America/Los_Angeles",
            "available": True,
        },
        {
            "slot_id": "sf_slot_3",
            "display": "周四 10:00",
            "timezone": "America/Los_Angeles",
            "available": True,
        },
    ],
    "Europe/London": [
        {
            "slot_id": "ldn_slot_1",
            "display": "周二 09:30",
            "timezone": "Europe/London",
            "available": True,
        },
        {
            "slot_id": "ldn_slot_2",
            "display": "周三 14:00",
            "timezone": "Europe/London",
            "available": True,
        },
        {
            "slot_id": "ldn_slot_3",
            "display": "周四 11:00",
            "timezone": "Europe/London",
            "available": False,
        },
    ],
    "Asia/Dubai": [],
}

# Compatibility helper for callers that used the old reset entrypoint. Runtime
# code should pass this store explicitly via the *_with_store tool wrappers.
def reset_mock_stores(overrides: Optional[Dict[str, Any]] = None) -> InMemoryStore:
    store = InMemoryStore()
    store.reset(overrides)
    return store


def _store_or_default(value: Any) -> InMemoryStore:
    return value if isinstance(value, InMemoryStore) else InMemoryStore()


def _next_call(store: InMemoryStore, tool_name: str) -> int:
    store.call_counts[tool_name] = store.call_counts.get(tool_name, 0) + 1
    return store.call_counts[tool_name]


def _forced_error(store: InMemoryStore, tool_name: str) -> Optional[str]:
    count = _next_call(store, tool_name)
    cfg = store.mock_overrides.get(tool_name, {})
    if cfg.get("force_failure"):
        return cfg.get("error", "forced_failure")
    if cfg.get("fail_once") and count == 1:
        return cfg.get("error", "mock_once_failure")
    return None


def get_lead_context(store_or_lead_id: Any, lead_id: Optional[str] = None) -> ToolResult:
    store = _store_or_default(store_or_lead_id)
    actual_lead_id = lead_id if isinstance(store_or_lead_id, InMemoryStore) else store_or_lead_id
    args = {"lead_id": actual_lead_id}
    cache_key = str(actual_lead_id)
    if cache_key in store.lead_context_cache:
        data = dict(store.lead_context_cache[cache_key])
        data["cache_hit"] = True
        return ToolResult(tool_name="get_lead_context", arguments=args, success=True, data=data)
    forced = _forced_error(store, "get_lead_context")
    if forced:
        return ToolResult(tool_name="get_lead_context", arguments=args, success=False, error=forced)
    data = dict(
        LEAD_DB.get(
            actual_lead_id,
            {
                "company_name": None,
                "previous_notes": [],
                "known_email": None,
                "known_timezone": None,
            },
        )
    )
    store.lead_context_cache[cache_key] = dict(data)
    data["cache_hit"] = False
    return ToolResult(
        tool_name="get_lead_context",
        arguments=args,
        success=True,
        data=data,
    )


def get_lead_context_with_store(store: InMemoryStore, lead_id: str) -> ToolResult:
    return get_lead_context(store, lead_id)


def _topic_for_query(query: str) -> str:
    topics = _topic_hints_for_query(query)
    return topics[0] if topics else "unknown"


def _topic_hints_for_query(query: str) -> List[str]:
    text = query.lower()
    topic_keywords = [
        ("pricing", ["价格", "报价", "多少钱", "费用", "标准版", "企业版", "最低价", "price"]),
        ("customer_case", ["500", "客户案例", "客户名称", "客户名", "客户", "在用", "案例", "case", "字节", "美团", "理想汽车", "京东", "百度"]),
        ("metric", ["转化率", "提升", "效果", "承诺", "保证", "指标", "metric"]),
        ("security", ["dpa", "soc2", "gdpr", "安全", "审计", "法务", "合规", "数据保护", "等保", "个保法", "采购条款", "安全问卷"]),
        ("crm_integration", ["crm", "集成", "对接", "字段映射", "salesforce", "hubspot", "api"]),
        ("lead_follow_up", ["线索", "跟进", "提醒", "自动分层", "销售工具", "销售流程"]),
    ]
    topics: List[str] = []
    for topic, keywords in topic_keywords:
        if any(keyword.lower() in text for keyword in keywords):
            topics.append(topic)
    return topics


def _rank_kb_docs(query: str) -> List[tuple[int, KBDocument]]:
    text = query.lower()
    topic_hints = _topic_hints_for_query(query)
    ranked = []
    for doc in KB_DOCS:
        score = 0
        if doc.topic in topic_hints:
            score += 5
        for keyword in doc.tags:
            if str(keyword).lower() in text:
                score += 3
        content = doc.content.lower()
        for term in _query_terms(query):
            if len(term) >= 2 and term.lower() in content:
                score += 1
        if score > 0:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return ranked[:4]


def _query_terms(query: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", query or "")


def _matched_topics(ranked_docs: List[tuple[int, KBDocument]]) -> List[str]:
    topics = []
    for _score, doc in ranked_docs:
        if doc.topic not in topics:
            topics.append(doc.topic)
    return topics


def _kb_result_status(ranked_docs: List[tuple[int, KBDocument]], topics: List[str]) -> str:
    if not ranked_docs:
        return "no_result"
    top_score = ranked_docs[0][0]
    top_docs = [doc for score, doc in ranked_docs if score == top_score]
    if top_docs and all(doc.visibility == "restricted" for doc in top_docs):
        return "restricted"
    top_topic_scores: Dict[str, int] = {}
    for score, doc in ranked_docs:
        top_topic_scores[doc.topic] = max(score, top_topic_scores.get(doc.topic, 0))
    if len(topics) > 1:
        scores = sorted(top_topic_scores.values(), reverse=True)
        if len(scores) > 1 and scores[0] - scores[1] <= 3:
            return "ambiguous"
    return "found"


def _merge_doc_policy(ranked_docs: List[tuple[int, KBDocument]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for _score, doc in ranked_docs:
        for key, value in doc.policy.items():
            if key not in merged:
                merged[key] = value
    return merged


def search_knowledge_base(store_or_query: Any, query: Optional[str] = None) -> ToolResult:
    store = _store_or_default(store_or_query)
    actual_query = query if isinstance(store_or_query, InMemoryStore) else store_or_query
    forced = _forced_error(store, "search_knowledge_base")
    args = {"query": actual_query}
    if forced:
        return ToolResult(tool_name="search_knowledge_base", arguments=args, success=False, error=forced)
    ranked_docs = _rank_kb_docs(actual_query or "")
    topics = _matched_topics(ranked_docs)
    topic = topics[0] if topics else _topic_for_query(actual_query or "")
    result_status = _kb_result_status(ranked_docs, topics)
    docs = [_copy_doc(doc, restricted=result_status == "restricted") for _score, doc in ranked_docs]
    docs = _apply_kb_doc_rewrites(store, docs)
    policy = _merge_doc_policy(ranked_docs)
    return ToolResult(
        tool_name="search_knowledge_base",
        arguments=args,
        success=True,
        data={
            "query": actual_query,
            "topic": topic,
            "matched_topic": topic,
            "topics": topics,
            "documents": docs,
            "evidence_ids": [doc["id"] for doc in docs],
            "result_status": result_status,
            "policy": policy,
        },
    )


def search_knowledge_base_with_store(store: InMemoryStore, query: str) -> ToolResult:
    return search_knowledge_base(store, query)


def _apply_kb_doc_rewrites(store: InMemoryStore, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rewrite = store.mock_overrides.get("kb_doc_rewrite", {})
    if not isinstance(rewrite, dict):
        return docs
    rewritten = [dict(doc) for doc in docs]
    pricing_content = rewrite.get("pricing_content")
    policy_override = rewrite.get("policy")
    for doc in rewritten:
        if pricing_content and doc.get("topic") == "pricing":
            doc["content"] = pricing_content
        if isinstance(policy_override, dict):
            policy = dict(doc.get("policy") or {})
            policy.update(policy_override)
            doc["policy"] = policy
    return rewritten


def calendar_slots(timezone: str) -> List[Dict[str, Any]]:
    return [dict(slot) for slot in CALENDAR_DB.get(timezone, []) if slot.get("available", True)]


def check_calendar(store_or_timezone: Any, timezone: Optional[Any] = None, duration_minutes: int = 30) -> ToolResult:
    if isinstance(store_or_timezone, InMemoryStore):
        store = store_or_timezone
        actual_timezone = timezone
        actual_duration = duration_minutes
    else:
        store = InMemoryStore()
        actual_timezone = store_or_timezone
        actual_duration = timezone if isinstance(timezone, int) else duration_minutes
    forced = _forced_error(store, "check_calendar")
    args = {"timezone": actual_timezone, "duration_minutes": actual_duration}
    if forced:
        return ToolResult(
            tool_name="check_calendar",
            arguments=args,
            success=False,
            data={"slots": [], "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": forced},
            error=forced,
        )
    cfg = store.mock_overrides.get("check_calendar", {})
    if cfg.get("unsupported_timezone"):
        return ToolResult(
            tool_name="check_calendar",
            arguments=args,
            success=False,
            data={"slots": [], "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": "unsupported_timezone"},
            error="unsupported_timezone",
        )
    if cfg.get("no_slots"):
        return ToolResult(
            tool_name="check_calendar",
            arguments=args,
            success=False,
            data={"slots": [], "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": "no_slots"},
            error="no_slots",
        )
    if actual_timezone == "CST":
        return ToolResult(
            tool_name="check_calendar",
            arguments=args,
            success=False,
            data={"slots": [], "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": "ambiguous_timezone"},
            error="ambiguous_timezone",
        )
    if actual_timezone not in CALENDAR_DB:
        return ToolResult(
            tool_name="check_calendar",
            arguments=args,
            success=False,
            data={"slots": [], "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": "unsupported_timezone"},
            error="unsupported_timezone",
        )
    slots = calendar_slots(actual_timezone)
    unavailable_slot_id = cfg.get("slot_unavailable")
    if unavailable_slot_id:
        slots = [slot for slot in slots if slot.get("slot_id") != unavailable_slot_id]
    if not slots:
        return ToolResult(
            tool_name="check_calendar",
            arguments=args,
            success=False,
            data={"slots": [], "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": "no_slots"},
            error="no_slots",
        )
    return ToolResult(
        tool_name="check_calendar",
        arguments=args,
        success=True,
        data={"slots": slots, "timezone": actual_timezone, "duration_minutes": actual_duration, "result_status": "available"},
    )


def check_calendar_with_store(store: InMemoryStore, timezone: str, duration_minutes: int = 30) -> ToolResult:
    return check_calendar(store, timezone, duration_minutes)


def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email or ""))


QUALIFICATION_FACTS = {"company_size", "industry", "sales_team_size", "pain_point", "budget", "decision_maker", "go_live_time"}
BOOKING_FACTS = {"email", "demo_time", "demo_purpose"}
HANDOFF_FACTS = set(HANDOFF_SOURCE_FACT_PATTERNS)
CRM_CONTRACT_ERRORS = {
    "missing_summary",
    "summary_missing_context",
    "summary_low_information",
    "summary_missing_demo_context",
    "summary_missing_booking_facts",
    "summary_missing_handoff_reason",
    "invalid_qualification_level",
    "missing_next_action",
}


def _crm_summary_error(summary: str, next_action: str = "", current_stage: str = "") -> Optional[str]:
    text = (summary or "").strip()
    if not text:
        return "missing_summary"
    compact = re.sub(r"[\s，,。；;：:！!？?、\-_]+", "", text)
    if len(compact) < 8:
        return "summary_missing_context"
    if _low_information_text(compact):
        return "summary_low_information"
    stage = _resolved_crm_stage(text, next_action, current_stage)
    facts = set(_crm_source_facts(text))
    if stage in ["demo_pending", "demo_booked"]:
        if not re.search(r"Demo|demo|演示|预约", text):
            return "summary_missing_demo_context"
        if len(facts.intersection(BOOKING_FACTS)) < 2:
            return "summary_missing_booking_facts"
        return None
    if stage == "handoff_required":
        if not facts.intersection(HANDOFF_FACTS) or not handoff_summary_has_context(text):
            return "summary_missing_handoff_reason"
        return None
    if len(facts.intersection(QUALIFICATION_FACTS)) < 2:
        return "summary_missing_context"
    return None


def _low_information_text(text: str) -> bool:
    chars = [char for char in text if re.match(r"[\w\u4e00-\u9fff]", char)]
    if len(chars) < 8:
        return True
    counts = Counter(chars)
    if len(counts) < 4:
        return True
    most_common = counts.most_common(1)[0][1]
    if most_common / len(chars) > 0.45:
        return True
    return bool(re.search(r"(.{1,3})\1{3,}", "".join(chars)))


def _crm_summary_dimensions(summary: str) -> List[str]:
    dimensions = []
    patterns = {
        "organization": r"\d+\s*人|公司|企业|集团|团队|制造业|互联网|软件|SaaS|教育|金融|医疗|医药|零售|消费品|快消|汽车|地产|物流|出版|销售团队",
        "need": r"痛点|需求|关注场景|Demo目的|想看|了解|线索|CRM|对接|集成|响应慢|漏|效率|转化|产品演示",
        "commercial_or_risk": r"阶段|预算|决策|上线|采购|报价|合同|法务|安全|合规|DPA|SOC2|审计|全球部署|定制|高风险",
        "action": r"Demo|演示|预约|参会邮箱|人工|销售同事|安全同事|跟进|下一步|重试",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, summary, re.I):
            dimensions.append(name)
    return dimensions


def _crm_note_stage(summary: str, next_action: str) -> str:
    combined = f"{summary} {next_action}"
    if "Demo booked" in next_action or "已预约" in summary:
        return "demo_booked"
    if re.search(r"CRM_SYNC_PENDING|crm_sync_pending|同步.*重试", combined, re.I):
        return "crm_sync_pending"
    if re.search(r"handoff|人工|法务|安全|合规|DPA|SOC2|合同|采购|定制报价|全球部署|投诉", combined, re.I):
        return "handoff_required"
    if re.search(r"Demo目的|演示目的|参会邮箱|Demo时间|演示时间|预约时间|slot_", summary):
        return "demo_pending"
    if re.search(r"痛点|预算|决策|上线|销售团队|行业|制造业|零售|\d+\s*人", summary):
        return "qualification"
    if re.search(r"Demo|demo|演示|预约", combined):
        return "demo_pending"
    return "discovery"


def _resolved_crm_stage(summary: str, next_action: str, current_stage: str = "") -> str:
    inferred_stage = _crm_note_stage(summary, next_action)
    if not current_stage:
        return inferred_stage
    if current_stage in ["discovery", "qualification"] and inferred_stage in [
        "handoff_required",
        "demo_pending",
        "demo_booked",
        "crm_sync_pending",
    ]:
        return inferred_stage
    return current_stage


def _crm_source_facts(summary: str) -> List[str]:
    fact_patterns = {
        "company_size": r"\d+\s*人.{0,8}(?:公司|企业|集团)|公司(?:规模)?\d+\s*人",
        "industry": r"制造业|互联网|软件|SaaS|教育|金融|医疗|医药|零售|消费品|快消|汽车|地产|物流|出版",
        "sales_team_size": r"销售团队\s*\d+\s*人|\d+\s*人销售团队|销售\s*\d+\s*人",
        "pain_point": r"痛点|漏跟进|漏客户|响应慢|跟进慢|效率低|客户流失|销售周期长|销售漏斗|渠道效率低",
        "budget": r"预算|报价|费用|价格",
        "decision_maker": r"决策人|老板|负责人|采购负责人|CEO|CFO|COO|CTO|CIO|CMO|VP|Owner",
        "go_live_time": r"上线|落地|试点|下周|本月|季度|尽快|Q[1-4]|FY\d{2,4}|年底前|年内",
        "email": r"[\w\.-]+@[\w\.-]+\.\w+|参会邮箱|邮箱",
        "demo_time": r"Demo时间|演示时间|预约时间|slot_|周[一二三四五六日天]|下周",
        "demo_purpose": r"Demo目的|演示目的|想看|了解|对接|集成|产品演示",
        **HANDOFF_SOURCE_FACT_PATTERNS,
    }
    return [name for name, pattern in fact_patterns.items() if re.search(pattern, summary, re.I)]


def _crm_customer_pain(summary: str) -> str:
    pain_match = re.search(r"痛点(?:是|：|:)?([^，,。；;]+)", summary)
    if pain_match:
        return pain_match.group(1).strip()
    if any(token in summary for token in ["漏跟进", "漏客户", "响应慢", "跟进慢", "效率低", "客户流失", "销售周期长", "销售漏斗", "渠道效率低"]):
        return "销售跟进或线索管理问题"
    if any(token in summary for token in ["技术", "架构", "延迟", "部署", "数据存储", "训练数据"]):
        return "需要确认技术、部署或数据合规细节"
    if any(token in summary for token in ["安全", "法务", "合规", "DPA", "SOC2", "审计"]):
        return "需要安全、法务或合规材料"
    if any(token in summary for token in ["Demo", "演示", "预约"]):
        return "需要产品演示"
    return "客户需求待澄清"


def _known_slot(slot_id: str) -> Optional[Dict[str, Any]]:
    for slots in CALENDAR_DB.values():
        for slot in slots:
            if slot["slot_id"] == slot_id:
                return slot
    return None


def _slot_already_booked(store: InMemoryStore, slot_id: Optional[str]) -> bool:
    if not slot_id:
        return False
    return any(booking.get("slot_id") == slot_id for booking in store.bookings.values())


def book_demo(
    store_or_lead_id: Any,
    lead_id: Optional[str] = None,
    slot_id: Optional[str] = None,
    attendee_email: Optional[str] = None,
    summary: str = "",
    selected_slot_confirmed: Optional[bool] = None,
) -> ToolResult:
    if isinstance(store_or_lead_id, InMemoryStore):
        store = store_or_lead_id
        actual_lead_id = lead_id
        actual_slot_id = slot_id
        actual_email = attendee_email
        actual_summary = summary
        actual_selected_slot_confirmed = selected_slot_confirmed
    else:
        store = InMemoryStore()
        actual_lead_id = store_or_lead_id
        actual_slot_id = lead_id
        actual_email = slot_id
        actual_summary = attendee_email or summary
        actual_selected_slot_confirmed = selected_slot_confirmed
    args = {
        "lead_id": actual_lead_id,
        "slot_id": actual_slot_id,
        "attendee_email": actual_email,
        "summary": actual_summary,
        "selected_slot_confirmed": actual_selected_slot_confirmed,
    }
    key = f"book_demo:{actual_lead_id}:{actual_slot_id}:{actual_email}"
    if key in store.booked_keys:
        data = dict(store.booked_keys[key])
        data["idempotent_replay"] = True
        data["existing_event_id"] = data.get("calendar_event_id")
        return ToolResult(tool_name="book_demo", arguments=args, success=True, data=data)
    forced = _forced_error(store, "book_demo")
    if forced:
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error=forced)
    cfg = store.mock_overrides.get("book_demo", {})
    unavailable_override = cfg.get("slot_unavailable")
    if unavailable_override is True or unavailable_override == actual_slot_id:
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="slot_unavailable")
    if actual_selected_slot_confirmed is False:
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="missing_slot_confirmation")
    if not _valid_email(actual_email or ""):
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="invalid_email")
    if actual_slot_id == "unavailable_slot":
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="slot_unavailable")
    slot = _known_slot(actual_slot_id or "")
    if not slot:
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="slot_not_found")
    if not slot.get("available", True):
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="slot_unavailable")
    if _slot_already_booked(store, actual_slot_id):
        return ToolResult(tool_name="book_demo", arguments=args, success=False, error="slot_already_booked")
    event_id = f"evt_{actual_lead_id}_{actual_slot_id}"
    data = {
        "calendar_event_id": event_id,
        "slot_id": actual_slot_id,
        "attendee_email": actual_email,
        "display": slot["display"],
        "timezone": slot["timezone"],
        "idempotency_key": key,
    }
    store.booked_keys[key] = dict(data)
    store.bookings[event_id] = dict(data)
    return ToolResult(tool_name="book_demo", arguments=args, success=True, data=data)


def book_demo_with_store(
    store: InMemoryStore,
    lead_id: str,
    slot_id: str,
    attendee_email: str,
    summary: str = "",
    selected_slot_confirmed: Optional[bool] = None,
) -> ToolResult:
    return book_demo(store, lead_id, slot_id, attendee_email, summary, selected_slot_confirmed)


def write_crm_note(
    store_or_lead_id: Any,
    lead_id: Optional[str] = None,
    summary: str = "",
    qualification_level: str = "unknown",
    next_action: str = "",
    customer_pain: str = "",
    current_stage: str = "",
) -> ToolResult:
    if isinstance(store_or_lead_id, InMemoryStore):
        store = store_or_lead_id
        actual_lead_id = lead_id
        actual_summary = summary
        actual_level = qualification_level
        actual_next_action = next_action
        actual_customer_pain = customer_pain
        actual_current_stage = current_stage
    else:
        store = InMemoryStore()
        actual_lead_id = store_or_lead_id
        actual_summary = lead_id or ""
        actual_level = summary or "unknown"
        actual_next_action = qualification_level or next_action
        actual_customer_pain = customer_pain
        actual_current_stage = current_stage
    forced = _forced_error(store, "write_crm_note")
    stage = _resolved_crm_stage(actual_summary, actual_next_action, actual_current_stage)
    customer_pain_value = actual_customer_pain or _crm_customer_pain(actual_summary)
    args = {
        "lead_id": actual_lead_id,
        "summary": actual_summary,
        "customer_pain": customer_pain_value,
        "current_stage": stage,
        "qualification_level": actual_level,
        "next_action": actual_next_action,
    }
    if forced:
        return ToolResult(tool_name="write_crm_note", arguments=args, success=False, error=forced)
    summary_error = _crm_summary_error(actual_summary, actual_next_action, stage)
    if summary_error:
        return ToolResult(tool_name="write_crm_note", arguments=args, success=False, error=summary_error)
    if actual_level not in ["high", "medium", "low", "unknown"]:
        return ToolResult(tool_name="write_crm_note", arguments=args, success=False, error="invalid_qualification_level")
    if not actual_next_action:
        return ToolResult(tool_name="write_crm_note", arguments=args, success=False, error="missing_next_action")
    note_id = f"crm_note_{len(store.crm_notes) + 1}"
    source_facts = _crm_source_facts(actual_summary)
    note = CRMNote(
        note_id=note_id,
        lead_id=str(actual_lead_id),
        summary=actual_summary,
        pain_point=customer_pain_value,
        customer_pain=customer_pain_value,
        current_stage=stage,
        qualification_level=actual_level,
        next_action=actual_next_action,
        stage=stage,
        source_facts=source_facts,
    )
    store.crm_notes.append(note.model_dump(mode="json"))
    return ToolResult(
        tool_name="write_crm_note",
        arguments=args,
        success=True,
        data={
            "note_id": note_id,
            "stage": note.stage,
            "current_stage": note.current_stage,
            "pain_point": note.pain_point,
            "customer_pain": note.customer_pain,
            "source_facts": list(note.source_facts),
        },
    )


def write_crm_note_with_store(
    store: InMemoryStore,
    lead_id: str,
    summary: str = "",
    qualification_level: str = "unknown",
    next_action: str = "",
    customer_pain: str = "",
    current_stage: str = "",
) -> ToolResult:
    return write_crm_note(store, lead_id, summary, qualification_level, next_action, customer_pain, current_stage)


def handoff_to_human(
    store_or_lead_id: Any,
    lead_id: Optional[str] = None,
    reason: str = "",
    urgency: str = "high",
) -> ToolResult:
    if isinstance(store_or_lead_id, InMemoryStore):
        store = store_or_lead_id
        actual_lead_id = lead_id
        actual_reason = reason
        actual_urgency = urgency
    else:
        store = InMemoryStore()
        actual_lead_id = store_or_lead_id
        actual_reason = lead_id or reason
        actual_urgency = reason if lead_id is not None and reason in ["high", "medium", "low"] and urgency == "high" else urgency
    forced = _forced_error(store, "handoff_to_human")
    args = {"lead_id": actual_lead_id, "reason": actual_reason, "urgency": actual_urgency}
    if forced:
        return ToolResult(tool_name="handoff_to_human", arguments=args, success=False, error=forced)
    if actual_urgency not in ["high", "medium", "low"]:
        return ToolResult(tool_name="handoff_to_human", arguments=args, success=False, error="invalid_urgency")
    if not actual_reason:
        return ToolResult(tool_name="handoff_to_human", arguments=args, success=False, error="missing_reason")
    route = _handoff_route(actual_reason, actual_urgency)
    ticket_id = f"handoff_{len(store.handoff_log) + 1}"
    record = dict(args)
    record["ticket_id"] = ticket_id
    record.update(route)
    store.handoff_log.append(record)
    return ToolResult(tool_name="handoff_to_human", arguments=args, success=True, data={"ticket_id": ticket_id, **route})


def _handoff_route(reason: str, urgency: str) -> Dict[str, Any]:
    queue = "sales_queue"
    effective_urgency = urgency
    sla = "1_business_day"
    routing_reason = "general_sales_follow_up"
    if re.search(r"投诉|complaint", reason, re.I):
        queue = "support_escalation_queue"
        effective_urgency = "critical"
        sla = "1h"
        routing_reason = "complaint"
    elif re.search(r"DPA|SOC2|安全|审计|法务|合规|安全问卷|报告", reason, re.I):
        queue = "security_sales_queue"
        effective_urgency = "high"
        sla = "4h"
        routing_reason = "security_or_legal_request"
    elif re.search(r"定制报价|采购|合同|条款|全球部署|大客户|procurement|contract|global|custom quote|enterprise", reason, re.I):
        queue = "enterprise_sales_queue"
        effective_urgency = "high"
        sla = "4h"
        routing_reason = "enterprise_or_procurement_request"

    sla_minutes_by_label = {"1h": 60, "4h": 240, "1_business_day": 1440}
    sla_minutes = sla_minutes_by_label[sla]
    escalation_minutes = 30 if sla == "1h" else (120 if sla == "4h" else None)
    due_at = (datetime.now(timezone.utc) + timedelta(minutes=sla_minutes)).isoformat().replace("+00:00", "Z")
    return {
        "queue": queue,
        "urgency": effective_urgency,
        "requested_urgency": urgency,
        "sla": sla,
        "priority": {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3"}.get(effective_urgency, "P2"),
        "sla_minutes": sla_minutes,
        "escalate_after_minutes": escalation_minutes,
        "due_at": due_at,
        "routing_reason": routing_reason,
    }


def handoff_to_human_with_store(
    store: InMemoryStore,
    lead_id: str,
    reason: str = "",
    urgency: str = "high",
) -> ToolResult:
    return handoff_to_human(store, lead_id, reason, urgency)


def enqueue_crm_after_booking(
    store_or_lead_id: Any,
    lead_id: Optional[str] = None,
    calendar_event_id: Optional[str] = None,
    idempotency_key: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(store_or_lead_id, InMemoryStore):
        store = store_or_lead_id
        actual_lead_id = lead_id or ""
        actual_calendar_event_id = calendar_event_id or ""
        actual_idempotency_key = idempotency_key
        actual_payload = payload or {}
    else:
        store = InMemoryStore()
        actual_lead_id = store_or_lead_id
        actual_calendar_event_id = lead_id or ""
        actual_idempotency_key = calendar_event_id or ""
        actual_payload = idempotency_key if isinstance(idempotency_key, dict) else (payload or {})
    key = actual_idempotency_key or f"crm_after_booking:{actual_lead_id}:{actual_calendar_event_id}"
    event = store.add_outbox_event(
        OutboxEvent(
            event_id=f"outbox_{len(store.outbox) + 1}",
            event_type="WRITE_CRM_AFTER_BOOKING",
            lead_id=actual_lead_id,
            idempotency_key=key,
            calendar_event_id=actual_calendar_event_id,
            payload=dict(actual_payload),
            status="pending",
        )
    )
    return event.model_dump(mode="json")


def enqueue_crm_after_handoff(
    store_or_lead_id: Any,
    lead_id: Optional[str] = None,
    ticket_id: str = "",
    payload: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(store_or_lead_id, InMemoryStore):
        store = store_or_lead_id
        actual_lead_id = lead_id or ""
        actual_ticket_id = ticket_id
        actual_payload = payload or {}
        actual_error = error
    else:
        store = InMemoryStore()
        actual_lead_id = store_or_lead_id
        actual_ticket_id = lead_id or ""
        actual_payload = ticket_id if isinstance(ticket_id, dict) else (payload or {})
        actual_error = payload if isinstance(payload, str) else error
    key = f"crm_after_handoff:{actual_lead_id}:{actual_ticket_id or 'unknown_ticket'}"
    event = store.add_outbox_event(
        OutboxEvent(
            event_id=f"outbox_{len(store.outbox) + 1}",
            event_type="WRITE_CRM_AFTER_HANDOFF",
            lead_id=actual_lead_id,
            idempotency_key=key,
            calendar_event_id=actual_ticket_id,
            payload=dict(actual_payload),
            status="pending",
            retry_count=1 if actual_error else 0,
            last_error=actual_error,
        )
    )
    return event.model_dump(mode="json")


def mark_outbox_done(
    store: InMemoryStore,
    *,
    event_id: Optional[str] = None,
    calendar_event_id: Optional[str] = None,
) -> None:
    store.mark_outbox_done(event_id=event_id, calendar_event_id=calendar_event_id)


def mark_outbox_error(
    store: InMemoryStore,
    *,
    event_id: Optional[str] = None,
    calendar_event_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    store.mark_outbox_error(event_id=event_id, calendar_event_id=calendar_event_id, error=error)


def mark_outbox_failed(
    store: InMemoryStore,
    *,
    event_id: Optional[str] = None,
    calendar_event_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    store.mark_outbox_failed(event_id=event_id, calendar_event_id=calendar_event_id, error=error)


def _copy_doc(doc: KBDocument, restricted: bool = False) -> Dict[str, Any]:
    copied = doc.model_dump(mode="json")
    copied["keywords"] = list(copied.get("tags", []))
    if restricted:
        copied["content"] = ""
        copied["redacted"] = True
    return copied
