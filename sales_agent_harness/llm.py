from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

from .crm_contract import HANDOFF_RISK_FLAGS, build_handoff_payload, build_handoff_reason
from .grounding import extract_requested_facets, format_facets, serialize_requested_facets

if TYPE_CHECKING:
    from .models import ActionPlan, AgentState, ConversationTurn


class LLMError(Exception):
    """Raised when a model call cannot produce usable content."""


class LLMClientUnavailable(LLMError):
    """Raised when the requested model provider is not configured."""


@dataclass
class LLMCompletion:
    content: str
    metadata: Dict[str, Any]


class ChatClient(Protocol):
    provider: str
    model: str

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
        request_name: str = "plan",
    ) -> LLMCompletion:
        ...


class PlannerProtocol(Protocol):
    def plan(
        self,
        conversation: List["ConversationTurn"],
        state: "AgentState",
        intent: str,
    ) -> "ActionPlan":
        ...

    def prompt_metadata(self) -> Dict[str, Any]:
        ...


class PromptTemplateLoader:
    @staticmethod
    def load(prompt_version: str) -> Tuple[str, str]:
        path = PromptTemplateLoader.path_for(prompt_version)
        prompt_text = path.read_text(encoding="utf-8")
        return prompt_text, hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    @staticmethod
    def path_for(prompt_version: str) -> Path:
        version = PromptTemplateLoader.normalize_version(prompt_version)
        return Path(__file__).with_name(f"{version}.txt")

    @staticmethod
    def normalize_version(prompt_version: str) -> str:
        if prompt_version == "prompt_v1":
            return "prompt_v1"
        if prompt_version in ["prompt_v2", "harness_v2", "rule_based"]:
            return "prompt_v2"
        return "prompt_v2"


class DisabledExternalChatClient:
    provider = "disabled_external_llm"
    model = "disabled"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        raise LLMClientUnavailable("External LLM API calls are disabled in this offline mock harness.")

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
        request_name: str = "plan",
    ) -> LLMCompletion:
        raise LLMClientUnavailable("External LLM API calls are disabled in this offline mock harness.")


class LocalMockLLM:
    """Prompt-sensitive local model double for offline planner tests."""

    provider = "mock_llm"
    model = "local-json-double"

    def complete(
        self,
        prompt: str,
        state_card: Dict[str, Any],
        conversation: List[Dict[str, str]],
        intent: str,
    ) -> Dict[str, Any]:
        payload = {
            "intent": intent,
            "state_card": state_card,
            "conversation": conversation,
            "allowed_actions": state_card.get("allowed_actions", []),
        }
        return _mock_plan(payload, strict_prompt=_is_strict_prompt(prompt))

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1600,
        request_name: str = "plan",
    ) -> LLMCompletion:
        system_prompt = messages[0]["content"] if messages else ""
        user_prompt = messages[-1]["content"] if messages else ""
        payload = _extract_input_payload(user_prompt)
        if request_name == "repair":
            plan = _mock_plan(payload, strict_prompt=True)
        elif os.environ.get("SALES_AGENT_MOCK_LLM_BAD_JSON") == "1":
            return LLMCompletion(
                content='{"intent": "broken", "tool_calls": [',
                metadata={"llm_provider": self.provider, "llm_model": self.model, "request_name": request_name},
            )
        else:
            plan = _mock_plan(payload, strict_prompt=_is_strict_prompt(system_prompt))
        return LLMCompletion(
            content=json.dumps(plan, ensure_ascii=False),
            metadata={"llm_provider": self.provider, "llm_model": self.model, "request_name": request_name},
        )


class MockLLMClient(LocalMockLLM):
    """Backward-compatible name for the local mock LLM client."""


def _extract_input_payload(prompt: str) -> Dict[str, Any]:
    match = re.search(r"INPUT_JSON:\s*(\{.*\})\s*$", prompt, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _is_strict_prompt(prompt: str) -> bool:
    markers = [
        "allowed_actions",
        "事实状态",
        "Harness",
        "book_demo 工具 success",
        "book_demo success",
        "只能基于",
        "evidence_policy",
    ]
    return sum(1 for marker in markers if marker in prompt) >= 2


def _mock_plan(payload: Dict[str, Any], strict_prompt: bool) -> Dict[str, Any]:
    intent = payload.get("intent") or "generic_inquiry"
    state = payload.get("state_card") or {}
    info = state.get("collected_info") or {}
    booking = state.get("booking") or {}
    risk_flags = state.get("risk_flags") or []
    lead_id = state.get("lead_id") or payload.get("lead_id") or "L123"
    last_user = _last_user(payload.get("conversation") or [])

    if not strict_prompt:
        return _weak_prompt_plan(intent, state, info, booking, risk_flags, lead_id, last_user)

    if intent == "handoff_required":
        facets = extract_requested_facets(last_user)
        product_facets = [facet for facet in facets if facet.facet not in ["security_audit", "training_data_compliance", "data_storage", "technical_architecture", "modeling_approach", "api_latency", "private_deployment"]]
        related_scope = format_facets(product_facets) if product_facets else ""
        reason = build_handoff_reason(risk_flags)
        crm_fields = _crm_fields(state, info, booking)
        crm_payload = build_handoff_payload(
            lead_id=lead_id,
            qualification_level=state.get("qualification_level", "unknown"),
            customer_pain=crm_fields["customer_pain"],
            risk_flags=risk_flags,
            related_scope=related_scope,
        )
        tool_calls = []
        if product_facets:
            tool_calls.append({"tool_name": "search_knowledge_base", "arguments": {"query": last_user}})
        tool_calls.extend(
            [
                {
                    "tool_name": "write_crm_note",
                    "arguments": crm_payload,
                },
                {"tool_name": "handoff_to_human", "arguments": {"lead_id": lead_id, "reason": reason, "urgency": "high"}},
            ]
        )
        return {
            "intent": intent,
            "tool_calls": tool_calls,
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "handoff_with_product" if product_facets else "handoff", "requested_facets": serialize_requested_facets(product_facets)},
        }

    if intent == "technical_or_security_question":
        facets = extract_requested_facets(last_user)
        facet_text = format_facets(facets) if facets else "技术架构、部署、性能或数据合规问题"
        reason = build_handoff_reason(risk_flags, requested_scope=facet_text)
        crm_fields = _crm_fields(state, info, booking)
        crm_payload = build_handoff_payload(
            lead_id=lead_id,
            qualification_level=state.get("qualification_level", "unknown"),
            customer_pain=crm_fields["customer_pain"],
            risk_flags=risk_flags,
            requested_scope=facet_text,
            next_action="转人工确认技术、安全或数据合规细节",
        )
        return {
            "intent": intent,
            "tool_calls": [
                {"tool_name": "search_knowledge_base", "arguments": {"query": last_user}},
                {
                    "tool_name": "write_crm_note",
                    "arguments": crm_payload,
                },
                {"tool_name": "handoff_to_human", "arguments": {"lead_id": lead_id, "reason": reason, "urgency": "high"}},
            ],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "technical_or_security", "requested_facets": serialize_requested_facets(facets)},
        }

    if intent in ["pricing_question", "metric_question", "customer_case_question", "product_question"]:
        kind_by_intent = {
            "pricing_question": "pricing",
            "metric_question": "metric",
            "customer_case_question": "customer_case",
            "product_question": "product",
        }
        metadata = {"response_kind": kind_by_intent[intent]}
        if intent == "product_question":
            metadata["requested_facets"] = serialize_requested_facets(extract_requested_facets(last_user))
        return {
            "intent": intent,
            "tool_calls": [{"tool_name": "search_knowledge_base", "arguments": {"query": last_user}}],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": metadata,
        }

    if intent == "booking":
        if "AMBIGUOUS_TIMEZONE" in risk_flags:
            return {"intent": intent, "metadata": {"response_kind": "ambiguous_timezone"}, "claims": [], "risk_flags": risk_flags}
        if "TIMEZONE_OFFSET_NEEDS_CITY" in risk_flags:
            return {"intent": intent, "metadata": {"response_kind": "timezone_offset_needs_city"}, "claims": [], "risk_flags": risk_flags}
        missing = []
        if info.get("invalid_email") and not info.get("email"):
            missing.append("invalid_email")
        elif not info.get("email"):
            missing.append("email")
        if not info.get("timezone"):
            missing.append("timezone")
        if not info.get("demo_purpose"):
            missing.append("demo_purpose")
        if missing:
            return {"intent": intent, "metadata": {"response_kind": "booking_missing_fields", "missing_fields": missing}, "claims": [], "risk_flags": risk_flags}
        if not booking.get("offered_slots"):
            return {
                "intent": intent,
                "tool_calls": [{"tool_name": "check_calendar", "arguments": {"timezone": info.get("timezone"), "duration_minutes": 30}}],
                "claims": [],
                "risk_flags": risk_flags,
                "metadata": {"response_kind": "offer_slots"},
            }
        if not booking.get("selected_slot_id") or not booking.get("selected_slot_confirmed"):
            return {"intent": intent, "metadata": {"response_kind": "waiting_slot_selection"}, "claims": [], "risk_flags": risk_flags}
        summary = _summary(info, booking=booking, include_demo=True)
        return {
            "intent": intent,
            "tool_calls": [
                {
                    "tool_name": "book_demo",
                    "arguments": {
                        "lead_id": lead_id,
                        "slot_id": booking.get("selected_slot_id"),
                        "attendee_email": info.get("email"),
                        "summary": summary,
                    },
                },
                {
                    "tool_name": "write_crm_note",
                    "arguments": {
                        "lead_id": lead_id,
                        "summary": summary,
                        **_crm_fields(state, info, booking),
                        "qualification_level": state.get("qualification_level", "unknown"),
                        "next_action": "Demo booked",
                    },
                },
            ],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "book_demo"},
        }

    if intent == "qualification":
        return {
            "intent": intent,
            "tool_calls": [
                {
                    "tool_name": "write_crm_note",
                    "arguments": {
                        "lead_id": lead_id,
                        "summary": _summary(info),
                        **_crm_fields(state, info, booking),
                        "qualification_level": state.get("qualification_level", "unknown"),
                        "next_action": state.get("next_action") or "继续确认线索信息",
                    },
                }
            ],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "qualification"},
        }

    draft = "可以，我先了解一下你们现在主要想解决哪类销售问题：新线索响应慢、老客户跟进断档，还是销售记录整理比较耗时？"
    response_kind = "generic_clarification"
    return {
        "intent": intent,
        "assistant_message_draft": draft,
        "tool_calls": [],
        "claims": [],
        "risk_flags": risk_flags,
        "metadata": {"response_kind": response_kind},
    }


def _weak_prompt_plan(
    intent: str,
    state: Dict[str, Any],
    info: Dict[str, Any],
    booking: Dict[str, Any],
    risk_flags: List[str],
    lead_id: str,
    last_user: str,
) -> Dict[str, Any]:
    if intent in ["pricing_question", "metric_question", "customer_case_question", "product_question"]:
        return {
            "intent": intent,
            "assistant_message_draft": "这个需要销售同事结合你们的情况确认，我可以先帮你记录需求。",
            "tool_calls": [],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "generic_clarification", "weak_prompt_behavior": "skipped_knowledge_base"},
        }

    if intent == "handoff_required":
        return {
            "intent": intent,
            "assistant_message_draft": "这类材料需要同事确认，我会先把需求转给人工跟进。",
            "tool_calls": [{"tool_name": "handoff_to_human", "arguments": {"lead_id": lead_id, "reason": "客户需要人工跟进", "urgency": "high"}}],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "handoff", "weak_prompt_behavior": "missed_crm_note"},
        }

    if intent == "booking":
        if not booking.get("offered_slots"):
            return {
                "intent": intent,
                "assistant_message_draft": "可以，我先帮你推进 Demo 安排，稍后再补齐必要信息。",
                "tool_calls": [],
                "claims": [],
                "risk_flags": risk_flags,
                "metadata": {"response_kind": "generic_clarification", "weak_prompt_behavior": "unstructured_booking_reply"},
            }
        if booking.get("selected_slot_id"):
            return {
                "intent": intent,
                "assistant_message_draft": "我会继续帮你处理这个 Demo 时间。",
                "tool_calls": [
                    {
                        "tool_name": "book_demo",
                        "arguments": {
                            "lead_id": lead_id,
                            "slot_id": booking.get("selected_slot_id"),
                            "attendee_email": info.get("email"),
                            "summary": _summary(info, booking=booking, include_demo=True),
                        },
                    }
                ],
                "claims": [],
                "risk_flags": risk_flags,
                "metadata": {"response_kind": "book_demo", "weak_prompt_behavior": "missed_crm_note"},
            }

    if intent == "qualification":
        return {
            "intent": intent,
            "assistant_message_draft": "你们的场景我了解了，后续可以继续沟通。",
            "tool_calls": [],
            "claims": [],
            "risk_flags": risk_flags,
            "metadata": {"response_kind": "generic_clarification", "weak_prompt_behavior": "missed_crm_note"},
        }

    return {
        "intent": intent,
        "assistant_message_draft": "可以，我先了解一下你们当前的销售流程和主要需求。",
        "tool_calls": [],
        "claims": [],
        "risk_flags": risk_flags,
        "metadata": {"response_kind": "generic_clarification", "weak_prompt_behavior": "direct_reply"},
    }


def _last_user(conversation: List[Dict[str, str]]) -> str:
    for turn in reversed(conversation):
        if turn.get("role") == "user":
            return turn.get("content", "")
    return ""


def _summary(info: Dict[str, Any], booking: Optional[Dict[str, Any]] = None, include_demo: bool = False) -> str:
    pieces = []
    if info.get("company_size"):
        pieces.append(f"{info['company_size']}人公司")
    if info.get("industry"):
        pieces.append(str(info["industry"]))
    if info.get("sales_team_size"):
        pieces.append(f"{info['sales_team_size']}人销售团队")
    if info.get("pain_point"):
        pieces.append(f"痛点是{info['pain_point']}")
    if include_demo:
        pieces.append(f"Demo需求是{info.get('demo_purpose') or '产品演示'}")
        if info.get("email"):
            pieces.append(f"参会邮箱{info['email']}")
        if booking and booking.get("selected_slot_id"):
            pieces.append(f"Demo时间已选择：{booking['selected_slot_id']}")
    if not pieces:
        pieces.append("客户需求待澄清")
    return "，".join(pieces)


def __getattr__(name: str) -> Any:
    if name == "MockLLMPlanner":
        from .policy import MockLLMPlanner

        return MockLLMPlanner
    raise AttributeError(name)


def _crm_fields(state: Dict[str, Any], info: Dict[str, Any], booking: Dict[str, Any]) -> Dict[str, str]:
    risk_flags = state.get("risk_flags") or []
    booking_status = booking.get("status") or "NO_DEMO_INTENT"
    if info.get("pain_point"):
        customer_pain = str(info["pain_point"])
    elif info.get("demo_purpose"):
        customer_pain = f"Demo关注：{info['demo_purpose']}"
    elif info.get("interest_area"):
        customer_pain = f"关注场景：{info['interest_area']}"
    elif any(flag in risk_flags for flag in HANDOFF_RISK_FLAGS):
        customer_pain = "高风险材料或商务条款需人工确认"
    else:
        customer_pain = "待澄清"

    if any(flag in risk_flags for flag in HANDOFF_RISK_FLAGS):
        current_stage = "handoff_required"
    elif booking.get("book_demo_success") or booking_status in ["BOOKED_CRM_PENDING", "BOOKED_CRM_SYNCED"]:
        current_stage = "demo_booked"
    elif booking_status != "NO_DEMO_INTENT":
        current_stage = "demo_pending"
    elif any(info.get(field) for field in ["company_size", "industry", "sales_team_size", "pain_point", "budget", "decision_maker", "go_live_time"]):
        current_stage = "qualification"
    else:
        current_stage = "discovery"
    return {"customer_pain": customer_pain, "current_stage": current_stage}
