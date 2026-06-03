from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from .crm_contract import HANDOFF_RISK_FLAGS
from .grounding import extract_requested_facets, format_facets, serialize_requested_facets
from .llm import ChatClient, DisabledExternalChatClient, LLMCompletion, LLMError, LocalMockLLM, MockLLMClient, PromptTemplateLoader
from .models import ActionPlan, AgentState, ConversationTurn, ToolCall
from .normalizer import extract_message
from .state import (
    MemoryManager,
    build_handoff_crm_payload,
    build_handoff_reason,
    build_summary,
    current_stage_for_crm,
    customer_pain_for_crm,
)


class PolicyRouter:
    @staticmethod
    def detect_intent(conversation: List[ConversationTurn], state: AgentState) -> str:
        last_user = MemoryManager.last_user_message(conversation)
        extraction = extract_message(last_user)
        if any(flag in state.risk_flags for flag in HANDOFF_RISK_FLAGS):
            return "handoff_required"
        if "BOOKING_DECLINED" in state.risk_flags:
            return "qualification" if _has_qualified_info(state) or extraction.budget else "generic_inquiry"
        if extraction.prompt_injection and _has_qualified_info(state):
            return "qualification"
        if extraction.prompt_injection and extraction.price_question:
            return "pricing_question"
        if extraction.technical_or_security_question:
            return "technical_or_security_question"
        if extraction.price_question:
            return "pricing_question"
        if extraction.metric_question:
            return "metric_question"
        if extraction.customer_case_question:
            return "customer_case_question"
        if extraction.demo_intent or state.booking.status in ["DEMO_INTERESTED", "READY_TO_CHECK_CALENDAR", "SLOTS_OFFERED", "SLOT_SELECTED", "BOOKING_FAILED"]:
            return "booking"
        if extraction.feature_question:
            return "product_question"
        if _has_qualified_info(state) or extraction.budget:
            return "qualification"
        return "generic_inquiry"

    @staticmethod
    def compute_allowed_actions(state: AgentState, intent: str) -> List[str]:
        if intent == "handoff_required":
            return ["search_knowledge_base", "handoff_to_human", "write_crm_note"]
        if intent == "technical_or_security_question":
            return ["search_knowledge_base", "write_crm_note", "handoff_to_human", "ask_clarification"]
        if intent in ["pricing_question", "metric_question", "customer_case_question", "product_question"]:
            return ["search_knowledge_base", "ask_clarification"]
        if intent == "booking":
            if state.booking.status == "BOOKED_CRM_PENDING":
                return ["write_crm_note"]
            if "AMBIGUOUS_TIMEZONE" in state.risk_flags or "TIMEZONE_OFFSET_NEEDS_CITY" in state.risk_flags:
                return ["ask_clarification"]
            if not state.collected_info.email or not state.collected_info.timezone or not state.collected_info.demo_purpose:
                return ["ask_clarification"]
            if not state.booking.offered_slots:
                return ["check_calendar"]
            if not state.booking.selected_slot_id or not state.booking.selected_slot_confirmed:
                return ["offer_slots", "ask_clarification"]
            return ["book_demo"]
        if intent == "qualification":
            return ["write_crm_note", "ask_clarification"]
        return ["ask_clarification"]


class RuleBasedPlanner:
    def plan(self, conversation: List[ConversationTurn], state: AgentState, intent: str) -> ActionPlan:
        if intent == "handoff_required":
            return self._handoff_plan(conversation, state)
        if intent == "pricing_question":
            return ActionPlan(
                intent=intent,
                tool_calls=[ToolCall(tool_name="search_knowledge_base", arguments={"query": MemoryManager.last_user_message(conversation)})],
                metadata={"response_kind": "pricing"},
            )
        if intent == "metric_question":
            return ActionPlan(
                intent=intent,
                tool_calls=[ToolCall(tool_name="search_knowledge_base", arguments={"query": MemoryManager.last_user_message(conversation)})],
                metadata={"response_kind": "metric"},
            )
        if intent == "customer_case_question":
            return ActionPlan(
                intent=intent,
                tool_calls=[ToolCall(tool_name="search_knowledge_base", arguments={"query": MemoryManager.last_user_message(conversation)})],
                metadata={"response_kind": "customer_case"},
            )
        if intent == "technical_or_security_question":
            return self._technical_or_security_plan(conversation, state)
        if intent == "product_question":
            last_user = MemoryManager.last_user_message(conversation)
            return ActionPlan(
                intent=intent,
                tool_calls=[ToolCall(tool_name="search_knowledge_base", arguments={"query": last_user})],
                metadata={"response_kind": "product", "requested_facets": serialize_requested_facets(extract_requested_facets(last_user))},
            )
        if intent == "booking":
            return self._booking_plan(state)
        if intent == "qualification":
            return self._qualification_plan(state)
        return ActionPlan(intent=intent, metadata={"response_kind": "generic_clarification"})

    def _technical_or_security_plan(self, conversation: List[ConversationTurn], state: AgentState) -> ActionPlan:
        last_user = MemoryManager.last_user_message(conversation)
        facets = extract_requested_facets(last_user)
        facet_text = format_facets(facets) if facets else "技术架构、部署、性能或数据合规问题"
        reason = build_handoff_reason(state, requested_scope=facet_text)
        crm_payload = build_handoff_crm_payload(
            state,
            requested_scope=facet_text,
            next_action="转人工确认技术、安全或数据合规细节",
        )
        return ActionPlan(
            intent="technical_or_security_question",
            tool_calls=[
                ToolCall(tool_name="search_knowledge_base", arguments={"query": last_user}),
                ToolCall(tool_name="write_crm_note", arguments=crm_payload),
                ToolCall(tool_name="handoff_to_human", arguments={"lead_id": state.lead_id, "reason": reason, "urgency": "high"}),
            ],
            metadata={"response_kind": "technical_or_security", "requested_facets": serialize_requested_facets(facets)},
        )

    def _handoff_plan(self, conversation: List[ConversationTurn], state: AgentState) -> ActionPlan:
        last_user = MemoryManager.last_user_message(conversation)
        facets = extract_requested_facets(last_user)
        product_facets = [facet for facet in facets if facet.facet not in ["security_audit", "training_data_compliance", "data_storage", "technical_architecture", "modeling_approach", "api_latency", "private_deployment"]]
        related_scope = format_facets(product_facets) if product_facets else ""
        reason = build_handoff_reason(state)
        crm_payload = build_handoff_crm_payload(state, related_scope=related_scope)
        tool_calls: List[ToolCall] = []
        if product_facets:
            tool_calls.append(ToolCall(tool_name="search_knowledge_base", arguments={"query": last_user}))
        tool_calls.extend(
            [
                ToolCall(tool_name="write_crm_note", arguments=crm_payload),
                ToolCall(tool_name="handoff_to_human", arguments={"lead_id": state.lead_id, "reason": reason, "urgency": "high"}),
            ]
        )
        return ActionPlan(
            intent="handoff_required",
            tool_calls=tool_calls,
            metadata={
                "response_kind": "handoff_with_product" if product_facets else "handoff",
                "requested_facets": serialize_requested_facets(product_facets),
            },
        )

    def _qualification_plan(self, state: AgentState) -> ActionPlan:
        summary = build_summary(state)
        return ActionPlan(
            intent="qualification",
            tool_calls=[
                ToolCall(
                    tool_name="write_crm_note",
                    arguments={
                        "lead_id": state.lead_id,
                        "summary": summary,
                        "customer_pain": customer_pain_for_crm(state),
                        "current_stage": current_stage_for_crm(state),
                        "qualification_level": state.qualification_level,
                        "next_action": state.next_action,
                    },
                )
            ],
            metadata={"response_kind": "qualification"},
        )

    def _booking_plan(self, state: AgentState) -> ActionPlan:
        if "AMBIGUOUS_TIMEZONE" in state.risk_flags:
            return ActionPlan(intent="booking", metadata={"response_kind": "ambiguous_timezone"})
        if "TIMEZONE_OFFSET_NEEDS_CITY" in state.risk_flags:
            return ActionPlan(intent="booking", metadata={"response_kind": "timezone_offset_needs_city"})
        missing = []
        if not state.collected_info.email:
            missing.append("invalid_email" if state.collected_info.invalid_email else "email")
        if not state.collected_info.timezone:
            missing.append("timezone")
        if not state.collected_info.demo_purpose:
            missing.append("demo_purpose")
        if missing:
            return ActionPlan(intent="booking", metadata={"response_kind": "booking_missing_fields", "missing_fields": missing})
        if not state.booking.offered_slots:
            return ActionPlan(
                intent="booking",
                tool_calls=[
                    ToolCall(
                        tool_name="check_calendar",
                        arguments={"timezone": state.collected_info.timezone, "duration_minutes": 30},
                    )
                ],
                metadata={"response_kind": "offer_slots"},
            )
        if not state.booking.selected_slot_id or not state.booking.selected_slot_confirmed:
            return ActionPlan(intent="booking", metadata={"response_kind": "waiting_slot_selection"})
        summary = build_summary(state, include_demo=True)
        return ActionPlan(
            intent="booking",
            tool_calls=[
                ToolCall(
                    tool_name="book_demo",
                    arguments={
                        "lead_id": state.lead_id,
                        "slot_id": state.booking.selected_slot_id,
                        "attendee_email": state.collected_info.email,
                        "summary": summary,
                    },
                ),
                ToolCall(
                    tool_name="write_crm_note",
                    arguments={
                        "lead_id": state.lead_id,
                        "summary": summary,
                        "customer_pain": customer_pain_for_crm(state),
                        "current_stage": current_stage_for_crm(state),
                        "qualification_level": state.qualification_level,
                        "next_action": "Demo booked",
                    },
                ),
            ],
            metadata={"response_kind": "book_demo"},
        )


class LLMPlanner:
    def __init__(
        self,
        agent_version: str,
        client: Optional[ChatClient] = None,
        fallback_planner: Optional[RuleBasedPlanner] = None,
        allow_fallback: bool = True,
    ) -> None:
        self.agent_version = agent_version
        self.client = client or DisabledExternalChatClient()
        self.fallback_planner = fallback_planner or RuleBasedPlanner()
        self.allow_fallback = allow_fallback
        self.prompt_version = PromptTemplateLoader.normalize_version(agent_version)
        self.prompt_path = PromptTemplateLoader.path_for(agent_version)
        self.prompt_text, self.prompt_hash = PromptTemplateLoader.load(agent_version)

    @classmethod
    def with_mock_client(cls, agent_version: str) -> "LLMPlanner":
        return cls(agent_version=agent_version, client=MockLLMClient())

    def plan(self, conversation: List[ConversationTurn], state: AgentState, intent: str) -> ActionPlan:
        messages = self._build_messages(conversation, state, intent)
        repair_attempted = False
        try:
            completion = self.client.complete_json(messages, request_name="plan")
            plan = self._parse_action_plan(completion.content)
        except Exception as first_error:
            repair_attempted = True
            try:
                repair_completion = self.client.complete_json(
                    self._build_repair_messages(messages, str(first_error)),
                    temperature=0,
                    request_name="repair",
                )
                completion = _merge_completion_metadata(repair_completion, {"repair_for": str(first_error)})
                plan = self._parse_action_plan(completion.content)
            except Exception as repair_error:
                if not self.allow_fallback:
                    raise LLMError(f"LLM plan failed after repair: {repair_error}") from repair_error
                plan = self.fallback_planner.plan(conversation, state, intent)
                plan.metadata.update(
                    {
                        "planner": "rule_fallback_after_llm_error",
                        "llm_error": str(repair_error),
                        "prompt_file": self.prompt_path.name,
                        "prompt_version": self.prompt_version,
                        "prompt_hash": self.prompt_hash,
                        "agent_version": self.agent_version,
                        "planner_type": self.__class__.__name__,
                    }
                )
                return plan

        plan.metadata.update(
            {
                "planner": "llm",
                "planner_type": self.__class__.__name__,
                "prompt_file": self.prompt_path.name,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
                "agent_version": self.agent_version,
                "repair_attempted": repair_attempted,
                **completion.metadata,
            }
        )
        return plan

    def prompt_metadata(self) -> Dict[str, Any]:
        return {
            "planner_type": self.__class__.__name__,
            "prompt_version": self.prompt_version,
            "prompt_file": self.prompt_path.name,
            "prompt_hash": self.prompt_hash,
        }

    def _build_messages(self, conversation: List[ConversationTurn], state: AgentState, intent: str) -> List[Dict[str, str]]:
        prompt_contract = {
            "required_json_shape": {
                "intent": intent,
                "assistant_message_draft": "optional safe customer-facing draft",
                "tool_calls": [
                    {
                        "tool_name": "one of search_knowledge_base, check_calendar, book_demo, write_crm_note, handoff_to_human",
                        "arguments": "object matching the selected tool",
                    }
                ],
                "claims": [
                    {
                        "type": (
                            "price|metric|customer_case|feature|technical_architecture|deployment|"
                            "security|data_compliance|delivery_commitment|booking|crm|other"
                        ),
                        "text": "claim text",
                        "evidence_ids": [],
                    }
                ],
                "risk_flags": [],
                "metadata": {
                    "response_kind": (
                        "pricing|metric|customer_case|product|technical_or_security|handoff|generic_clarification|"
                        "qualification|ambiguous_timezone|booking_missing_fields|offer_slots|"
                        "waiting_slot_selection|book_demo"
                    ),
                    "requested_facets": [{"facet": "requested facet id", "phrase": "matched user phrase"}],
                },
            },
            "tool_argument_contracts": _tool_argument_contracts(),
            "hard_rules": [
                "Return exactly one JSON object. Do not wrap it in markdown.",
                "Only include real tool calls, never pseudo-actions such as ask_clarification or offer_slots.",
                "Only propose tool calls whose action is in allowed_actions.",
                "Do not claim booking or CRM success; the Harness will say that only after tool success.",
                "Do not quote prices, metric guarantees, public customer names, or product capabilities unless evidence_policy allows it.",
                "For legal, security, procurement, DPA, SOC2, custom quote, or contract requests, propose write_crm_note and handoff_to_human; CRM note should come first, but handoff must still happen if CRM writing fails.",
            ],
        }
        input_payload = {
            "lead_id": state.lead_id,
            "agent_version": self.agent_version,
            "intent": intent,
            "allowed_actions": list(state.allowed_actions),
            "conversation": [turn.model_dump(mode="json") for turn in conversation],
            "state_card": _state_card(state),
            "prompt_contract": prompt_contract,
        }
        return [
            {"role": "system", "content": self.prompt_text},
            {
                "role": "user",
                "content": (
                    "Use the prompt contract and return valid json only.\n"
                    "INPUT_JSON:\n"
                    f"{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
                ),
            },
        ]

    def _build_repair_messages(self, original_messages: List[Dict[str, str]], error: str) -> List[Dict[str, str]]:
        original_input = original_messages[-1]["content"]
        return [
            {
                "role": "system",
                "content": (
                    "You repair malformed Sales Agent Harness planner output. "
                    "Return exactly one valid JSON object matching ActionPlan. json only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Validation or JSON parse error: {error}\n"
                    "Recreate a valid ActionPlan from the original input. Do not add unsupported claims.\n"
                    f"{original_input}"
                ),
            },
        ]

    @staticmethod
    def _parse_action_plan(raw_content: str) -> ActionPlan:
        raw = _extract_json_object(raw_content)
        normalized = _normalize_plan_payload(raw)
        try:
            return ActionPlan.model_validate(normalized)
        except ValidationError as exc:
            raise LLMError(str(exc)) from exc


class MockLLMPlanner:
    def __init__(
        self,
        prompt_version: str,
        llm: Optional[LocalMockLLM] = None,
        agent_version: Optional[str] = None,
    ) -> None:
        self.agent_version = agent_version or prompt_version
        self.prompt_version = PromptTemplateLoader.normalize_version(prompt_version)
        self.prompt_path = PromptTemplateLoader.path_for(prompt_version)
        self.prompt_text, self.prompt_hash = PromptTemplateLoader.load(prompt_version)
        self.llm = llm or LocalMockLLM()

    def plan(self, conversation: List[ConversationTurn], state: AgentState, intent: str) -> ActionPlan:
        state_card = _state_card(state)
        proposal = self.llm.complete(
            prompt=self.prompt_text,
            state_card=state_card,
            conversation=[turn.model_dump(mode="json") for turn in conversation],
            intent=intent,
        )
        normalized = _normalize_plan_payload(proposal)
        try:
            plan = ActionPlan.model_validate(normalized)
        except ValidationError as exc:
            raise LLMError(str(exc)) from exc
        plan.metadata.update(
            {
                "planner": "mock_llm",
                "planner_type": self.__class__.__name__,
                "prompt_file": self.prompt_path.name,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
                "agent_version": self.agent_version,
                "llm_provider": self.llm.provider,
                "llm_model": self.llm.model,
            }
        )
        return plan

    def prompt_metadata(self) -> Dict[str, Any]:
        return {
            "planner_type": self.__class__.__name__,
            "prompt_version": self.prompt_version,
            "prompt_file": self.prompt_path.name,
            "prompt_hash": self.prompt_hash,
        }


def _has_qualified_info(state: AgentState) -> bool:
    info = state.collected_info
    return bool(info.company_size or info.industry or info.sales_team_size or info.pain_point)


def _prompt_path_for_version(agent_version: str) -> Path:
    return PromptTemplateLoader.path_for(agent_version)


def _state_card(state: AgentState) -> Dict[str, Any]:
    return {
        "lead_id": state.lead_id,
        "qualification_level": state.qualification_level,
        "collected_info": state.collected_info.model_dump(mode="json"),
        "missing_info": list(state.missing_info),
        "next_action": state.next_action,
        "risk_flags": list(state.risk_flags),
        "context_conflicts": list(state.context_conflicts),
        "booking": state.booking.model_dump(mode="json"),
        "crm": state.crm.model_dump(mode="json"),
        "allowed_actions": list(state.allowed_actions),
        "evidence_policy": state.evidence_policy.model_dump(mode="json"),
        "last_kb_evidence_ids": list(state.last_kb_evidence_ids),
        "last_kb_evidence_topics": list(state.last_kb_evidence_topics),
        "last_kb_evidence_coverage": dict(state.last_kb_evidence_coverage),
    }


def _tool_argument_contracts() -> Dict[str, Dict[str, Any]]:
    return {
        "search_knowledge_base": {"query": "string; usually the latest user question"},
        "check_calendar": {"timezone": "IANA timezone string, never ambiguous CST", "duration_minutes": 30},
        "book_demo": {"lead_id": "string", "slot_id": "offered slot id", "attendee_email": "email", "summary": "short CRM-safe summary"},
        "write_crm_note": {
            "lead_id": "string",
            "summary": "non-empty, includes customer need/context",
            "customer_pain": "structured customer pain or '待澄清'",
            "current_stage": "discovery|qualification|demo_pending|demo_booked|handoff_required|crm_sync_pending",
            "qualification_level": "high|medium|low|unknown",
            "next_action": "non-empty operational next action",
        },
        "handoff_to_human": {"lead_id": "string", "reason": "why human is needed", "urgency": "high|medium|low"},
    }


def _extract_json_object(raw_content: str) -> Dict[str, Any]:
    content = raw_content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    if fence_match:
        content = fence_match.group(1)
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            content = content[start : end + 1]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Model JSON root must be an object")
    return parsed


def _normalize_plan_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(raw)
    if "tool_calls" not in payload and "proposed_tool_calls" in payload:
        payload["tool_calls"] = payload.pop("proposed_tool_calls")
    payload.setdefault("assistant_message_draft", "")
    payload.setdefault("tool_calls", [])
    payload.setdefault("claims", [])
    payload.setdefault("risk_flags", [])
    payload.setdefault("metadata", {})
    payload["tool_calls"] = [_normalize_tool_call(item) for item in payload.get("tool_calls") or []]
    return payload


def _normalize_tool_call(raw: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(raw)
    if "tool_name" not in item and "name" in item:
        item["tool_name"] = item.pop("name")
    if "arguments" not in item:
        if "args" in item:
            item["arguments"] = item.pop("args")
        else:
            item["arguments"] = {}
    return item


def _merge_completion_metadata(completion: LLMCompletion, extra: Dict[str, Any]) -> LLMCompletion:
    metadata = dict(completion.metadata)
    metadata.update(extra)
    return LLMCompletion(content=completion.content, metadata=metadata)
