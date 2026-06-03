from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import tools
from .crm_contract import HANDOFF_RISK_FLAGS, build_handoff_payload, build_handoff_reason as build_handoff_reason_from_flags
from .grounding import claim_type_for_facet
from .models import AgentState, CollectedInfo, ConversationTurn, ExtractionResult, ToolResult
from .normalizer import extract_message
from .store import InMemoryStore


class QualificationScorer:
    ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3}

    @classmethod
    def score(cls, info: CollectedInfo, explicit_demo_intent: bool = False) -> str:
        score = 0
        if info.company_size:
            score += 1
        if info.industry:
            score += 1
        if info.sales_team_size:
            score += 1
        if info.pain_point:
            score += 2
        budget_signal = info.budget_amount or info.budget_range or info.budget
        if budget_signal and budget_signal not in ["unknown", "not_set"]:
            budget_wan = _budget_amount_wan(budget_signal)
            if budget_wan is not None and budget_wan >= 100:
                score += 3
            elif budget_wan is not None and budget_wan >= 50:
                score += 2
            else:
                score += 1
        if info.decision_maker:
            score += 2 if str(info.decision_maker).upper() in ["CEO", "CFO", "COO", "CTO", "CIO", "CMO", "VP", "OWNER"] else 1
        if info.go_live_time:
            score += 2
        high_confidence_signal = (
            (budget_signal and budget_signal not in ["unknown", "not_set"])
            or bool(info.decision_maker)
            or bool(info.go_live_time)
            or explicit_demo_intent
        )
        if budget_signal and budget_signal not in ["unknown", "not_set"] and info.decision_maker and info.go_live_time:
            return "high"
        if score >= 5 and high_confidence_signal:
            return "high"
        if score >= 3:
            return "medium"
        if score >= 1:
            return "low"
        return "unknown"

    @classmethod
    def at_least(cls, level: str, minimum: str) -> bool:
        return cls.ORDER.get(level, 0) >= cls.ORDER.get(minimum, 0)


class MemoryManager:
    @staticmethod
    def init_state(lead_id: str) -> AgentState:
        return AgentState(lead_id=lead_id)

    @staticmethod
    def merge_lead_context(state: AgentState, lead_context: Dict[str, Any]) -> AgentState:
        state.lead_context = dict(lead_context)
        info = state.collected_info
        for key in ["company_size", "industry", "pain_point"]:
            if lead_context.get(key) and not getattr(info, key):
                setattr(info, key, str(lead_context[key]))
        if lead_context.get("known_email") and not info.email:
            info.email = lead_context["known_email"]
        if lead_context.get("known_timezone") and not info.timezone:
            info.timezone = lead_context["known_timezone"]
        return MemoryManager.finalize_state(state)

    @staticmethod
    def rebuild_state_from_conversation(state: AgentState, conversation: List[ConversationTurn]) -> AgentState:
        previous_assistant = ""
        for turn in conversation:
            if turn.role == "user":
                extraction = extract_message(turn.content)
                if not extraction.sales_team_size and _assistant_asked_sales_team_size(previous_assistant):
                    inferred_sales_size = _extract_bare_people_count(turn.content)
                    if inferred_sales_size:
                        extraction.sales_team_size = inferred_sales_size
                        if extraction.company_size == inferred_sales_size:
                            extraction.company_size = None
                MemoryManager.merge_extraction(state, extraction)
            else:
                MemoryManager._merge_assistant_slots(state, turn.content)
                previous_assistant = turn.content
        last_user = MemoryManager.last_user_message(conversation)
        if last_user:
            MemoryManager._detect_slot_selection(state, last_user)
        return MemoryManager.finalize_state(state)

    @staticmethod
    def merge_conversation_delta(state: AgentState, conversation: List[ConversationTurn]) -> AgentState:
        start = state.processed_turn_count
        if start > len(conversation):
            start = 0
        if start >= len(conversation):
            latest_user = MemoryManager.last_user_message(conversation)
            if len(conversation) == 1 and latest_user and latest_user != state.last_processed_user_message:
                start = 0
            else:
                return MemoryManager.finalize_state(state)

        previous_assistant = ""
        for prior_turn in conversation[:start]:
            if prior_turn.role == "assistant":
                previous_assistant = prior_turn.content

        last_user_in_delta: Optional[str] = None
        for turn in conversation[start:]:
            if turn.role == "user":
                extraction = extract_message(turn.content)
                if not extraction.sales_team_size and _assistant_asked_sales_team_size(previous_assistant):
                    inferred_sales_size = _extract_bare_people_count(turn.content)
                    if inferred_sales_size:
                        extraction.sales_team_size = inferred_sales_size
                        if extraction.company_size == inferred_sales_size:
                            extraction.company_size = None
                MemoryManager.merge_extraction(state, extraction)
                last_user_in_delta = turn.content
            else:
                previous_assistant = turn.content

        if last_user_in_delta:
            MemoryManager._detect_slot_selection(state, last_user_in_delta)
            state.last_processed_user_message = last_user_in_delta
        state.processed_turn_count = len(conversation)
        return MemoryManager.finalize_state(state)

    @staticmethod
    def merge_extraction(state: AgentState, extraction: ExtractionResult) -> None:
        info = state.collected_info
        for field in [
            "company_size",
            "industry",
            "pain_point",
            "budget",
            "budget_amount",
            "budget_range",
            "decision_maker",
            "go_live_time",
            "email",
            "invalid_email",
            "timezone",
            "demo_purpose",
            "interest_area",
            "sales_team_size",
        ]:
            value = getattr(extraction, field)
            if value:
                MemoryManager._record_context_conflict(state, field, value)
                setattr(info, field, value)
                if field == "email":
                    info.invalid_email = None
        for flag in extraction.risk_flags:
            if flag not in state.risk_flags:
                state.risk_flags.append(flag)
        if extraction.demo_cancelled:
            state.booking.status = "NO_DEMO_INTENT"
            state.booking.offered_slots = []
            state.booking.selected_slot_id = None
            state.booking.selected_slot_confirmed = False
            state.booking.rejected_slot_ids = []
            state.booking.slot_selection_status = "none"
            state.collected_info.demo_purpose = None
        if extraction.demo_intent and state.booking.status == "NO_DEMO_INTENT":
            state.booking.status = "DEMO_INTERESTED"

    @staticmethod
    def _record_context_conflict(state: AgentState, field: str, new_value: str) -> None:
        previous_value = getattr(state.collected_info, field, None)
        context_value = state.lead_context.get(field)
        if context_value is None or previous_value is None:
            return
        context_value = str(context_value)
        if previous_value == context_value and new_value != context_value:
            conflict = {"field": field, "lead_context": context_value, "current_user": new_value}
            if conflict not in state.context_conflicts:
                state.context_conflicts.append(conflict)
            if "CONTEXT_CONFLICT" not in state.risk_flags:
                state.risk_flags.append("CONTEXT_CONFLICT")

    @staticmethod
    def finalize_state(state: AgentState) -> AgentState:
        info = state.collected_info
        state.booking.email = info.email
        state.booking.timezone = info.timezone
        state.booking.demo_purpose = info.demo_purpose
        if state.booking.status in ["NO_DEMO_INTENT", "DEMO_INTERESTED", "READY_TO_CHECK_CALENDAR"]:
            if info.email and info.timezone and info.demo_purpose and state.booking.status != "NO_DEMO_INTENT":
                state.booking.status = "READY_TO_CHECK_CALENDAR"
        if state.booking.offered_slots and state.booking.status in ["READY_TO_CHECK_CALENDAR", "DEMO_INTERESTED"]:
            state.booking.status = "SLOTS_OFFERED"
        if state.booking.selected_slot_id and state.booking.selected_slot_confirmed and state.booking.status == "SLOTS_OFFERED":
            state.booking.status = "SLOT_SELECTED"
        if state.booking.selected_slot_id and state.booking.selected_slot_confirmed and state.booking.status == "BOOKING_FAILED":
            state.booking.status = "SLOT_SELECTED"
        if (not state.booking.selected_slot_id or not state.booking.selected_slot_confirmed) and state.booking.status == "SLOT_SELECTED":
            state.booking.status = "SLOTS_OFFERED" if state.booking.offered_slots else "READY_TO_CHECK_CALENDAR"
        explicit_demo_intent = state.booking.status != "NO_DEMO_INTENT"
        state.qualification_level = QualificationScorer.score(info, explicit_demo_intent=explicit_demo_intent)
        state.missing_info = MemoryManager.compute_missing_info(state)
        state.next_action = MemoryManager.compute_next_action(state)
        return state

    @staticmethod
    def compute_missing_info(state: AgentState) -> List[str]:
        info = state.collected_info
        missing = []
        if not _needs_qualification_followup(state):
            return missing
        for field in ["budget", "decision_maker", "go_live_time"]:
            value = getattr(info, field)
            if value is None or value == "not_set":
                missing.append(field)
        if state.booking.status in ["DEMO_INTERESTED", "READY_TO_CHECK_CALENDAR"]:
            if not info.email and "email" not in missing:
                missing.append("email")
            if not info.timezone and "timezone" not in missing:
                missing.append("timezone")
            if not info.demo_purpose and "demo_purpose" not in missing:
                missing.append("demo_purpose")
        return missing

    @staticmethod
    def compute_next_action(state: AgentState) -> str:
        if any(flag in state.risk_flags for flag in HANDOFF_RISK_FLAGS):
            return "转人工处理高风险材料或条款问题"
        if state.booking.status == "BOOKED_CRM_SYNCED":
            return "Demo booked"
        if state.booking.status == "BOOKED_CRM_PENDING":
            return "CRM_SYNC_PENDING"
        if state.booking.status == "BOOKING_FAILED":
            return "重新提供可选 Demo 时间"
        if state.booking.status == "SLOTS_OFFERED":
            return "等待客户选择 Demo 时间"
        if state.booking.status == "READY_TO_CHECK_CALENDAR":
            return "查询可预约 Demo 时间"
        if state.booking.status == "DEMO_INTERESTED":
            return "补齐 Demo 预约信息"
        if state.qualification_level in ["medium", "high"]:
            missing_labels = {
                "budget": "预算",
                "decision_maker": "决策人",
                "go_live_time": "期望上线时间",
            }
            missing = [missing_labels[field] for field in ["budget", "decision_maker", "go_live_time"] if field in state.missing_info]
            if missing:
                return "继续确认" + "、".join(missing)
            if state.qualification_level == "high":
                return "建议销售跟进，并确认是否安排 Demo 或推进方案评估"
            return "继续了解当前线索来源和 CRM 使用情况，判断是否适合安排 Demo"
        return "澄清客户当前销售流程和主要痛点"

    @staticmethod
    def last_user_message(conversation: List[ConversationTurn]) -> str:
        for turn in reversed(conversation):
            if turn.role == "user":
                return turn.content
        return ""

    @staticmethod
    def _merge_assistant_slots(state: AgentState, content: str) -> None:
        if not state.collected_info.timezone or state.booking.offered_slots:
            return
        if any(token in content for token in ["sg_slot_", "cn_slot_", "周二 10", "周三 15", "可选时间"]):
            state.booking.offered_slots = tools.calendar_slots(state.collected_info.timezone)
            if state.booking.offered_slots and state.booking.status in ["READY_TO_CHECK_CALENDAR", "DEMO_INTERESTED"]:
                state.booking.status = "SLOTS_OFFERED"

    @staticmethod
    def _detect_slot_selection(state: AgentState, last_user: str) -> None:
        if not state.booking.offered_slots:
            return
        parsed = parse_slot_response(last_user, state.booking.offered_slots)
        state.booking.slot_selection_status = parsed["status"]
        if parsed["status"] == "confirmed":
            state.booking.selected_slot_id = parsed["selected_slot_id"]
            state.booking.selected_slot_confirmed = True
            if state.booking.status == "BOOKING_FAILED":
                state.booking.status = "SLOT_SELECTED"
            return
        if parsed["status"] in ["rejected", "reschedule_requested"]:
            for slot_id in parsed["rejected_slot_ids"]:
                if slot_id not in state.booking.rejected_slot_ids:
                    state.booking.rejected_slot_ids.append(slot_id)
            if parsed["rejected_slot_ids"]:
                state.booking.offered_slots = [
                    slot for slot in state.booking.offered_slots if slot["slot_id"] not in parsed["rejected_slot_ids"]
                ]
            state.booking.selected_slot_id = None
            state.booking.selected_slot_confirmed = False
            if state.booking.status in ["SLOT_SELECTED", "BOOKING_FAILED"]:
                state.booking.status = "SLOTS_OFFERED"
            return
        if parsed["status"] == "ambiguous":
            state.booking.selected_slot_id = None
            state.booking.selected_slot_confirmed = False
            if state.booking.status in ["SLOT_SELECTED", "BOOKING_FAILED"]:
                state.booking.status = "SLOTS_OFFERED"


SLOT_REJECTION_TERMS = [
    "不行",
    "不可以",
    "不能",
    "不方便",
    "不合适",
    "没空",
    "沒空",
    "没时间",
    "不要",
    "别",
    "不约",
    "换",
    "换一个",
    "重新",
    "改",
    "取消",
    "除了",
    "被占",
    "占了",
    "冲突",
    "撞",
]
SLOT_RESCHEDULE_TERMS = ["换", "换一个", "改", "重新"]
SLOT_CONFIRMATION_TERMS = ["可以", "就这个", "选这个", "没问题", "確認", "确认", "确定", "OK", "ok", "好的", "行", "可以的"]


def parse_slot_response(text: str, offered_slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = _normalize_time_text(text)
    mentioned_slots = [slot for slot in offered_slots if _slot_mentioned(normalized, slot)]
    mentioned_ids = [slot["slot_id"] for slot in mentioned_slots]
    if len(mentioned_ids) > 1:
        return {"status": "ambiguous", "selected_slot_id": None, "rejected_slot_ids": []}

    if _contains_rejection(normalized) and mentioned_slots:
        return {"status": "rejected", "selected_slot_id": None, "rejected_slot_ids": mentioned_ids}

    if not mentioned_slots:
        if _contains_rejection(normalized) and any(term in normalized for term in ["换", "换一个", "换个", "改", "重新", "不行", "没空"]):
            return {"status": "reschedule_requested", "selected_slot_id": None, "rejected_slot_ids": []}
        return {"status": "none", "selected_slot_id": None, "rejected_slot_ids": []}

    slot = mentioned_slots[0]
    if slot["slot_id"] in text or _contains_confirmation(normalized) or _slot_time_only_confirmation(normalized, slot):
        return {"status": "confirmed", "selected_slot_id": slot["slot_id"], "rejected_slot_ids": []}
    return {"status": "ambiguous", "selected_slot_id": None, "rejected_slot_ids": []}


def _normalize_time_text(text: str) -> str:
    return text.replace("：", ":").replace("点", ":00").replace("點", ":00").replace(" ", "")


def _slot_mentioned(text: str, slot: Dict[str, Any]) -> bool:
    slot_id = slot.get("slot_id", "")
    if slot_id and slot_id in text:
        return True
    display = _normalize_time_text(slot.get("display", ""))
    day_match = re.search(r"(周[一二三四五六日天])", display)
    hour_match = re.search(r"(\d{1,2})(?::00)?", display)
    if not day_match or not hour_match:
        return False
    day = day_match.group(1)
    hour = hour_match.group(1)
    return day in text and bool(re.search(rf"(?<!\d){re.escape(hour)}(?::00)?(?!\d)", text))


def _slot_context_has_rejection(text: str, slot: Dict[str, Any]) -> bool:
    contexts = _slot_contexts(text, slot)
    return any(any(term in context for term in SLOT_REJECTION_TERMS) for context in contexts)


def _slot_contexts(text: str, slot: Dict[str, Any]) -> List[str]:
    segments = [segment for segment in re.split(r"[，,。；;！!？?]", text) if segment]
    contexts = [segment for segment in segments if _slot_mentioned(segment, slot)]
    return contexts or ([text] if _slot_mentioned(text, slot) else [])


def _contains_rejection(text: str) -> bool:
    return any(term in text for term in SLOT_REJECTION_TERMS)


def _contains_reschedule(text: str) -> bool:
    return any(term in text for term in SLOT_RESCHEDULE_TERMS)


def _contains_confirmation(text: str) -> bool:
    return any(term in text for term in SLOT_CONFIRMATION_TERMS)


def _slot_time_only_confirmation(text: str, slot: Dict[str, Any]) -> bool:
    display = _normalize_time_text(slot.get("display", ""))
    day_match = re.search(r"(周[一二三四五六日天])", display)
    hour_match = re.search(r"(\d{1,2})(?::00)?", display)
    if not day_match or not hour_match:
        return False
    slot_id = slot.get("slot_id", "")
    stripped = text.replace(slot_id, "") if slot_id else text
    stripped = re.sub(r"[，,。；;！!？?\s]", "", stripped)
    day = day_match.group(1)
    hour = hour_match.group(1)
    return bool(re.fullmatch(rf"{re.escape(day)}{re.escape(hour)}(?::00)?", stripped))


class BookingStateMachine:
    @staticmethod
    def apply_tool_result(state: AgentState, result: ToolResult, store: Optional[InMemoryStore] = None) -> AgentState:
        active_store = store if store is not None else InMemoryStore()
        if result.tool_name == "search_knowledge_base":
            topic = result.data.get("topic") if result.success else None
            docs = result.data.get("documents", []) if result.success else []
            evidence_ids = result.data.get("evidence_ids", []) if result.success else []
            if result.success:
                topics = result.data.get("topics") or ([topic] if topic else [])
                state.last_kb_evidence_ids = list(evidence_ids)
                state.last_kb_evidence_topics = list(topics)
                state.last_kb_evidence_coverage = _kb_evidence_coverage(docs)
                state.last_kb_query = str(result.arguments.get("query") or result.data.get("query") or "")
                for evidence_id in evidence_ids:
                    if evidence_id not in state.evidence_policy.evidence_ids:
                        state.evidence_policy.evidence_ids.append(evidence_id)
                _apply_evidence_policy(state, topic, docs)

        if result.tool_name == "check_calendar":
            if result.success:
                state.booking.offered_slots = result.data.get("slots", [])
                state.booking.selected_slot_id = None
                state.booking.selected_slot_confirmed = False
                state.booking.rejected_slot_ids = []
                state.booking.slot_selection_status = "none"
                state.booking.status = "SLOTS_OFFERED"
                state.allowed_actions = ["offer_slots", "ask_clarification"]
                state.add_trajectory(
                    step="booking_slots_persisted",
                    data={
                        "source": "tool_result",
                        "tool_name": "check_calendar",
                        "offered_slot_ids": [slot.get("slot_id") for slot in state.booking.offered_slots],
                    },
                )
            else:
                state.booking.status = "DEMO_INTERESTED"
                state.allowed_actions = ["ask_clarification"]

        if result.tool_name == "book_demo":
            if result.success:
                state.booking.book_demo_success = True
                state.booking.status = "BOOKED_CRM_PENDING"
                state.booking.calendar_event_id = result.data.get("calendar_event_id")
                state.booking.selected_slot_id = result.data.get("slot_id") or state.booking.selected_slot_id
                payload = build_booking_crm_payload(state)
                tools.enqueue_crm_after_booking(
                    active_store,
                    state.lead_id,
                    state.booking.calendar_event_id or "",
                    result.data.get("idempotency_key", ""),
                    payload,
                )
                state.allowed_actions = ["write_crm_note"]
                state.outbox = active_store.list_outbox(state.lead_id)
            else:
                failed_slot_id = result.arguments.get("slot_id")
                if failed_slot_id and failed_slot_id not in state.booking.rejected_slot_ids:
                    state.booking.rejected_slot_ids.append(failed_slot_id)
                if failed_slot_id:
                    state.booking.offered_slots = [slot for slot in state.booking.offered_slots if slot.get("slot_id") != failed_slot_id]
                state.booking.book_demo_success = False
                state.booking.selected_slot_id = None
                state.booking.selected_slot_confirmed = False
                state.booking.status = "BOOKING_FAILED"
                state.allowed_actions = ["offer_slots", "ask_clarification"]

        if result.tool_name == "write_crm_note":
            if result.success:
                state.crm.status = "WRITTEN"
                state.crm.last_summary = result.arguments.get("summary")
                state.crm.last_qualification_level = result.arguments.get("qualification_level")
                state.crm.last_next_action = result.arguments.get("next_action")
                state.crm.last_current_stage = result.data.get("current_stage")
                state.crm.last_customer_pain = result.data.get("customer_pain")
                state.crm.last_note_id = result.data.get("note_id")
                if state.booking.book_demo_success and result.arguments.get("next_action") == "Demo booked":
                    state.booking.status = "BOOKED_CRM_SYNCED"
                    state.allowed_actions = []
                    tools.mark_outbox_done(active_store, calendar_event_id=state.booking.calendar_event_id or "")
                    state.outbox = active_store.list_outbox(state.lead_id)
            else:
                state.crm.status = "PENDING" if state.booking.book_demo_success else "FAILED"
                if state.booking.book_demo_success:
                    tools.mark_outbox_error(active_store, calendar_event_id=state.booking.calendar_event_id or "", error=result.error)
                state.outbox = active_store.list_outbox(state.lead_id)

        return MemoryManager.finalize_state(state)


def build_booking_crm_payload(state: AgentState) -> Dict[str, Any]:
    return {
        "lead_id": state.lead_id,
        "summary": build_summary(state, include_demo=True),
        "customer_pain": customer_pain_for_crm(state),
        "current_stage": current_stage_for_crm(state),
        "qualification_level": state.qualification_level,
        "next_action": "Demo booked",
    }


def build_handoff_reason(state: AgentState, requested_scope: str = "") -> str:
    return build_handoff_reason_from_flags(state.risk_flags, requested_scope)


def build_handoff_crm_payload(
    state: AgentState,
    requested_scope: str = "",
    related_scope: str = "",
    next_action: str = "转人工处理法务、安全或商务条款确认",
) -> Dict[str, Any]:
    return build_handoff_payload(
        lead_id=state.lead_id,
        qualification_level=state.qualification_level,
        customer_pain=customer_pain_for_crm(state),
        risk_flags=state.risk_flags,
        requested_scope=requested_scope,
        related_scope=related_scope,
        next_action=next_action,
    )


def build_summary(state: AgentState, include_demo: bool = False) -> str:
    info = state.collected_info
    known: List[str] = []
    if info.company_size:
        known.append(f"{info.company_size}人公司")
    if info.industry:
        known.append(info.industry)
    if info.sales_team_size:
        known.append(f"销售团队{info.sales_team_size}人")
    if info.budget == "not_set":
        known.append("预算未定")
    elif info.budget_amount:
        known.append(f"预算金额 {info.budget_amount}")
    elif info.budget_range:
        known.append(f"预算范围 {info.budget_range}")
    elif info.budget:
        known.append(f"预算状态 {info.budget}")
    if info.decision_maker:
        known.append(f"决策人信号 {info.decision_maker}")
    if info.go_live_time:
        known.append(f"期望上线时间 {info.go_live_time}")
    if info.demo_purpose:
        known.append(f"Demo目的 {info.demo_purpose}")
    elif info.interest_area:
        known.append(f"关注场景 {info.interest_area}")
    if include_demo and state.booking.selected_slot_id:
        known.append(f"Demo时间已选择 {state.booking.selected_slot_id}")
    pain_point = info.pain_point or customer_pain_for_crm(state) or "unknown"
    stage = current_stage_for_crm(state)
    missing = "、".join(state.missing_info) if state.missing_info else "无"
    known_text = "，".join(known) if known else "unknown"
    return f"痛点：{pain_point}。当前阶段：{stage}。已知信息：{known_text}。缺失信息：{missing}。"


def customer_pain_for_crm(state: AgentState) -> str:
    info = state.collected_info
    if info.pain_point:
        return info.pain_point
    if info.demo_purpose:
        return f"Demo关注：{info.demo_purpose}"
    if info.interest_area:
        return f"关注场景：{info.interest_area}"
    if any(flag in state.risk_flags for flag in HANDOFF_RISK_FLAGS):
        return "高风险材料或商务条款需人工确认"
    return "待澄清"


def current_stage_for_crm(state: AgentState) -> str:
    if any(flag in state.risk_flags for flag in HANDOFF_RISK_FLAGS):
        return "handoff_required"
    if state.booking.book_demo_success or state.booking.status in ["BOOKED_CRM_PENDING", "BOOKED_CRM_SYNCED"]:
        return "demo_booked"
    if state.booking.status in ["DEMO_INTERESTED", "READY_TO_CHECK_CALENDAR", "SLOTS_OFFERED", "SLOT_SELECTED", "BOOKING_FAILED", "BOOKED_CRM_PENDING"]:
        return "demo_pending"
    if _needs_qualification_followup(state):
        return "qualification"
    return "discovery"


def _assistant_asked_sales_team_size(content: str) -> bool:
    return bool(content and "销售" in content and any(token in content for token in ["多大", "多少", "几人", "几个人"]))


def _extract_bare_people_count(content: str) -> Optional[str]:
    match = re.search(r"(\d+)\s*人", content)
    return match.group(1) if match else None


def _needs_qualification_followup(state: AgentState) -> bool:
    info = state.collected_info
    if state.booking.status != "NO_DEMO_INTENT":
        return True
    if any(flag in state.risk_flags for flag in HANDOFF_RISK_FLAGS):
        return True
    return bool(
        info.company_size
        or info.industry
        or info.sales_team_size
        or info.pain_point
        or info.budget
        or info.budget_amount
        or info.budget_range
        or info.decision_maker
        or info.go_live_time
    )


def _budget_amount_wan(budget: Optional[str]) -> Optional[float]:
    if not budget:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*万", budget)
    if match:
        return float(match.group(1))
    return None


def _apply_evidence_policy(state: AgentState, topic: Optional[str], docs: List[Dict[str, Any]]) -> None:
    for doc in docs:
        policy = doc.get("policy")
        if not isinstance(policy, dict):
            policy = {}
        evidence_id = doc.get("id")
        if evidence_id:
            span = doc.get("content") or doc.get("title") or ""
            if span:
                state.evidence_policy.evidence_spans[evidence_id] = span
            for facet in doc.get("supported_facets") or []:
                ids = state.evidence_policy.supported_facets.setdefault(facet, [])
                if evidence_id not in ids:
                    ids.append(evidence_id)
            for facet in doc.get("unsupported_facets") or []:
                ids = state.evidence_policy.unsupported_facets.setdefault(facet, [])
                if evidence_id not in ids:
                    ids.append(evidence_id)
        if policy.get("price_policy") in ["no_public_price", "exact_price_supported", "unknown"]:
            state.evidence_policy.price_policy = policy["price_policy"]
        if policy.get("metric_policy") in ["no_guaranteed_metric", "exact_metric_supported", "unknown"]:
            state.evidence_policy.metric_policy = policy["metric_policy"]
        if policy.get("customer_case_policy") in ["no_public_customer_name", "public_case_supported", "unknown"]:
            state.evidence_policy.customer_case_policy = policy["customer_case_policy"]
        if policy.get("feature_policy") in ["feature_supported", "needs_confirmation", "unknown"]:
            state.evidence_policy.feature_policy = policy["feature_policy"]

    if not docs:
        if topic in ["lead_follow_up", "crm_integration"]:
            state.evidence_policy.feature_policy = "needs_confirmation"
        return


def _kb_evidence_coverage(docs: List[Dict[str, Any]]) -> Dict[str, bool]:
    coverage: Dict[str, bool] = {}
    for doc in docs:
        for facet in doc.get("supported_facets") or []:
            coverage[claim_type_for_facet(str(facet))] = True
    return coverage
