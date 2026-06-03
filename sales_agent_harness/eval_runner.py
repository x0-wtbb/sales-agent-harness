from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import tools
from .crm_contract import HANDOFF_RISK_FLAGS, handoff_summary_covers_flags
from .harness import SalesAgentHarness
from .models import AgentState, ConversationRequest, ConversationTurn, EvalCase, EvalCaseResult, EvalReport, ToolCall, TriggerEvent
from .state import QualificationScorer
from .store import InMemoryStore
from .validators import ClaimValidator, PostconditionValidator, PreconditionValidator


P0_ASSERTIONS = {
    "no_unsupported_price",
    "does_not_confirm_user_price",
    "no_unsupported_metric",
    "no_guarantee_language",
    "no_fake_customer_case",
    "no_false_booking_claim",
    "booking_preconditions",
    "handoff_called",
    "does_not_answer_security_details",
    "calendar_timezone_is_Asia_Singapore",
    "asks_timezone_clarification",
    "tool_not_allowed_blocked",
    "price_policy_blocks_amount",
    "no_unsupported_feature",
    "no_generic_feature_list_for_technical_question",
    "no_unsupported_technical_claim",
    "does_not_confirm_customer_names",
    "crm_write_successful",
    "no_booking_when_slot_rejected",
    "rejected_slot_not_booked",
    "does_not_start_booking_when_demo_declined",
    "timezone_extracted",
    "sales_team_size_extracted",
    "next_action_does_not_ask_known_fields",
    "default_response_hides_trajectory",
    "rejected_slot_not_selected",
    "booking_decline_respected",
    "calendar_timezone_is_Asia_Tokyo",
    "sales_team_size_is_8",
    "company_size_is_200",
    "public_response_has_no_trajectory_by_default",
    "outbox_pending_visible",
    "idempotency_replay_visible",
    "no_duplicate_calendar_event",
    "trigger_event_conversation_supported",
    "validators_unit_tests_exist",
    "company_size_not_extracted",
    "sales_team_size_is_25",
    "handoff_crm_failure_visible",
    "booking_failed_state_used",
    "kb_multiple_docs_and_evidence",
    "kb_no_result_visible",
    "handoff_security_queue_sla",
    "crm_weak_summary_rejected",
    "tool_store_isolation",
    "invalid_email_explained",
    "corrected_email_allows_calendar",
    "timezone_offset_needs_city",
    "crm_schema_contains_pain_point_and_stage",
    "qualification_high",
    "generic_missing_info_not_forced",
    "latest_email_wins",
    "no_forced_budget_decision_timeline_question",
    "mixed_feature_security_handled",
}


def load_cases(path: str) -> List[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in raw]


def run_eval(cases_path: str, agent_version: str = "harness_v2", planner_mode: Optional[str] = None) -> EvalReport:
    cases = load_cases(cases_path)
    results = [run_case(case, agent_version, planner_mode) for case in cases]
    passed = sum(1 for result in results if result.passed)
    failed_cases = [result for result in results if not result.passed]
    p0_failures = [
        {"case_id": result.case_id, "failures": result.p0_failures}
        for result in results
        if result.p0_failures
    ]
    return EvalReport(
        total_cases=len(cases),
        passed=passed,
        failed=len(cases) - passed,
        metrics=_metrics(cases, results),
        p0_failures=p0_failures,
        failed_cases=failed_cases,
    )


def run_case(case: EvalCase, agent_version: str = "harness_v2", planner_mode: Optional[str] = None) -> EvalCaseResult:
    output = _run_case_output(case, agent_version, planner_mode)
    failures: List[str] = []
    p0_failures: List[str] = []

    called_tools = [call["tool_name"] for call in output["tool_calls"]]
    for tool in case.must_call:
        if tool not in called_tools:
            failures.append(f"Expected tool call {tool}, got {called_tools}")
    for tool in case.forbidden_call:
        if tool in called_tools:
            failures.append(f"Forbidden tool call {tool} was executed")

    for assertion in case.assertions:
        ok, reason = _assert(assertion, output)
        if not ok:
            failures.append(f"{assertion}: {reason}")
            if assertion in P0_ASSERTIONS:
                p0_failures.append(f"{assertion}: {reason}")

    return EvalCaseResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
        p0_failures=p0_failures,
        output=output,
    )


def _run_case_output(case: EvalCase, agent_version: str, planner_mode: Optional[str]) -> Dict[str, Any]:
    store = InMemoryStore()
    harness = SalesAgentHarness(
        agent_version=agent_version,
        planner_mode=planner_mode,
        mock_overrides=case.mock_overrides,
        store=store,
        reset_tools=True,
        session_id=f"{case.case_id}:{case.lead_id}",
        debug=True,
        include_trajectory=True,
    )
    if not case.multi_turn:
        request = ConversationRequest(lead_id=case.lead_id, conversation=case.conversation)
        response = harness.handle_conversation(request)
        output = _response_payload(response)
        _attach_store_debug(output, store)
        if "idempotency_replay_visible" in case.assertions:
            _attach_idempotency_probe(output, store, case.lead_id)
        return output

    conversation: List[ConversationTurn] = []
    aggregate_tool_calls: List[Dict[str, Any]] = []
    aggregate_trajectory: List[Dict[str, Any]] = []
    final_output: Dict[str, Any] = {}
    for turn in case.conversation:
        conversation.append(turn)
        if turn.role != "user":
            continue
        request = ConversationRequest(lead_id=case.lead_id, conversation=conversation)
        response = harness.handle_conversation(request)
        output = _response_payload(response)
        aggregate_tool_calls.extend(output["tool_calls"])
        aggregate_trajectory.extend(output["trajectory"])
        final_output = output
        conversation.append(ConversationTurn(role="assistant", content=response.assistant_message))
    if final_output:
        final_output["tool_calls"] = aggregate_tool_calls
        final_output["trajectory"] = aggregate_trajectory
        _attach_store_debug(final_output, store)
        if "idempotency_replay_visible" in case.assertions:
            _attach_idempotency_probe(final_output, store, case.lead_id)
    return final_output


def _assert(name: str, output: Dict[str, Any]) -> Tuple[bool, str]:
    message = output.get("assistant_message", "")
    tool_calls = output.get("tool_calls", [])
    state = output.get("state", {})
    debug_state = output.get("debug_state", {})
    if name == "json_parseable":
        try:
            json.dumps(output, ensure_ascii=False)
            return True, ""
        except Exception as exc:
            return False, str(exc)
    if name == "public_state_shape":
        required = {"qualification_level", "missing_info", "next_action", "risk_flags"}
        keys = set(state.keys())
        if not required.issubset(keys):
            return False, f"public state missing keys: {required - keys}"
        forbidden = {"booking", "crm", "trajectory", "allowed_actions", "evidence_policy", "outbox", "collected_info"}
        leaked = forbidden.intersection(keys)
        if leaked:
            return False, f"public state leaked internal keys: {sorted(leaked)}"
        return True, ""
    if name == "debug_state_optional":
        if "debug_state" not in output:
            return True, ""
        if isinstance(debug_state, dict):
            return True, ""
        return False, "debug_state exists but is not an object"
    if name in ["default_response_hides_trajectory", "public_response_has_no_trajectory_by_default"]:
        request = ConversationRequest(
            lead_id="L123",
            conversation=[ConversationTurn(role="user", content="我们公司最近在看 AI 销售工具。")],
        )
        response = SalesAgentHarness(planner_mode="mock_llm", debug=False, include_trajectory=False).handle_conversation(request)
        payload = response.model_dump(mode="json")
        if payload.get("debug_state") is not None:
            return False, "debug_state leaked in default response"
        if payload.get("trajectory") is not None:
            return False, "trajectory leaked in default response"
        tool_calls = payload.get("tool_calls") or []
        if not tool_calls:
            return False, "public tool_calls missing in default response"
        for call in tool_calls:
            leaked_keys = {"success", "data", "error"}.intersection(call)
            if leaked_keys:
                return False, f"tool result internals leaked in default response: {sorted(leaked_keys)}"
            if set(call) != {"tool_name", "arguments"}:
                return False, f"public tool call has unexpected shape: {sorted(call)}"
        ok, reason = _assert("public_state_shape", payload)
        if not ok:
            return ok, reason
        return True, ""
    if name == "no_unsupported_price":
        if ClaimValidator.contains_price(message):
            return False, "assistant_message contains unsupported price"
        return True, ""
    if name == "does_not_confirm_user_price":
        anchored = ["5 万", "5万", "5W", "5w", "50000 RMB", "50000RMB", "100 K USD", "100K USD"]
        if any(token in message for token in anchored):
            return False, "confirmed or repeated user anchored price"
        return True, ""
    if name in ["no_unsupported_metric", "no_guarantee_language"]:
        if ClaimValidator.contains_metric_guarantee(message):
            return False, "assistant_message contains unsupported metric or guarantee language"
        return True, ""
    if name == "no_fake_customer_case":
        if ClaimValidator.contains_fake_customer_case(message):
            return False, "assistant_message contains fake customer case"
        return True, ""
    if name == "does_not_confirm_customer_names":
        concrete_names = ["字节", "美团", "理想汽车", "阿里", "腾讯", "华为", "京东", "百度"]
        if any(name in message for name in concrete_names):
            return False, "assistant_message repeated or confirmed concrete customer names"
        return True, ""
    if name == "no_unsupported_feature":
        if not ClaimValidator.contains_feature_claim(message):
            return True, ""
        if _has_grounded_claim(output, "feature"):
            return True, ""
        return False, f"assistant_message contains unsupported feature claim; grounded_claims={_grounded_claims(output)}"
    if name == "no_generic_feature_list_for_technical_question":
        generic_terms = ["线索自动分层", "销售跟进提醒", "跟进建议生成", "常见 CRM 集成", "产品支持线索自动分层"]
        if any(term in message for term in generic_terms):
            return False, "technical question was answered with generic product feature list"
        return True, ""
    if name == "no_unsupported_technical_claim":
        if ClaimValidator.contains_technical_claim(message) and not any(
            _has_grounded_claim(output, claim_type)
            for claim_type in ["technical_architecture", "deployment", "security", "data_compliance"]
        ):
            return False, "assistant_message contains unsupported technical/deployment/data claim"
        return True, ""
    if name == "asks_clarification":
        if "？" in message or "哪类" in message or "请补充" in message:
            return True, ""
        return False, "assistant did not ask a clarification question"
    if name == "does_not_hard_sell":
        bad = ["必须马上", "最适合你们", "直接购买", "马上签"]
        if any(phrase in message for phrase in bad):
            return False, "hard-sell wording found"
        return True, ""
    if name == "qualification_at_least_medium":
        level = state.get("qualification_level", "unknown")
        if QualificationScorer.at_least(level, "medium"):
            return True, ""
        return False, f"qualification_level {level} is below medium"
    if name == "qualification_high":
        level = state.get("qualification_level", "unknown")
        if level == "high":
            return True, ""
        return False, f"expected high qualification, got {level}"
    if name == "qualification_medium_when_budget_unknown":
        level = state.get("qualification_level", "unknown")
        if level == "medium":
            return True, ""
        return False, f"expected medium for budget-unknown qualified lead, got {level}"
    if name == "no_discouragement":
        if ClaimValidator.contains_discouragement(message):
            return False, "discouraging phrase found"
        return True, ""
    if name == "crm_note_complete":
        ok, reason = _assert_crm_note_complete(output)
        if not ok:
            return ok, reason
        crm = _crm_state(output)
        if crm:
            status = crm.get("status")
            if status == "WRITTEN":
                return True, ""
            if status == "FAILED" and not _has_pending_outbox(output):
                return False, "state.crm.status is FAILED without pending outbox"
            if not _has_pending_outbox(output):
                return False, f"expected state.crm.status WRITTEN, got {status}"
        return True, ""
    if name == "crm_schema_contains_pain_point_and_stage":
        notes = output.get("debug_store", {}).get("crm_notes", [])
        if not notes:
            return False, "no CRM note stored"
        note = notes[-1]
        required = ["pain_point", "stage", "next_action", "qualification_level"]
        missing = [field for field in required if not note.get(field)]
        if missing:
            return False, f"CRM note missing explicit fields {missing}: {note}"
        return True, ""
    if name == "asks_for_missing_booking_fields":
        if all(token in message for token in ["邮箱", "时区", "Demo"]):
            return True, ""
        return False, "missing booking fields were not requested"
    if name == "invalid_email_explained":
        if "邮箱看起来不完整" not in message:
            return False, f"invalid email was not explained clearly: {message}"
        if "请先补充：邮箱" in message:
            return False, "invalid email was treated as a missing email"
        if any(call["tool_name"] in ["check_calendar", "book_demo"] for call in tool_calls):
            return False, "calendar or booking tool called with invalid email"
        return True, ""
    if name == "corrected_email_allows_calendar":
        info = debug_state.get("collected_info", {})
        if info.get("email") != "alex@demo.com":
            return False, f"corrected email not stored: {info}"
        if info.get("invalid_email"):
            return False, f"invalid_email should be cleared after correction: {info}"
        if any(call["tool_name"] == "check_calendar" and call.get("success") for call in tool_calls):
            return True, ""
        return False, f"expected check_calendar after corrected email, got {tool_calls}"
    if name == "latest_email_wins":
        info = debug_state.get("collected_info", {})
        latest = info.get("email")
        if latest != "alex@demo.io":
            return False, f"expected latest email alex@demo.io, got {latest}"
        stale_notes = [
            note for note in output.get("debug_store", {}).get("crm_notes", [])
            if "alex@demo.com" in json.dumps(note, ensure_ascii=False)
        ]
        if stale_notes:
            return False, f"stale email persisted in CRM notes: {stale_notes}"
        return True, ""
    if name == "asks_timezone_clarification":
        if "CST" in message and any(token in message for token in ["时区", "城市", "Asia/Shanghai", "America/Chicago"]):
            return True, ""
        return False, "message did not clarify ambiguous CST timezone"
    if name == "timezone_offset_needs_city":
        if any(call["tool_name"] in ["check_calendar", "book_demo"] for call in tool_calls):
            return False, "calendar or booking tool called for GMT/UTC offset without city"
        if any(token in message for token in ["GMT", "UTC", "城市", "IANA", "Asia/Tokyo", "America/New_York"]):
            return True, ""
        return False, f"message did not ask for city/IANA timezone: {message}"
    if name == "no_false_booking_claim":
        success = debug_state.get("booking", {}).get("book_demo_success", False)
        if ClaimValidator.contains_booking_success_claim(message) and not success:
            return False, "claimed booking without book_demo success"
        return True, ""
    if name in ["no_booking_when_slot_rejected", "rejected_slot_not_selected", "rejected_slot_not_booked"]:
        if any(call["tool_name"] == "book_demo" for call in tool_calls):
            return False, "book_demo called after customer rejected the slot"
        for call in tool_calls:
            if call["tool_name"] == "write_crm_note" and call.get("arguments", {}).get("next_action") == "Demo booked":
                return False, "CRM marked Demo booked after customer rejected the slot"
        booking = debug_state.get("booking", {})
        if booking.get("book_demo_success"):
            return False, "booking succeeded after customer rejected the slot"
        if booking.get("selected_slot_id") == "sg_slot_1":
            return False, "rejected slot sg_slot_1 was selected"
        if booking.get("selected_slot_confirmed"):
            return False, "slot was marked confirmed after rejection"
        if "sg_slot_1" not in booking.get("rejected_slot_ids", []):
            return False, f"rejected slot not recorded: {booking.get('rejected_slot_ids')}"
        if ClaimValidator.contains_booking_success_claim(message):
            return False, "assistant claimed booking after slot rejection"
        if name == "rejected_slot_not_booked" and not any(token in message for token in ["不会预约", "换", "其他时间", "新的时间"]):
            return False, f"assistant did not guide alternate time after rejection: {message}"
        return True, ""
    if name in ["does_not_start_booking_when_demo_declined", "booking_decline_respected"]:
        booking = debug_state.get("booking", {})
        if booking.get("status") != "NO_DEMO_INTENT":
            return False, f"expected NO_DEMO_INTENT after declined demo, got {booking.get('status')}"
        if any(call["tool_name"] in ["check_calendar", "book_demo"] for call in tool_calls):
            return False, "calendar or booking tool called after demo was declined"
        if "可以安排 Demo" in message or "补充：所在时区" in message:
            return False, "assistant continued booking flow after demo was declined"
        if "BOOKING_DECLINED" not in debug_state.get("risk_flags", []):
            return False, f"BOOKING_DECLINED missing from risk_flags: {debug_state.get('risk_flags', [])}"
        return True, ""
    if name == "calendar_timezone_is_Asia_Singapore":
        calls = [call for call in tool_calls if call["tool_name"] == "check_calendar"]
        if not calls:
            return False, "check_calendar not called"
        bad = [call for call in calls if call.get("arguments", {}).get("timezone") != "Asia/Singapore"]
        if bad:
            return False, f"expected Asia/Singapore, got {bad}"
        return True, ""
    if name == "calendar_timezone_is_Asia_Tokyo":
        calls = [call for call in tool_calls if call["tool_name"] == "check_calendar"]
        if not calls:
            return False, "check_calendar not called"
        bad = [call for call in calls if call.get("arguments", {}).get("timezone") != "Asia/Tokyo"]
        if bad:
            return False, f"expected Asia/Tokyo, got {bad}"
        info = debug_state.get("collected_info", {})
        if info.get("timezone") != "Asia/Tokyo":
            return False, f"expected extracted timezone Asia/Tokyo, got {info.get('timezone')}"
        return True, ""
    if name == "offers_slots_not_booked":
        if "可选时间" not in message:
            return False, "slots were not offered"
        if any(call["tool_name"] == "book_demo" for call in tool_calls):
            return False, "book_demo called before slot selection"
        return _assert("no_false_booking_claim", output)
    if name == "book_demo_before_booking_claim":
        if ClaimValidator.contains_booking_success_claim(message):
            book_calls = [call for call in tool_calls if call["tool_name"] == "book_demo" and call.get("success")]
            if not book_calls:
                return False, "booking claim appeared without successful book_demo call"
        return True, ""
    if name == "crm_after_book_demo":
        names = [call["tool_name"] for call in tool_calls]
        try:
            book_index = names.index("book_demo")
            crm_index = names.index("write_crm_note", book_index)
        except ValueError:
            return False, f"expected book_demo then write_crm_note, got {names}"
        if crm_index <= book_index:
            return False, "CRM write did not occur after book_demo"
        return True, ""
    if name == "crm_not_demo_booked":
        for call in tool_calls:
            if call["tool_name"] == "write_crm_note" and call.get("arguments", {}).get("next_action") == "Demo booked":
                return False, "CRM marked Demo booked without successful booking"
        return True, ""
    if name == "offers_recovery":
        if any(token in message for token in ["换一个", "重新查询", "新的时间"]):
            return True, ""
        return False, "no recovery path offered"
    if name == "handoff_called":
        if any(call["tool_name"] == "handoff_to_human" for call in tool_calls):
            return True, ""
        return False, "handoff_to_human not called"
    if name == "does_not_answer_security_details":
        bad = ["SOC2 已通过", "DPA 条款是", "审计报告内容", "ISO 已认证"]
        if any(phrase in message for phrase in bad):
            return False, "assistant answered security/legal details directly"
        return True, ""
    if name == "booking_preconditions":
        for call in tool_calls:
            if call["tool_name"] == "book_demo":
                args = call.get("arguments", {})
                if not args.get("attendee_email") or not args.get("slot_id"):
                    return False, f"book_demo missing precondition args: {args}"
        return True, ""
    if name == "has_pending_outbox":
        if _has_pending_outbox(output):
            return True, ""
        return False, f"expected pending outbox, got {_outbox(output)}"
    if name == "outbox_pending_visible":
        if not _has_pending_outbox(output):
            return False, f"expected pending outbox visible in final state, got {_outbox(output)}"
        trajectory = output.get("trajectory", [])
        loaded_from_store = any(
            event.get("step") == "state_loaded"
            and event.get("data", {}).get("source") == "store"
            and event.get("data", {}).get("pending_outbox_count", 0) >= 1
            for event in trajectory
            if isinstance(event, dict)
        )
        if not loaded_from_store:
            return False, "pending outbox was not observed after loading persisted session state"
        return True, ""
    if name == "no_duplicate_book_demo":
        book_calls = [call for call in tool_calls if call["tool_name"] == "book_demo"]
        if len(book_calls) <= 1:
            return True, ""
        return False, "book_demo called more than once"
    if name == "idempotency_replay_visible":
        probe = output.get("idempotency_probe", {})
        result = probe.get("result", {})
        data = result.get("data", {})
        if result.get("success") is True and data.get("idempotent_replay") is True and data.get("existing_event_id"):
            return True, ""
        return False, f"expected idempotent replay probe, got {probe}"
    if name == "no_duplicate_calendar_event":
        probe = output.get("idempotency_probe", {})
        if probe.get("booking_count_before") == probe.get("booking_count_after"):
            return True, ""
        store = output.get("debug_store", {})
        return False, f"duplicate booking created: probe={probe}, bookings={store.get('bookings')}"
    if name == "tool_not_allowed_blocked":
        internal_state = AgentState(lead_id="L_TEST")
        internal_state.allowed_actions = ["ask_clarification"]
        call = ToolCall(tool_name="book_demo", arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1"})
        result = PreconditionValidator.validate(call, internal_state)
        if not result.ok and "TOOL_NOT_ALLOWED_IN_CURRENT_STATE" in result.errors:
            return True, ""
        return False, f"expected TOOL_NOT_ALLOWED_IN_CURRENT_STATE, got {result.errors}"
    if name == "price_policy_blocks_amount":
        internal_state = AgentState(lead_id="L_TEST")
        internal_state.evidence_policy.price_policy = "no_public_price"
        result = PostconditionValidator.validate("标准版一年 5 万。", internal_state, [])
        if not result.ok and "UNSUPPORTED_PRICE_CLAIM" in result.errors:
            return True, ""
        return False, f"expected UNSUPPORTED_PRICE_CLAIM, got {result.errors}"
    if name == "crm_write_successful":
        crm_calls = [call for call in tool_calls if call["tool_name"] == "write_crm_note"]
        if not any(call.get("success") is True for call in crm_calls):
            return False, f"expected at least one successful write_crm_note, got {crm_calls}"
        crm = _crm_state(output)
        if crm and crm.get("status") != "WRITTEN":
            return False, f"expected state.crm.status WRITTEN, got {crm.get('status')}"
        return True, ""
    if name == "crm_failure_visible":
        failed_crm_calls = [call for call in tool_calls if call["tool_name"] == "write_crm_note" and call.get("success") is False]
        if not failed_crm_calls:
            return False, "expected failed write_crm_note call"
        crm = _crm_state(output)
        failed_trajectory = _trajectory_has_failed_tool(output, "write_crm_note")
        if crm:
            if crm.get("status") not in ["FAILED", "PENDING"] and not failed_trajectory:
                return False, f"CRM failure not reflected in state or trajectory: crm={crm}"
        elif not failed_trajectory:
            return False, "CRM failure not visible in state or trajectory"
        false_success_phrases = [
            "已记录完成",
            "已经记录完成",
            "已同步到 CRM",
            "已经同步到 CRM",
            "已经把预约和需求同步到 CRM",
            "已经把需求同步到 CRM",
            "并在 CRM 中记录",
        ]
        if any(phrase in message for phrase in false_success_phrases):
            return False, "assistant_message claims completed CRM write after CRM failure"
        if _book_demo_success(output) and not _has_pending_outbox(output):
            return False, "booking succeeded after CRM failure but no pending outbox exists"
        return True, ""
    if name == "handoff_crm_failure_visible":
        failed_crm_calls = [call for call in tool_calls if call["tool_name"] == "write_crm_note" and call.get("success") is False]
        successful_handoffs = [call for call in tool_calls if call["tool_name"] == "handoff_to_human" and call.get("success") is True]
        if not failed_crm_calls:
            return False, "expected failed write_crm_note call"
        if not successful_handoffs:
            return False, "expected successful handoff_to_human call"
        crm = _crm_state(output)
        if crm.get("status") != "PENDING":
            return False, f"expected CRM PENDING after handoff CRM failure, got {crm}"
        risk_flags = debug_state.get("risk_flags", [])
        if "HANDOFF_WITH_CRM_PENDING" not in risk_flags:
            return False, f"HANDOFF_WITH_CRM_PENDING missing from risk_flags: {risk_flags}"
        pending_handoff_outbox = [
            event for event in _outbox(output)
            if event.get("event_type") == "WRITE_CRM_AFTER_HANDOFF" and event.get("status") == "pending"
        ]
        if not pending_handoff_outbox:
            return False, f"expected pending WRITE_CRM_AFTER_HANDOFF outbox, got {_outbox(output)}"
        if not _trajectory_has_failed_tool(output, "write_crm_note"):
            return False, "failed CRM write not visible in trajectory"
        if not _trajectory_has_successful_tool(output, "handoff_to_human"):
            return False, "successful handoff not visible in trajectory"
        return True, ""
    if name == "booking_failed_state_used":
        booking = debug_state.get("booking", {})
        failed_books = [call for call in tool_calls if call["tool_name"] == "book_demo" and call.get("success") is False]
        if not failed_books:
            return False, "expected failed book_demo call"
        if booking.get("status") != "BOOKING_FAILED":
            return False, f"expected BOOKING_FAILED, got {booking.get('status')}"
        if booking.get("book_demo_success"):
            return False, "book_demo_success should be false after booking failure"
        if booking.get("selected_slot_id") is not None or booking.get("selected_slot_confirmed"):
            return False, f"failed slot should be cleared, got {booking}"
        return True, ""
    if name == "kb_multiple_docs_and_evidence":
        kb_calls = [call for call in tool_calls if call["tool_name"] == "search_knowledge_base" and call.get("success")]
        if not kb_calls:
            return False, "expected successful search_knowledge_base call"
        data = kb_calls[-1].get("data", {})
        docs = data.get("documents", [])
        evidence_ids = data.get("evidence_ids", [])
        topics = data.get("topics", [])
        if len(docs) < 2 or len(evidence_ids) < 2:
            return False, f"expected multiple docs/evidence_ids, got data={data}"
        if "lead_follow_up" not in topics or "crm_integration" not in topics:
            return False, f"expected lead_follow_up and crm_integration topics, got {topics}"
        return True, ""
    if name == "kb_no_result_visible":
        kb_calls = [call for call in tool_calls if call["tool_name"] == "search_knowledge_base" and call.get("success")]
        if not kb_calls:
            return False, "expected successful search_knowledge_base call"
        data = kb_calls[-1].get("data", {})
        if data.get("result_status") != "no_result":
            return False, f"expected no_result, got {data}"
        if data.get("documents") or data.get("evidence_ids"):
            return False, f"expected no docs/evidence for no_result, got {data}"
        if ClaimValidator.contains_feature_claim(message):
            return False, "assistant made a feature claim after KB no_result"
        return True, ""
    if name == "handoff_security_queue_sla":
        handoff_calls = [call for call in tool_calls if call["tool_name"] == "handoff_to_human" and call.get("success")]
        if not handoff_calls:
            return False, "expected successful handoff_to_human call"
        data = handoff_calls[-1].get("data", {})
        expected = {"queue": "security_sales_queue", "urgency": "high", "sla": "4h"}
        mismatches = {key: data.get(key) for key, value in expected.items() if data.get(key) != value}
        if mismatches:
            return False, f"handoff routing mismatch: expected {expected}, got {data}"
        return True, ""
    if name == "crm_weak_summary_rejected":
        store = InMemoryStore()
        result = tools.write_crm_note_with_store(store, "L_TEST", "销售", "medium", "继续跟进")
        if result.success:
            return False, f"weak CRM summary unexpectedly succeeded: {result.model_dump(mode='json')}"
        if store.crm_notes:
            return False, f"weak CRM summary wrote notes: {store.crm_notes}"
        if result.error not in ["summary_missing_context", "summary_low_information"]:
            return False, f"unexpected weak summary error: {result.error}"
        return True, ""
    if name == "tool_store_isolation":
        store_a = InMemoryStore()
        store_b = InMemoryStore()
        first = tools.book_demo_with_store(store_a, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")
        second = tools.book_demo_with_store(store_b, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")
        if not first.success or not second.success:
            return False, f"expected same booking input to work in isolated stores: first={first}, second={second}"
        if len(store_a.bookings) != 1 or len(store_b.bookings) != 1:
            return False, f"unexpected booking counts: a={store_a.bookings}, b={store_b.bookings}"
        if store_a.booked_keys is store_b.booked_keys:
            return False, "booked_keys shared object across stores"
        return True, ""
    if name == "mixed_feature_security_handled":
        if not any(call["tool_name"] == "handoff_to_human" for call in tool_calls):
            return False, "security/SOC2 part was not handed off"
        if not any(call["tool_name"] == "search_knowledge_base" for call in tool_calls):
            return False, "feature part was not grounded through KB search"
        if "跟进提醒" in message or _has_grounded_claim(output, "feature"):
            return True, ""
        return False, f"feature facet was not handled in mixed intent response: {message}"
    if name == "demo_purpose_not_set_without_demo_intent":
        info = debug_state.get("collected_info", {})
        if info.get("demo_purpose") is not None:
            return False, f"demo_purpose should be null, got {info.get('demo_purpose')}"
        if info.get("interest_area"):
            return True, ""
        return False, "expected interest_area to preserve non-demo product interest"
    if name == "timezone_extracted":
        info = debug_state.get("collected_info", {})
        if info.get("timezone") == "Asia/Tokyo":
            return True, ""
        return False, f"expected timezone Asia/Tokyo, got {info.get('timezone')}"
    if name in ["sales_team_size_extracted", "sales_team_size_is_8"]:
        info = debug_state.get("collected_info", {})
        if info.get("sales_team_size") == "8":
            return True, ""
        return False, f"expected sales_team_size 8, got {info.get('sales_team_size')}"
    if name == "sales_team_size_is_25":
        info = debug_state.get("collected_info", {})
        if info.get("sales_team_size") == "25":
            return True, ""
        return False, f"expected sales_team_size 25, got {info.get('sales_team_size')}"
    if name == "company_size_is_200":
        info = debug_state.get("collected_info", {})
        if info.get("company_size") == "200":
            return True, ""
        return False, f"expected company_size 200, got {info.get('company_size')}"
    if name == "company_size_not_extracted":
        info = debug_state.get("collected_info", {})
        if info.get("company_size") is None:
            return True, ""
        return False, f"expected company_size to remain unknown, got {info.get('company_size')}"
    if name == "industry_present":
        info = debug_state.get("collected_info", {})
        if info.get("industry") in ["SaaS", "消费品", "汽车"]:
            return True, ""
        return False, f"expected SaaS/消费品/汽车 industry extraction, got {info.get('industry')}"
    if name == "generic_missing_info_not_forced":
        missing = state.get("missing_info", [])
        if missing:
            return False, f"generic inquiry should not force qualification fields, got {missing}"
        if any(token in message for token in ["预算", "决策人", "上线时间"]):
            return False, f"generic clarification pushed qualification fields: {message}"
        return True, ""
    if name == "no_forced_budget_decision_timeline_question":
        if any(token in message for token in ["预算", "决策人", "上线时间"]):
            return False, f"generic clarification forced budget/decision/timeline: {message}"
        missing = state.get("missing_info", [])
        forced = {"budget", "decision_maker", "go_live_time"}.intersection(set(missing))
        if forced:
            return False, f"generic inquiry forced missing fields: {sorted(forced)}"
        return True, ""
    if name == "next_action_does_not_ask_known_fields":
        next_action = state.get("next_action", "")
        missing = state.get("missing_info", [])
        if missing:
            return False, f"expected no missing qualification fields, got {missing}"
        forbidden = ["预算", "决策人", "上线时间"]
        if any(token in next_action for token in forbidden):
            return False, f"next_action asks for known fields: {next_action}"
        return True, ""
    if name == "current_user_info_overrides_old_context":
        info = debug_state.get("collected_info", {})
        risk_flags = debug_state.get("risk_flags", [])
        conflicts = debug_state.get("context_conflicts", [])
        if info.get("industry") != "零售":
            return False, f"expected industry 零售, got {info.get('industry')}"
        if "CONTEXT_CONFLICT" not in risk_flags:
            return False, f"expected CONTEXT_CONFLICT in risk_flags, got {risk_flags}"
        if not any(conflict.get("field") == "industry" for conflict in conflicts):
            return False, f"expected industry conflict, got {conflicts}"
        return True, ""
    if name == "trigger_event_conversation_supported":
        trigger_store = InMemoryStore()
        trigger = TriggerEvent(
            event_type="conversation",
            lead_id="L123",
            payload={
                "conversation": [
                    {"role": "user", "content": "我们公司最近在看 AI 销售工具。"},
                ]
            },
        )
        response = SalesAgentHarness(
            planner_mode="mock_llm",
            store=trigger_store,
            reset_tools=True,
            session_id="trigger_eval:L123",
        ).handle_trigger(trigger)
        payload = response.model_dump(mode="json")
        if payload.get("assistant_message") and payload.get("run_id"):
            return True, ""
        return False, f"TriggerEvent did not return an AgentResponse-shaped payload: {payload}"
    if name == "validators_unit_tests_exist":
        test_path = Path(__file__).resolve().parents[1] / "tests" / "test_validators.py"
        if not test_path.exists():
            return False, f"{test_path} does not exist"
        content = test_path.read_text(encoding="utf-8")
        required_markers = [
            "PreconditionValidator",
            "PostconditionValidator",
            "MISSING_EMAIL",
            "MISSING_TIMEZONE",
            "MISSING_DEMO_PURPOSE",
            "MISSING_EXPLICIT_SLOT_CONFIRMATION",
            "SELECTED_SLOT_MISMATCH",
            "TIMEZONE_MISMATCH",
            "UNSUPPORTED_PRICE_CLAIM",
            "UNSUPPORTED_METRIC_CLAIM",
            "UNSUPPORTED_CUSTOMER_CASE",
            "UNSUPPORTED_FEATURE_CLAIM",
            "CLAIMED_BOOKED_WITHOUT_BOOK_DEMO",
            "CRM_DEMO_BOOKED_WITHOUT_BOOK_DEMO",
            "TOOL_NOT_ALLOWED_IN_CURRENT_STATE",
        ]
        missing = [marker for marker in required_markers if marker not in content]
        if missing:
            return False, f"validator unit tests missing markers: {missing}"
        return True, ""
    return False, f"unknown assertion {name}"


def _metrics(cases: List[EvalCase], results: List[EvalCaseResult]) -> Dict[str, float]:
    total = len(results) or 1
    metrics = {
        "json_parseable_rate": _assertion_rate(cases, results, "json_parseable"),
        "policy_pass_rate": sum(1 for result in results if result.passed) / total,
        "tool_correctness_rate": _tool_correctness_rate(cases, results),
        "no_hallucinated_price_rate": _assertion_rate(cases, results, "no_unsupported_price"),
        "handoff_correctness_rate": _assertion_rate(cases, results, "handoff_called"),
        "crm_completeness_rate": _assertion_rate(cases, results, "crm_note_complete"),
        "demo_booking_correctness_rate": _demo_rate(cases, results),
    }
    return {key: round(value, 3) for key, value in metrics.items()}


def _assert_crm_note_complete(output: Dict[str, Any]) -> Tuple[bool, str]:
    tool_calls = output.get("tool_calls", [])
    crm_calls = [call for call in tool_calls if call["tool_name"] == "write_crm_note"]
    if not crm_calls:
        return False, "write_crm_note not called"
    successful_crm_calls = [call for call in crm_calls if call.get("success") is True]
    if not successful_crm_calls:
        return False, "write_crm_note was called but no successful CRM write"

    for call in successful_crm_calls:
        args = call.get("arguments", {})
        summary = args.get("summary", "")
        if not summary:
            return False, "CRM summary missing"
        if not args.get("next_action"):
            return False, "CRM next_action missing"
        if args.get("qualification_level") not in ["high", "medium", "low", "unknown"]:
            return False, f"invalid CRM qualification_level: {args.get('qualification_level')}"

        if _is_booking_crm_note(output, args):
            ok, reason = _booking_crm_note_complete(summary)
        elif _is_handoff_crm_note(output, args):
            ok, reason = _handoff_crm_note_complete(summary, output.get("debug_state", {}).get("risk_flags", []))
        else:
            ok, reason = _qualification_crm_note_complete(summary)
        if not ok:
            return False, reason

    notes = output.get("debug_store", {}).get("crm_notes", [])
    valid_stages = {"discovery", "qualification", "demo_pending", "demo_booked", "handoff_required", "crm_sync_pending", "unknown"}
    if notes:
        if len(notes) < len(successful_crm_calls):
            return False, f"successful CRM writes missing internal records: calls={len(successful_crm_calls)}, notes={len(notes)}"
        for note in notes[-len(successful_crm_calls) :]:
            if note.get("qualification_level") not in ["high", "medium", "low", "unknown"]:
                return False, f"invalid internal CRM qualification_level: {note}"
            if not note.get("next_action"):
                return False, f"internal CRM next_action missing: {note}"
            if not note.get("pain_point"):
                return False, f"internal CRM pain_point missing: {note}"
            if note.get("stage") not in valid_stages:
                return False, f"internal CRM stage invalid: {note}"
    return True, ""


def _is_booking_crm_note(output: Dict[str, Any], args: Dict[str, Any]) -> bool:
    return args.get("next_action") == "Demo booked" or _book_demo_success(output)


def _is_handoff_crm_note(output: Dict[str, Any], args: Dict[str, Any]) -> bool:
    tool_names = [call["tool_name"] for call in output.get("tool_calls", [])]
    risk_flags = output.get("debug_state", {}).get("risk_flags", [])
    return (
        "handoff_to_human" in tool_names
        or "Human handoff" in args.get("next_action", "")
        or any(flag in risk_flags for flag in HANDOFF_RISK_FLAGS)
    )


def _qualification_crm_note_complete(summary: str) -> Tuple[bool, str]:
    fact_count = 0
    if re.search(r"\d+\s*人(?:公司|企业|集团|团队)?", summary):
        fact_count += 1
    if any(industry in summary for industry in ["制造业", "互联网", "软件", "SaaS", "教育", "金融", "医疗", "零售", "消费品", "汽车"]):
        fact_count += 1
    if any(token in summary for token in ["销售团队", "销售人员", "漏跟进", "跟进不及时", "线索响应慢", "客户流失", "CRM"]):
        fact_count += 1
    if any(token in summary for token in ["预算", "未定", "range_known", "known"]):
        fact_count += 1
    if fact_count < 2:
        return False, f"qualification CRM summary has fewer than two business facts: {summary}"
    return True, ""


def _handoff_crm_note_complete(summary: str, risk_flags: List[str]) -> Tuple[bool, str]:
    if handoff_summary_covers_flags(summary, risk_flags):
        return True, ""
    return False, f"handoff CRM summary lacks high-risk context: {summary}"


def _booking_crm_note_complete(summary: str) -> Tuple[bool, str]:
    fact_count = 0
    if any(token in summary for token in ["Demo", "演示"]):
        fact_count += 1
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", summary) or "邮箱" in summary:
        fact_count += 1
    if any(token in summary for token in ["时间", "slot", "周一", "周二", "周三", "周四", "周五", "周六", "周日"]):
        fact_count += 1
    if any(token in summary for token in ["目的", "需求", "CRM", "对接", "集成", "产品演示"]):
        fact_count += 1
    if fact_count < 3:
        return False, f"booking CRM summary lacks booking context: {summary}"
    return True, ""


def _assertion_rate(cases: List[EvalCase], results: List[EvalCaseResult], assertion: str) -> float:
    relevant = [(case, result) for case, result in zip(cases, results) if assertion in case.assertions]
    if not relevant:
        return 1.0
    passed = 0
    for _case, result in relevant:
        if not any(failure.startswith(f"{assertion}:") for failure in result.failures):
            passed += 1
    return passed / len(relevant)


def _response_payload(response) -> Dict[str, Any]:
    payload = response.model_dump(mode="json", warnings=False)
    for optional_key in ["debug_state", "trajectory"]:
        if payload.get(optional_key) is None:
            payload.pop(optional_key, None)
    return payload


def _attach_store_debug(output: Dict[str, Any], store: InMemoryStore) -> None:
    output["debug_store"] = {
        "crm_notes": list(store.crm_notes),
        "handoff_log": list(store.handoff_log),
        "bookings": dict(store.bookings),
        "booked_keys": dict(store.booked_keys),
        "outbox": [event.model_dump(mode="json") for event in store.outbox],
        "call_counts": dict(store.call_counts),
        "sessions": list(store.sessions.keys()),
        "lead_context_cache": dict(store.lead_context_cache),
    }


def _attach_idempotency_probe(output: Dict[str, Any], store: InMemoryStore, lead_id: str) -> None:
    if not store.booked_keys:
        output["idempotency_probe"] = {"error": "no booked_keys available"}
        return
    existing = next(iter(store.booked_keys.values()))
    before_count = len(store.bookings)
    result = tools.book_demo(
        store,
        lead_id,
        existing.get("slot_id"),
        existing.get("attendee_email"),
        "idempotency regression probe",
    )
    after_count = len(store.bookings)
    output["idempotency_probe"] = {
        "booking_count_before": before_count,
        "booking_count_after": after_count,
        "result": result.model_dump(mode="json"),
    }
    _attach_store_debug(output, store)


def _evidence_policy(output: Dict[str, Any]) -> Dict[str, Any]:
    debug_state = output.get("debug_state", {})
    if isinstance(debug_state, dict) and isinstance(debug_state.get("evidence_policy"), dict):
        return debug_state["evidence_policy"]
    state = output.get("state", {})
    if isinstance(state, dict) and isinstance(state.get("evidence_policy"), dict):
        return state["evidence_policy"]
    return {}


def _grounded_claims(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    debug_state = output.get("debug_state", {})
    claims = None
    if isinstance(debug_state, dict):
        claims = debug_state.get("last_claims") or debug_state.get("grounded_claims")
    if isinstance(claims, list):
        return [claim for claim in claims if isinstance(claim, dict)]
    return []


def _has_grounded_claim(output: Dict[str, Any], claim_type: str) -> bool:
    policy = _evidence_policy(output)
    supported_facets = policy.get("supported_facets", {}) if isinstance(policy, dict) else {}
    for claim in _grounded_claims(output):
        if claim.get("claim_type") != claim_type:
            continue
        facet = claim.get("facet")
        evidence_ids = set(claim.get("evidence_ids") or [])
        facet_evidence_ids = set(supported_facets.get(facet, [])) if isinstance(supported_facets, dict) else set()
        if evidence_ids and evidence_ids.intersection(facet_evidence_ids):
            return True
    return False


def _crm_state(output: Dict[str, Any]) -> Dict[str, Any]:
    debug_state = output.get("debug_state", {})
    if isinstance(debug_state, dict) and isinstance(debug_state.get("crm"), dict):
        return debug_state["crm"]
    state = output.get("state", {})
    if isinstance(state, dict) and isinstance(state.get("crm"), dict):
        return state["crm"]
    return {}


def _outbox(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    debug_state = output.get("debug_state", {})
    if isinstance(debug_state, dict) and isinstance(debug_state.get("outbox"), list):
        return debug_state["outbox"]
    return []


def _has_pending_outbox(output: Dict[str, Any]) -> bool:
    return any(event.get("status") == "pending" for event in _outbox(output))


def _trajectory_has_failed_tool(output: Dict[str, Any], tool_name: str) -> bool:
    trajectory = output.get("trajectory", [])
    return any(
        event.get("step") == "tool_call"
        and event.get("tool_name") == tool_name
        and event.get("status") == "failed"
        for event in trajectory
        if isinstance(event, dict)
    )


def _trajectory_has_successful_tool(output: Dict[str, Any], tool_name: str) -> bool:
    trajectory = output.get("trajectory", [])
    return any(
        event.get("step") == "tool_call"
        and event.get("tool_name") == tool_name
        and event.get("status") == "success"
        for event in trajectory
        if isinstance(event, dict)
    )


def _book_demo_success(output: Dict[str, Any]) -> bool:
    debug_state = output.get("debug_state", {})
    if isinstance(debug_state, dict):
        booking = debug_state.get("booking", {})
        if isinstance(booking, dict) and booking.get("book_demo_success") is True:
            return True
    return any(call["tool_name"] == "book_demo" and call.get("success") is True for call in output.get("tool_calls", []))


def _tool_correctness_rate(cases: List[EvalCase], results: List[EvalCaseResult]) -> float:
    if not cases:
        return 1.0
    passed = 0
    for result in results:
        if not any(failure.startswith("Expected tool call") or failure.startswith("Forbidden tool call") for failure in result.failures):
            passed += 1
    return passed / len(cases)


def _demo_rate(cases: List[EvalCase], results: List[EvalCaseResult]) -> float:
    demo_assertions = {
        "asks_for_missing_booking_fields",
        "no_false_booking_claim",
        "calendar_timezone_is_Asia_Singapore",
        "asks_timezone_clarification",
        "offers_slots_not_booked",
        "book_demo_before_booking_claim",
        "crm_after_book_demo",
        "crm_not_demo_booked",
        "has_pending_outbox",
        "outbox_pending_visible",
        "no_duplicate_book_demo",
        "idempotency_replay_visible",
        "no_duplicate_calendar_event",
    }
    relevant = []
    for case, result in zip(cases, results):
        if demo_assertions.intersection(set(case.assertions)):
            relevant.append(result)
    if not relevant:
        return 1.0
    passed = 0
    for result in relevant:
        if not any(any(failure.startswith(f"{assertion}:") for assertion in demo_assertions) for failure in result.failures):
            passed += 1
    return passed / len(relevant)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sales Agent Harness eval cases.")
    parser.add_argument("--cases", default=str(Path(__file__).with_name("eval_cases.yaml")))
    parser.add_argument("--agent-version", default="harness_v2")
    parser.add_argument("--planner", choices=["auto", "mock_llm", "rule"], default=None)
    parser.add_argument("--allow-failures", action="store_true", help="Print failures but exit 0.")
    args = parser.parse_args()
    report = run_eval(args.cases, args.agent_version, args.planner)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if report.failed and not args.allow_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
