from __future__ import annotations

from sales_agent_harness.eval_runner import _assert_crm_note_complete
from sales_agent_harness.harness import SalesAgentHarness
from sales_agent_harness.models import AgentState, Claim, ConversationRequest, ConversationTurn, ToolCall, ToolResult
from sales_agent_harness.normalizer import extract_message
from sales_agent_harness.policy import PolicyRouter
from sales_agent_harness.state import BookingStateMachine, MemoryManager
from sales_agent_harness.store import InMemoryStore
from sales_agent_harness.validators import ClaimValidator, PostconditionValidator, PreconditionValidator


def _booking_ready_state() -> AgentState:
    state = AgentState(lead_id="L_TEST")
    state.allowed_actions = ["book_demo"]
    state.collected_info.email = "alex@example.com"
    state.collected_info.timezone = "Asia/Singapore"
    state.collected_info.demo_purpose = "了解 CRM 对接"
    state.booking.offered_slots = [{"slot_id": "sg_slot_1", "display": "周二 10:00", "timezone": "Asia/Singapore"}]
    state.booking.selected_slot_id = "sg_slot_1"
    state.booking.selected_slot_confirmed = True
    return state


def test_book_demo_precondition_requires_booking_fields() -> None:
    state = AgentState(lead_id="L_TEST")
    state.allowed_actions = ["book_demo"]
    call = ToolCall(tool_name="book_demo", arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1"})

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "MISSING_EMAIL" in result.errors
    assert "MISSING_TIMEZONE" in result.errors
    assert "MISSING_DEMO_PURPOSE" in result.errors
    assert "MISSING_EXPLICIT_SLOT_CONFIRMATION" in result.errors


def test_book_demo_rejects_selected_slot_mismatch() -> None:
    state = _booking_ready_state()
    state.booking.selected_slot_id = "sg_slot_2"
    state.booking.offered_slots.append({"slot_id": "sg_slot_2", "display": "周三 15:00", "timezone": "Asia/Singapore"})
    call = ToolCall(tool_name="book_demo", arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1"})

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "SELECTED_SLOT_MISMATCH" in result.errors
    assert "UNKNOWN_SLOT_ID" not in result.errors


def test_book_demo_unknown_slot_does_not_also_report_mismatch() -> None:
    state = _booking_ready_state()
    call = ToolCall(tool_name="book_demo", arguments={"lead_id": "L_TEST", "slot_id": "missing_slot"})

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "UNKNOWN_SLOT_ID" in result.errors
    assert "SELECTED_SLOT_MISMATCH" not in result.errors


def test_check_calendar_timezone_mismatch_guardrail() -> None:
    state = AgentState(lead_id="L_TEST")
    state.allowed_actions = ["check_calendar"]
    state.collected_info.timezone = "Asia/Singapore"
    call = ToolCall(tool_name="check_calendar", arguments={"timezone": "Asia/Shanghai", "duration_minutes": 30})

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "TIMEZONE_MISMATCH" in result.errors


def test_time_arrangement_is_not_product_question() -> None:
    state = AgentState(lead_id="L_TEST")
    conversation = [ConversationTurn(role="user", content="我们能不能下周聊？")]

    intent = PolicyRouter.detect_intent(conversation, state)

    assert intent != "product_question"


def test_feature_question_still_routes_product_when_object_and_intent_exist() -> None:
    state = AgentState(lead_id="L_TEST")
    conversation = [ConversationTurn(role="user", content="你们能不能做线索自动分层、销售跟进提醒，还能对接 CRM？")]

    intent = PolicyRouter.detect_intent(conversation, state)

    assert intent == "product_question"


def test_crm_integration_question_routes_product() -> None:
    state = AgentState(lead_id="L_TEST")
    conversation = [ConversationTurn(role="user", content="你们能不能对接 CRM？")]

    intent = PolicyRouter.detect_intent(conversation, state)

    assert intent == "product_question"


def test_industry_extraction_uses_current_non_negated_industry() -> None:
    first = extract_message("我们现在不是制造业了，是零售业务团队。")
    second = extract_message("我们不再是制造业，转去做零售。")
    third = extract_message("以前是制造业，现在主要做零售。")

    assert first.industry == "零售"
    assert second.industry == "零售"
    assert third.industry == "零售"
    assert "INDUSTRY_UPDATED" in first.risk_flags
    assert "INDUSTRY_UPDATED" in second.risk_flags


def test_company_size_does_not_guess_from_unrelated_people_counts() -> None:
    extraction = extract_message("我们公司过去 5 人创业，现在 25 人销售团队，10 人研发。")

    assert extraction.sales_team_size == "25"
    assert extraction.company_size is None


def test_company_size_uses_company_context_when_present() -> None:
    extraction = extract_message("我们是 200 人公司，但销售只有 8 人，经常漏跟进客户。")

    assert extraction.company_size == "200"
    assert extraction.sales_team_size == "8"


def test_crm_note_complete_rejects_token_only_summary() -> None:
    output = {
        "tool_calls": [
            {
                "tool_name": "write_crm_note",
                "success": True,
                "arguments": {
                    "summary": "销售",
                    "qualification_level": "medium",
                    "next_action": "继续跟进",
                },
            }
        ],
        "debug_state": {"crm": {"status": "WRITTEN"}},
    }

    ok, reason = _assert_crm_note_complete(output)

    assert not ok
    assert "fewer than two business facts" in reason


def test_crm_note_complete_accepts_structured_qualification_summary() -> None:
    output = {
        "tool_calls": [
            {
                "tool_name": "write_crm_note",
                "success": True,
                "arguments": {
                    "summary": "200人公司，销售团队8人，痛点是销售线索漏跟进，预算未定。",
                    "qualification_level": "medium",
                    "next_action": "继续确认决策人和上线时间",
                },
            }
        ],
        "debug_state": {"crm": {"status": "WRITTEN"}},
    }

    ok, reason = _assert_crm_note_complete(output)

    assert ok, reason


def test_crm_note_complete_requires_handoff_context() -> None:
    output = {
        "tool_calls": [
            {
                "tool_name": "write_crm_note",
                "success": True,
                "arguments": {
                    "summary": "客户需要人工跟进。",
                    "qualification_level": "unknown",
                    "next_action": "Human handoff for legal/security review",
                },
            },
            {"tool_name": "handoff_to_human", "success": True, "arguments": {"reason": "security"}},
        ],
        "debug_state": {"crm": {"status": "WRITTEN"}, "risk_flags": ["LEGAL_OR_SECURITY"]},
    }

    ok, reason = _assert_crm_note_complete(output)

    assert not ok
    assert "lacks high-risk context" in reason


def test_crm_note_complete_accepts_booking_summary_with_context() -> None:
    output = {
        "tool_calls": [
            {"tool_name": "book_demo", "success": True, "arguments": {}},
            {
                "tool_name": "write_crm_note",
                "success": True,
                "arguments": {
                    "summary": "Demo需求是了解 CRM 对接，参会邮箱 alex@example.com，Demo时间已选择：sg_slot_1。",
                    "qualification_level": "medium",
                    "next_action": "Demo booked",
                },
            },
        ],
        "debug_state": {"booking": {"book_demo_success": True}, "crm": {"status": "WRITTEN"}},
    }

    ok, reason = _assert_crm_note_complete(output)

    assert ok, reason


def test_no_public_price_blocks_amount_claim() -> None:
    state = AgentState(lead_id="L_TEST")
    state.evidence_policy.price_policy = "no_public_price"

    result = PostconditionValidator.validate("标准版一年 5 万。", state, [])

    assert not result.ok
    assert "UNSUPPORTED_PRICE_CLAIM" in result.errors


def test_no_guaranteed_metric_blocks_guarantee_claim() -> None:
    state = AgentState(lead_id="L_TEST")
    state.evidence_policy.metric_policy = "no_guaranteed_metric"

    result = PostconditionValidator.validate("我们保证提升 50%。", state, [])

    assert not result.ok
    assert "UNSUPPORTED_METRIC_CLAIM" in result.errors


def test_no_public_customer_name_blocks_customer_case_claim() -> None:
    state = AgentState(lead_id="L_TEST")
    state.evidence_policy.customer_case_policy = "no_public_customer_name"

    result = PostconditionValidator.validate("我们有多家 500 强客户在用。", state, [])

    assert not result.ok
    assert "UNSUPPORTED_CUSTOMER_CASE" in result.errors


def test_feature_policy_unknown_blocks_feature_claim() -> None:
    state = AgentState(lead_id="L_TEST")
    state.evidence_policy.feature_policy = "needs_confirmation"

    result = PostconditionValidator.validate("产品支持 CRM 集成。", state, [])

    assert not result.ok
    assert "UNSUPPORTED_FEATURE_CLAIM" in result.errors


def test_price_obfuscation_blocks_without_grounded_price_claim() -> None:
    state = AgentState(lead_id="L_TEST")

    for message in ["一年 5W。", "一年 50000 RMB。", "一年五十 K。"]:
        result = PostconditionValidator.validate(message, state, [])

        assert not result.ok
        assert "UNSUPPORTED_PRICE_CLAIM" in result.errors


def test_customer_case_variant_names_blocked_without_grounding() -> None:
    state = AgentState(lead_id="L_TEST")

    result = PostconditionValidator.validate("字节、美团、理想汽车都在用。", state, [])

    assert not result.ok
    assert "UNSUPPORTED_CUSTOMER_CASE" in result.errors


def test_feature_claim_requires_matching_facet_evidence() -> None:
    state = AgentState(lead_id="L_TEST")
    state.evidence_policy.supported_facets = {"lead_scoring": ["kb_feature_001"]}
    tool_results = [
        ToolResult(
            tool_name="search_knowledge_base",
            success=True,
            data={
                "documents": [
                    {
                        "id": "kb_feature_001",
                        "topic": "lead_follow_up",
                        "title": "线索自动分层",
                        "content": "产品支持对线索进行自动分层。",
                        "supported_facets": ["lead_scoring"],
                        "unsupported_facets": [],
                        "policy": {"feature_policy": "feature_supported"},
                        "visibility": "public",
                    }
                ],
                "evidence_ids": ["kb_feature_001"],
            },
        )
    ]
    state.grounded_claims = [
        Claim(
            text="产品支持线索自动分层。",
            claim_type="feature",
            facet="lead_scoring",
            evidence_ids=["kb_feature_001"],
            evidence_span="产品支持对线索进行自动分层。",
            confidence="high",
        )
    ]

    mismatch = PostconditionValidator.validate("产品支持 CRM 集成。", state, tool_results)
    matched = PostconditionValidator.validate("产品支持线索自动分层。", state, tool_results)

    assert not mismatch.ok
    assert "UNSUPPORTED_FEATURE_CLAIM" in mismatch.errors
    assert matched.ok


def test_booking_success_claim_requires_book_demo_success() -> None:
    state = AgentState(lead_id="L_TEST")

    result = PostconditionValidator.validate("Demo 已预约。", state, [])

    assert not result.ok
    assert "CLAIMED_BOOKED_WITHOUT_BOOK_DEMO" in result.errors


def test_negated_booking_phrase_is_not_success_claim() -> None:
    assert not ClaimValidator.contains_booking_success_claim("这个时间不可用，我不会把它记为已预约。")


def test_crm_demo_booked_requires_successful_booking() -> None:
    state = AgentState(lead_id="L_TEST")
    state.allowed_actions = ["write_crm_note"]
    call = ToolCall(
        tool_name="write_crm_note",
        arguments={
            "lead_id": "L_TEST",
            "summary": "Demo需求是了解 CRM 对接",
            "qualification_level": "medium",
            "next_action": "Demo booked",
        },
    )

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "CRM_DEMO_BOOKED_WITHOUT_BOOK_DEMO" in result.errors


def test_tool_not_allowed_blocks_book_demo() -> None:
    state = _booking_ready_state()
    state.allowed_actions = ["ask_clarification"]
    call = ToolCall(tool_name="book_demo", arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1"})

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "TOOL_NOT_ALLOWED_IN_CURRENT_STATE" in result.errors


def test_rejected_or_unconfirmed_slot_blocks_book_demo() -> None:
    state = _booking_ready_state()
    state.booking.selected_slot_confirmed = False
    state.booking.rejected_slot_ids = ["sg_slot_1"]
    call = ToolCall(tool_name="book_demo", arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1"})

    result = PreconditionValidator.validate(call, state)

    assert not result.ok
    assert "MISSING_EXPLICIT_SLOT_CONFIRMATION" in result.errors


def test_book_demo_failure_enters_recoverable_failed_state() -> None:
    state = _booking_ready_state()
    result = ToolResult(
        tool_name="book_demo",
        arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1", "attendee_email": "alex@example.com"},
        success=False,
        error="slot_unavailable",
    )

    BookingStateMachine.apply_tool_result(state, result)

    assert state.booking.status == "BOOKING_FAILED"
    assert state.booking.selected_slot_id is None
    assert not state.booking.selected_slot_confirmed
    assert "sg_slot_1" in state.booking.rejected_slot_ids


def test_booking_failed_can_reselect_remaining_slot() -> None:
    state = _booking_ready_state()
    state.booking.offered_slots.append({"slot_id": "sg_slot_2", "display": "周三 15:00", "timezone": "Asia/Singapore"})
    result = ToolResult(
        tool_name="book_demo",
        arguments={"lead_id": "L_TEST", "slot_id": "sg_slot_1", "attendee_email": "alex@example.com"},
        success=False,
        error="slot_unavailable",
    )
    BookingStateMachine.apply_tool_result(state, result)

    MemoryManager.rebuild_state_from_conversation(
        state,
        [ConversationTurn(role="user", content="那就周三 15 点可以。")],
    )

    assert state.booking.status == "SLOT_SELECTED"
    assert state.booking.selected_slot_id == "sg_slot_2"
    assert state.booking.selected_slot_confirmed


def test_handoff_success_with_crm_failure_creates_pending_compensation() -> None:
    store = InMemoryStore()
    harness = SalesAgentHarness(
        planner_mode="mock_llm",
        store=store,
        reset_tools=True,
        mock_overrides={"write_crm_note": {"force_failure": True, "error": "crm_timeout"}},
        debug=True,
        session_id="handoff_compensation:L_TEST",
    )
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我们法务想看 DPA 和 SOC2 材料。")],
    )

    response = harness.handle_conversation(request)

    assert any(result.tool_name == "handoff_to_human" and result.success for result in response.tool_calls)
    assert response.debug_state is not None
    assert response.debug_state.crm.status == "PENDING"
    assert "HANDOFF_WITH_CRM_PENDING" in response.debug_state.risk_flags
    assert any(event.event_type == "WRITE_CRM_AFTER_HANDOFF" and event.status == "pending" for event in response.debug_state.outbox)


def test_lead_context_cache_hit_is_visible_in_trajectory() -> None:
    store = InMemoryStore()
    first = SalesAgentHarness(
        planner_mode="mock_llm",
        store=store,
        reset_tools=True,
        debug=True,
        include_trajectory=True,
        session_id="lead_cache:first",
    )
    request = ConversationRequest(
        lead_id="L123",
        conversation=[ConversationTurn(role="user", content="我们公司最近在看 AI 销售工具。")],
    )
    first.handle_conversation(request)
    second = SalesAgentHarness(
        planner_mode="mock_llm",
        store=store,
        reset_tools=False,
        debug=True,
        include_trajectory=True,
        session_id="lead_cache:second",
    )

    response = second.handle_conversation(request)

    lead_context_result = next(result for result in response.tool_calls if result.tool_name == "get_lead_context")
    assert lead_context_result.data["cache_hit"] is True
    assert response.trajectory is not None
    assert any(
        event.tool_name == "get_lead_context"
        and event.data.get("result", {}).get("cache_hit") is True
        for event in response.trajectory
    )


def test_evidence_policy_uses_metadata_before_content_substrings() -> None:
    state = AgentState(lead_id="L_TEST")
    result = ToolResult(
        tool_name="search_knowledge_base",
        arguments={"query": "你们多少钱？"},
        success=True,
        data={
            "topic": "pricing",
            "documents": [
                {
                    "id": "kb_pricing_reworded",
                    "topic": "pricing",
                    "content": "公开资料暂时不会给固定报价。",
                    "policy": {"price_policy": "no_public_price"},
                }
            ],
            "evidence_ids": ["kb_pricing_reworded"],
        },
    )

    BookingStateMachine.apply_tool_result(state, result)
    validation = PostconditionValidator.validate("标准版一年 5 万。", state, [])

    assert state.evidence_policy.price_policy == "no_public_price"
    assert not validation.ok
    assert "UNSUPPORTED_PRICE_CLAIM" in validation.errors


def test_mixed_product_and_security_intent_answers_product_and_handoffs() -> None:
    harness = SalesAgentHarness(planner_mode="rule", debug=True, session_id="mixed_intent:L_TEST")
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我先看看你们功能能不能做线索分层。另外我们 CIO 也想看 SOC2 报告。")],
    )

    response = harness.handle_conversation(request)

    assert "线索自动分层" in response.assistant_message
    assert "转给人工" in response.assistant_message or "转给人工同事" in response.assistant_message
    assert [result.tool_name for result in response.tool_calls] == [
        "get_lead_context",
        "search_knowledge_base",
        "write_crm_note",
        "handoff_to_human",
    ]
    handoff = next(result for result in response.tool_calls if result.tool_name == "handoff_to_human")
    assert "客户询问" in handoff.arguments["reason"]
    assert "Customer asks" not in handoff.arguments["reason"]
    crm = next(result for result in response.tool_calls if result.tool_name == "write_crm_note")
    assert crm.arguments["current_stage"] == "handoff_required"
    assert crm.arguments["customer_pain"]


def test_contract_handoff_writes_crm_without_internal_compensation() -> None:
    harness = SalesAgentHarness(planner_mode="rule", debug=True, session_id="contract_handoff:L_TEST")
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[
            ConversationTurn(
                role="user",
                content="我们要签合同，顺便想问你们能不能做线索跟进提醒，最好今天约 demo，我在新加坡，邮箱 alex@example.com，目的看 CRM 对接。",
            )
        ],
    )

    response = harness.handle_conversation(request)

    crm = next(result for result in response.tool_calls if result.tool_name == "write_crm_note")
    handoff = next(result for result in response.tool_calls if result.tool_name == "handoff_to_human")
    assert crm.success
    assert handoff.success
    assert crm.arguments["current_stage"] == "handoff_required"
    assert "合同" in crm.arguments["summary"] or "采购" in crm.arguments["summary"]
    assert response.debug_state is not None
    assert response.debug_state.crm.status == "WRITTEN"
    assert "HANDOFF_WITH_CRM_PENDING" not in response.debug_state.risk_flags
    assert not response.debug_state.outbox


def test_compliance_synonyms_route_to_handoff() -> None:
    state = AgentState(lead_id="L_TEST")
    conversation = [ConversationTurn(role="user", content="我们有合规问题，想看数据保护、GDPR、等保 2.0、个保法材料。")]
    MemoryManager.rebuild_state_from_conversation(state, conversation)

    intent = PolicyRouter.detect_intent(conversation, state)

    assert intent == "handoff_required"
    assert "LEGAL_OR_SECURITY" in state.risk_flags


def test_high_value_vp_budget_and_timeline_scores_high_and_writes_crm() -> None:
    harness = SalesAgentHarness(planner_mode="rule", debug=True, session_id="high_value:L_TEST")
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我们公司刚拿了 B 轮，下季度准备上线 AI 销售工具，预算 200 万，我是 VP。")],
    )

    response = harness.handle_conversation(request)

    assert response.state.qualification_level == "high"
    assert response.state.missing_info == []
    crm = next(result for result in response.tool_calls if result.tool_name == "write_crm_note")
    assert crm.success
    assert crm.arguments["qualification_level"] == "high"
    assert crm.arguments["current_stage"] == "qualification"
    assert "200万" in crm.arguments["summary"]
    assert "VP" in crm.arguments["summary"]
    assert "下季度" in crm.arguments["summary"]


def test_generic_inquiry_does_not_expose_qualification_slots_as_missing() -> None:
    harness = SalesAgentHarness(planner_mode="rule", debug=True, session_id="generic_inquiry:L_TEST")
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我们最近在做销售流程数字化转型评估。")],
    )

    response = harness.handle_conversation(request)

    assert response.state.missing_info == []
    assert "预算" not in response.assistant_message
    assert "更关注" in response.assistant_message


def test_invalid_email_is_explained_not_treated_as_absent() -> None:
    harness = SalesAgentHarness(planner_mode="rule", debug=True, session_id="invalid_email:L_TEST")
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我们想约 Demo，邮箱 alex@demo")],
    )

    response = harness.handle_conversation(request)

    assert "邮箱看起来不完整" in response.assistant_message
    assert "INVALID_EMAIL_FORMAT" in response.state.risk_flags
    assert response.debug_state is not None
    assert response.debug_state.collected_info.invalid_email == "alex@demo"


def test_handoff_tool_failure_is_visible() -> None:
    harness = SalesAgentHarness(
        planner_mode="rule",
        debug=True,
        session_id="handoff_failure:L_TEST",
        mock_overrides={"handoff_to_human": {"force_failure": True, "error": "handoff_timeout"}},
        reset_tools=True,
    )
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我们法务想看 SOC2 和 DPA 材料。")],
    )

    response = harness.handle_conversation(request)

    handoff = next(result for result in response.tool_calls if result.tool_name == "handoff_to_human")
    assert not handoff.success
    assert "HANDOFF_FAILED" in response.state.risk_flags
    assert "人工转接失败" in response.assistant_message
