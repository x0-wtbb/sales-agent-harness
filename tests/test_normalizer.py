from __future__ import annotations

from sales_agent_harness.harness import SalesAgentHarness
from sales_agent_harness.models import ConversationRequest, ConversationTurn
from sales_agent_harness.normalizer import extract_message


def test_invalid_email_is_separate_from_missing_email() -> None:
    extraction = extract_message("我在新加坡，邮箱 alex@demo，想看 CRM 对接，下周约 Demo。")

    assert extraction.email is None
    assert extraction.invalid_email == "alex@demo"
    assert extraction.timezone == "Asia/Singapore"
    assert extraction.demo_intent


def test_corrected_email_clears_invalid_email_in_session_state() -> None:
    harness = SalesAgentHarness(planner_mode="rule", debug=True, session_id="normalizer_email:L123")
    first = ConversationRequest(
        lead_id="L123",
        conversation=[ConversationTurn(role="user", content="我在新加坡，邮箱 alex@demo，想看 CRM 对接，下周约 Demo。")],
    )
    harness.handle_conversation(first)
    second = ConversationRequest(
        lead_id="L123",
        conversation=[
            ConversationTurn(role="user", content="我在新加坡，邮箱 alex@demo，想看 CRM 对接，下周约 Demo。"),
            ConversationTurn(role="assistant", content="邮箱看起来不完整，请确认完整邮箱格式。"),
            ConversationTurn(role="user", content="邮箱更正为 alex@demo.com。"),
        ],
    )

    response = harness.handle_conversation(second)

    assert response.debug_state is not None
    assert response.debug_state.collected_info.email == "alex@demo.com"
    assert response.debug_state.collected_info.invalid_email is None
    assert any(result.tool_name == "check_calendar" for result in response.tool_calls)


def test_gmt_offset_without_city_sets_risk_flag() -> None:
    extraction = extract_message("Please schedule demo, timezone is GMT+8, email alex@example.com, want CRM integration.")

    assert extraction.demo_intent
    assert extraction.email == "alex@example.com"
    assert extraction.demo_purpose == "了解 CRM 对接"
    assert extraction.timezone is None
    assert "TIMEZONE_OFFSET_NEEDS_CITY" in extraction.risk_flags


def test_b_round_vp_budget_and_q3_are_extracted() -> None:
    extraction = extract_message("我们是 B 轮 SaaS 公司，预算 200 万，我是 VP，计划 Q3 上线，痛点是线索响应慢。")

    assert extraction.industry == "SaaS"
    assert extraction.budget == "200万"
    assert extraction.budget_amount == "200万"
    assert extraction.decision_maker == "VP"
    assert extraction.go_live_time == "Q3"


def test_industry_synonyms_are_extracted() -> None:
    consumer = extract_message("我们是消费品业务，销售团队 25 人，经常漏跟进客户。")
    auto = extract_message("我们是汽车相关业务，销售团队 25 人，经常漏跟进客户。")

    assert consumer.industry == "消费品"
    assert auto.industry == "汽车"


def test_generic_digital_transformation_is_not_feature_or_booking() -> None:
    extraction = extract_message("我们最近在做销售流程数字化转型评估。")

    assert not extraction.feature_question
    assert not extraction.demo_intent
    assert extraction.budget is None
