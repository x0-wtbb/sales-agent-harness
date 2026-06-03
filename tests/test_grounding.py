from __future__ import annotations

import pytest

from sales_agent_harness.grounding import extract_requested_facets, find_evidence_for_facet, first_match, requested_facet_ids
from sales_agent_harness.harness import SalesAgentHarness
from sales_agent_harness.models import AgentState, Claim, ConversationRequest, ConversationTurn, GroundedResponse, ToolResult
from sales_agent_harness.state import BookingStateMachine
from sales_agent_harness.validators import ClaimValidator, PostconditionValidator


def _kb_result(doc: dict) -> ToolResult:
    return ToolResult(
        tool_name="search_knowledge_base",
        success=True,
        data={"documents": [doc], "evidence_ids": [doc["id"]]},
    )


def test_extract_requested_facets_returns_phrase_objects() -> None:
    facets = extract_requested_facets("核心架构是 RAG 吗？API p99 延迟是多少？")

    assert requested_facet_ids(facets) == ["technical_architecture", "modeling_approach", "api_latency"]
    assert facets[0].phrase == "核心架构"


@pytest.mark.parametrize(
    "text,expected_facets",
    [
        ("你们的核心架构是什么？", ["technical_architecture"]),
        ("底层用啥模型？", ["modeling_approach"]),
        ("fine-tuned 模型支持吗？", ["modeling_approach"]),
        ("API latency 和 p99 怎么样？", ["api_latency"]),
        ("能 on-prem 部署吗？", ["private_deployment"]),
        ("标准版报价是多少？", ["pricing"]),
        ("有没有客户案例？", ["customer_case"]),
        ("ROI 和转化率指标能保证吗？", ["metric"]),
        ("客户数据存哪里？", ["data_storage"]),
        ("有没有 SOC2 审计报告？", ["security_audit"]),
        ("能做线索自动分层吗？", ["lead_scoring"]),
        ("销售跟进提醒支持吗？", ["follow_up_reminder"]),
        ("CRM 字段同步怎么做？", ["crm_field_sync"]),
        ("能对接 Salesforce 吗？", ["crm_integration"]),
        ("请介绍产品能力。", []),
        ("你们支持微信吗？", []),
    ],
)
def test_extract_requested_facets_pattern_variants(text: str, expected_facets: list[str]) -> None:
    assert requested_facet_ids(extract_requested_facets(text)) == expected_facets


def test_first_match_returns_none_and_skips_empty_terms() -> None:
    assert first_match("RAG 架构", ["", "rag"]) == "rag"
    assert first_match("没有命中", ["", "rag"]) is None


def test_lead_followup_evidence_does_not_cover_api_latency() -> None:
    docs = [
        {
            "id": "kb_feature_001",
            "topic": "lead_follow_up",
            "content": "产品支持线索自动分层。",
            "supported_facets": ["lead_scoring"],
        }
    ]

    assert find_evidence_for_facet("lead_scoring", docs) is not None
    assert find_evidence_for_facet("api_latency", docs) is None


def test_crm_integration_evidence_does_not_cover_private_deployment() -> None:
    docs = [
        {
            "id": "kb_crm_001",
            "topic": "crm_integration",
            "content": "产品支持与常见 CRM 系统集成。",
            "supported_facets": ["crm_integration"],
        }
    ]

    assert find_evidence_for_facet("crm_integration", docs) is not None
    assert find_evidence_for_facet("private_deployment", docs) is None


def test_no_public_price_policy_comes_from_metadata_not_content_substring() -> None:
    state = AgentState(lead_id="L_TEST")
    result = ToolResult(
        tool_name="search_knowledge_base",
        success=True,
        data={
            "topic": "pricing",
            "documents": [
                {
                    "id": "kb_pricing_reworded",
                    "topic": "pricing",
                    "content": "公开知识库不会给固定报价。",
                    "policy": {"price_policy": "no_public_price"},
                    "supported_facets": ["pricing"],
                    "unsupported_facets": ["exact_price"],
                    "visibility": "public",
                }
            ],
            "evidence_ids": ["kb_pricing_reworded"],
        },
    )

    BookingStateMachine.apply_tool_result(state, result)

    assert state.evidence_policy.price_policy == "no_public_price"


def test_kb_result_updates_single_snapshot_with_coverage() -> None:
    state = AgentState(lead_id="L_TEST")
    result = ToolResult(
        tool_name="search_knowledge_base",
        arguments={"query": "线索自动分层"},
        success=True,
        data={
            "topic": "lead_follow_up",
            "topics": ["lead_follow_up"],
            "documents": [
                {
                    "id": "kb_feature_001",
                    "topic": "lead_follow_up",
                    "content": "产品支持线索自动分层。",
                    "policy": {"feature_policy": "feature_supported"},
                    "supported_facets": ["lead_scoring"],
                }
            ],
            "evidence_ids": ["kb_feature_001"],
        },
    )

    BookingStateMachine.apply_tool_result(state, result)

    assert state.last_kb_evidence_ids == ["kb_feature_001"]
    assert state.last_kb_evidence_topics == ["lead_follow_up"]
    assert state.last_kb_evidence_coverage == {"feature": True}
    assert not hasattr(state, "last_kb_evidence_for_price")
    assert not hasattr(state, "last_kb_evidence_for_metric")
    assert not hasattr(state, "last_kb_evidence_for_customer_case")


def test_price_obfuscation_patterns_are_smoke_layer() -> None:
    for text in ["一年 5W。", "一年 50000 RMB。", "一年五万。", "约 100K USD。"]:
        assert ClaimValidator.contains_price(text)


def test_customer_case_variants_fail_without_evidence() -> None:
    state = AgentState(lead_id="L_TEST")

    result = PostconditionValidator.validate("字节、美团、理想汽车这些客户都在用。", state, [])

    assert not result.ok
    assert "UNSUPPORTED_CUSTOMER_CASE" in result.errors


def test_grounded_response_feature_claim_without_evidence_ids_fails() -> None:
    state = AgentState(lead_id="L_TEST")
    grounded = GroundedResponse(
        message="产品支持 CRM 集成。",
        claims=[Claim(text="产品支持 CRM 集成。", claim_type="feature", facet="crm_integration")],
    )
    state.last_claims = grounded.claims

    result = PostconditionValidator.validate(grounded.message, state, [])

    assert not result.ok
    assert "UNSUPPORTED_FEATURE_CLAIM" in result.errors


def test_grounded_response_facet_mismatch_fails() -> None:
    state = AgentState(lead_id="L_TEST")
    state.evidence_policy.supported_facets = {"lead_scoring": ["kb_feature_001"]}
    doc = {
        "id": "kb_feature_001",
        "topic": "lead_follow_up",
        "content": "产品支持线索自动分层。",
        "supported_facets": ["lead_scoring"],
        "visibility": "public",
    }
    state.last_claims = [
        Claim(
            text="产品支持 CRM 集成。",
            claim_type="feature",
            facet="crm_integration",
            evidence_ids=["kb_feature_001"],
            evidence_span="产品支持线索自动分层。",
            confidence="high",
        )
    ]

    result = PostconditionValidator.validate("产品支持 CRM 集成。", state, [_kb_result(doc)])

    assert not result.ok
    assert "EVIDENCE_FACET_MISMATCH" in result.errors


def test_technical_question_without_exact_evidence_handoffs() -> None:
    request = ConversationRequest(
        lead_id="L123",
        conversation=[
            ConversationTurn(role="user", content="你们 API p99 延迟是多少？支持私有化部署吗？数据存哪里？")
        ],
    )

    response = SalesAgentHarness(planner_mode="rule", debug=True, session_id="grounding_technical:L123").handle_conversation(request)
    tool_names = [result.tool_name for result in response.tool_calls]

    assert "search_knowledge_base" in tool_names
    assert "write_crm_note" in tool_names
    assert "handoff_to_human" in tool_names
    assert "线索自动分层" not in response.assistant_message
    assert "不能在没有资料的情况下确认" in response.assistant_message
