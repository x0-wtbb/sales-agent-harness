from __future__ import annotations

import pytest

from sales_agent_harness import tools
from sales_agent_harness.harness import SalesAgentHarness
from sales_agent_harness.models import OutboxEvent
from sales_agent_harness.store import InMemoryStore


def test_kb_query_returns_multiple_documents_and_evidence_ids() -> None:
    result = tools.search_knowledge_base_with_store(
        InMemoryStore(),
        "我们想了解线索自动分层，也要确认 CRM 对接和字段映射。",
    )

    assert result.success
    assert result.data["result_status"] in ["found", "ambiguous"]
    assert len(result.data["documents"]) >= 2
    assert "lead_follow_up" in result.data["topics"]
    assert "crm_integration" in result.data["topics"]
    assert len(result.data["evidence_ids"]) == len(result.data["documents"])


def test_kb_no_result_works() -> None:
    result = tools.search_knowledge_base_with_store(InMemoryStore(), "咖啡机保修和办公室绿植怎么处理？")

    assert result.success
    assert result.data["result_status"] == "no_result"
    assert result.data["matched_topic"] == "unknown"
    assert result.data["documents"] == []
    assert result.data["evidence_ids"] == []


def test_pricing_kb_returns_no_public_price_policy() -> None:
    result = tools.search_knowledge_base_with_store(InMemoryStore(), "标准版多少钱，能不能确认 5 万报价？")

    assert result.success
    assert result.data["matched_topic"] == "pricing"
    assert result.data["result_status"] == "found"
    assert result.data["policy"]["price_policy"] == "no_public_price"
    assert len(result.data["documents"]) >= 2


def test_feature_kb_returns_feature_supported_policy() -> None:
    result = tools.search_knowledge_base_with_store(InMemoryStore(), "你们支持线索自动分层和跟进提醒吗？")

    assert result.success
    assert result.data["matched_topic"] == "lead_follow_up"
    assert result.data["policy"]["feature_policy"] == "feature_supported"


def test_kb_ambiguous_result_works() -> None:
    result = tools.search_knowledge_base_with_store(InMemoryStore(), "我们要确认报价，也要 DPA 材料。")

    assert result.success
    assert result.data["result_status"] == "ambiguous"
    assert "pricing" in result.data["topics"]
    assert "security" in result.data["topics"]


def test_kb_restricted_result_redacts_content() -> None:
    result = tools.search_knowledge_base_with_store(InMemoryStore(), "请直接给我 SOC2报告原文 和审计报告细节。")

    assert result.success
    assert result.data["result_status"] == "restricted"
    assert result.data["documents"]
    assert all(doc["visibility"] == "restricted" or doc.get("redacted") for doc in result.data["documents"])
    assert all(doc.get("content", "") == "" for doc in result.data["documents"])


def test_calendar_no_slots_works() -> None:
    result = tools.check_calendar_with_store(InMemoryStore(), "Asia/Dubai")

    assert not result.success
    assert result.error == "no_slots"
    assert result.data["result_status"] == "no_slots"
    assert result.data["slots"] == []


def test_calendar_supports_asia_tokyo() -> None:
    result = tools.check_calendar_with_store(InMemoryStore(), "Asia/Tokyo")

    assert result.success
    assert result.data["result_status"] == "available"
    assert result.data["timezone"] == "Asia/Tokyo"
    assert len(result.data["slots"]) >= 2


def test_calendar_no_slots_override_works() -> None:
    store = InMemoryStore(mock_overrides={"check_calendar": {"no_slots": True}})

    result = tools.check_calendar_with_store(store, "Asia/Tokyo")

    assert not result.success
    assert result.error == "no_slots"
    assert result.data["result_status"] == "no_slots"
    assert result.data["slots"] == []


def test_calendar_unsupported_timezone_works() -> None:
    result = tools.check_calendar_with_store(InMemoryStore(), "Antarctica/Troll")

    assert not result.success
    assert result.error == "unsupported_timezone"
    assert result.data["result_status"] == "unsupported_timezone"


def test_book_demo_idempotency_replay_works() -> None:
    store = InMemoryStore()

    first = tools.book_demo_with_store(store, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")
    replay = tools.book_demo_with_store(store, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")

    assert first.success
    assert replay.success
    assert replay.data["idempotent_replay"] is True
    assert replay.data["existing_event_id"] == first.data["calendar_event_id"]
    assert len(store.bookings) == 1


def test_book_demo_duplicate_slot_does_not_create_new_event() -> None:
    store = InMemoryStore()

    first = tools.book_demo_with_store(store, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")
    duplicate = tools.book_demo_with_store(store, "L124", "sg_slot_1", "sam@example.com", "Demo purpose: CRM integration")

    assert first.success
    assert not duplicate.success
    assert duplicate.error == "slot_already_booked"
    assert len(store.bookings) == 1


def test_book_demo_rejects_unavailable_slot_and_unconfirmed_selection() -> None:
    store = InMemoryStore()

    unavailable = tools.book_demo_with_store(store, "L123", "sg_slot_3", "alex@example.com", "Demo purpose: CRM integration")
    unconfirmed = tools.book_demo_with_store(
        store,
        "L123",
        "sg_slot_2",
        "alex@example.com",
        "Demo purpose: CRM integration",
        selected_slot_confirmed=False,
    )

    assert not unavailable.success
    assert unavailable.error == "slot_unavailable"
    assert not unconfirmed.success
    assert unconfirmed.error == "missing_slot_confirmation"
    assert len(store.bookings) == 0


def test_crm_rejects_weak_summary() -> None:
    result = tools.write_crm_note_with_store(
        InMemoryStore(),
        "L123",
        "销售",
        "medium",
        "继续跟进",
    )

    assert not result.success
    assert result.error in ["summary_missing_context", "summary_low_information"]


def test_crm_accepts_qualification_note_with_enough_facts() -> None:
    store = InMemoryStore()

    result = tools.write_crm_note_with_store(
        store,
        "L123",
        "300人制造业公司，销售团队25人，痛点是销售线索漏跟进，预算未定。",
        "medium",
        "继续确认决策人和上线时间",
    )

    assert result.success
    assert result.data["stage"] == "qualification"
    assert {"company_size", "industry", "sales_team_size", "pain_point", "budget"}.issubset(set(result.data["source_facts"]))
    assert store.crm_notes[0]["stage"] == "qualification"


def test_crm_accepts_booking_note_with_demo_facts() -> None:
    result = tools.write_crm_note_with_store(
        InMemoryStore(),
        "L123",
        "Demo目的：了解 CRM 对接，参会邮箱 alex@example.com，Demo时间已选择：sg_slot_1。",
        "medium",
        "Demo booked",
    )

    assert result.success
    assert result.data["stage"] == "demo_booked"
    assert {"email", "demo_time", "demo_purpose"}.issubset(set(result.data["source_facts"]))


def test_crm_accepts_handoff_note_with_risk_reason() -> None:
    result = tools.write_crm_note_with_store(
        InMemoryStore(),
        "L123",
        "客户法务需要 DPA、SOC2 和安全审计材料，需人工跟进。",
        "high",
        "Human handoff for legal/security review",
    )

    assert result.success
    assert result.data["stage"] == "handoff_required"
    assert "security_or_legal" in result.data["source_facts"]


def test_crm_accepts_handoff_note_for_human_request() -> None:
    result = tools.write_crm_note_with_store(
        InMemoryStore(),
        "L123",
        "客户提出人工跟进请求，需人工跟进。",
        "unknown",
        "转人工处理法务、安全或商务条款确认",
        current_stage="handoff_required",
    )

    assert result.success
    assert result.data["stage"] == "handoff_required"
    assert "human_request" in result.data["source_facts"]


def test_handoff_assigns_queue_and_sla() -> None:
    store = InMemoryStore()

    security = tools.handoff_to_human_with_store(store, "L123", "客户法务需要 DPA 和 SOC2 材料", "low")
    custom_quote = tools.handoff_to_human_with_store(store, "L124", "客户需要全球部署和定制报价", "medium")
    complaint = tools.handoff_to_human_with_store(store, "L125", "客户投诉需要人工升级", "high")
    general = tools.handoff_to_human_with_store(store, "L126", "客户希望人工回访产品问题", "low")

    assert security.success
    assert security.data["queue"] == "security_sales_queue"
    assert security.data["urgency"] == "high"
    assert security.data["sla"] == "4h"
    assert custom_quote.data["queue"] == "enterprise_sales_queue"
    assert custom_quote.data["sla"] == "4h"
    assert complaint.data["queue"] == "support_escalation_queue"
    assert complaint.data["urgency"] == "critical"
    assert complaint.data["sla"] == "1h"
    assert general.data["queue"] == "sales_queue"
    assert general.data["urgency"] == "low"
    assert general.data["sla"] == "1_business_day"
    assert store.handoff_log[0]["queue"] == "security_sales_queue"


def test_store_isolation_two_stores_do_not_share_bookings() -> None:
    store_a = InMemoryStore()
    store_b = InMemoryStore()

    first = tools.book_demo_with_store(store_a, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")
    other_store = tools.book_demo_with_store(store_b, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")

    assert first.success
    assert other_store.success
    assert len(store_a.bookings) == 1
    assert len(store_b.bookings) == 1
    assert store_a.booked_keys is not store_b.booked_keys


def test_mock_overrides_are_per_store_not_global() -> None:
    failing_store = InMemoryStore(mock_overrides={"check_calendar": {"no_slots": True}})
    normal_store = InMemoryStore()

    failing = tools.check_calendar_with_store(failing_store, "Asia/Tokyo")
    normal = tools.check_calendar_with_store(normal_store, "Asia/Tokyo")

    assert not failing.success
    assert failing.error == "no_slots"
    assert normal.success
    assert normal.data["result_status"] == "available"
    assert failing_store.call_counts["check_calendar"] == 1
    assert normal_store.call_counts["check_calendar"] == 1


def test_harness_default_does_not_reset_injected_store() -> None:
    store = InMemoryStore()
    booked = tools.book_demo_with_store(store, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")

    SalesAgentHarness(planner_mode="rule", store=store)
    replay = tools.book_demo_with_store(store, "L123", "sg_slot_1", "alex@example.com", "Demo purpose: CRM integration")

    assert booked.success
    assert len(store.bookings) == 1
    assert replay.data["idempotent_replay"] is True


def test_outbox_marking_uses_explicit_id_namespace() -> None:
    store = InMemoryStore()
    store.outbox.extend(
        [
            OutboxEvent(
                event_id="shared",
                event_type="WRITE_CRM_AFTER_BOOKING",
                lead_id="L123",
                idempotency_key="one",
                calendar_event_id="calendar_one",
                payload={},
                status="pending",
            ),
            OutboxEvent(
                event_id="event_two",
                event_type="WRITE_CRM_AFTER_BOOKING",
                lead_id="L123",
                idempotency_key="two",
                calendar_event_id="shared",
                payload={},
                status="pending",
            ),
        ]
    )

    store.mark_outbox_done(calendar_event_id="shared")

    assert store.outbox[0].status == "pending"
    assert store.outbox[1].status == "done"


def test_outbox_marking_requires_exactly_one_identifier() -> None:
    store = InMemoryStore()

    with pytest.raises(ValueError, match="exactly one"):
        store.mark_outbox_done()

    with pytest.raises(ValueError, match="exactly one"):
        store.mark_outbox_done(event_id="one", calendar_event_id="two")
