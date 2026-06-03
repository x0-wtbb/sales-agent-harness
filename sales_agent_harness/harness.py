from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from . import tools
from .llm import ChatClient
from .grounding import (
    claim_type_for_facet,
    docs_from_tool_results,
    extract_requested_facets,
    find_evidence_for_facet,
    format_facets,
    has_technical_or_security_facets,
    last_kb_query,
    normalize_requested_facets,
    requested_facet_ids,
)
from .models import ActionPlan, AgentResponse, AgentState, Claim, ConversationRequest, GroundedResponse, PublicAgentState, ToolCall, ToolResult, TriggerEvent
from .policy import MockLLMPlanner, PolicyRouter, RuleBasedPlanner
from .state import BookingStateMachine, MemoryManager, build_handoff_crm_payload
from .store import InMemoryStore
from .validators import FallbackPolicy, PostconditionValidator, PreconditionValidator


class SalesAgentHarness:
    def __init__(
        self,
        agent_version: str = "harness_v2",
        mock_overrides: Optional[Dict] = None,
        reset_tools: bool = False,
        debug: bool = False,
        include_trajectory: bool = False,
        planner_mode: Optional[str] = None,
        llm_client: Optional[ChatClient] = None,
        store: Optional[InMemoryStore] = None,
        session_id: Optional[str] = None,
    ):
        self.agent_version = agent_version
        self.mock_overrides = mock_overrides or {}
        self.reset_tools = reset_tools
        self.debug = debug
        self.include_trajectory = include_trajectory
        self.store = store or InMemoryStore()
        self.session_id = session_id
        if reset_tools:
            self.store.reset(self.mock_overrides)
        elif self.mock_overrides:
            self.store.mock_overrides.update(self.mock_overrides)
        self.planner_mode = planner_mode or os.environ.get("SALES_AGENT_PLANNER")
        self.router = PolicyRouter()
        self.planner = self._build_planner(llm_client)

    def handle_conversation(
        self,
        request: ConversationRequest,
        debug: Optional[bool] = None,
        include_trajectory: Optional[bool] = None,
    ) -> AgentResponse:
        debug_enabled = self.debug if debug is None else debug
        trajectory_enabled = self.include_trajectory if include_trajectory is None else include_trajectory
        run_id = self._create_run_id()
        session_id = self.session_id or request.lead_id
        state = self.store.get_session(session_id)
        state_source = "store" if state is not None else "new"
        executed: List[ToolResult] = []
        if state is None:
            state = MemoryManager.init_state(request.lead_id)
        state.outbox = self.store.list_outbox(state.lead_id)
        state.add_trajectory(
            step="state_loaded",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "trigger_type": "conversation",
                "session_id": session_id,
                "source": state_source,
                "pending_outbox_count": len(self.store.list_pending_outbox(state.lead_id)),
                "offered_slots_source": "store" if state.booking.offered_slots else None,
                "offered_slot_ids": [slot.get("slot_id") for slot in state.booking.offered_slots],
            },
        )
        state.add_trajectory(
            step="run_started",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "trigger_type": "conversation",
                "state_summary_before": _state_summary(state),
            },
        )

        if state_source == "new":
            lead_context = tools.get_lead_context_with_store(self.store, request.lead_id)
            executed.append(lead_context)
            state.add_trajectory(
                step="tool_call",
                tool_name="get_lead_context",
                arguments=lead_context.arguments,
                status="success" if lead_context.success else "failed",
                data={
                    "run_id": run_id,
                    "agent_version": self.agent_version,
                    "trigger_type": "conversation",
                    "success": lead_context.success,
                    "error": lead_context.error,
                    "result": lead_context.data,
                },
                error=lead_context.error,
            )
            if lead_context.success:
                MemoryManager.merge_lead_context(state, lead_context.data)
                state.add_trajectory(
                    step="lead_context_loaded",
                    data={
                        "run_id": run_id,
                        "agent_version": self.agent_version,
                        "trigger_type": "conversation",
                        "lead_context_loaded": True,
                        "lead_context": lead_context.data,
                        "state_summary_after": _state_summary(state),
                    },
                )

        MemoryManager.merge_conversation_delta(state, request.conversation)
        state.outbox = self.store.list_outbox(state.lead_id)
        state.add_trajectory(
            step="extract_state",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "trigger_type": "conversation",
                "collected_info": state.collected_info.model_dump(mode="json"),
                "booking_status": state.booking.status,
                "risk_flags": state.risk_flags,
                "conflicts": state.context_conflicts,
                "pending_outbox_count": len(self.store.list_pending_outbox(state.lead_id)),
                "state_summary_after": _state_summary(state),
            },
        )
        state.add_trajectory(
            step="state_merge_result",
            status="success",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "trigger_type": "conversation",
                "state_summary_after": _state_summary(state),
                "conflicts": state.context_conflicts,
            },
        )

        intent = self.router.detect_intent(request.conversation, state)
        state.allowed_actions = self.router.compute_allowed_actions(state, intent)
        state.add_trajectory(
            step="policy_routing",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "intent": intent,
                "allowed_actions": state.allowed_actions,
                "risk_flags": state.risk_flags,
                "state_summary_before": _state_summary(state),
            },
        )

        prompt_metadata = self._planner_prompt_metadata()
        if prompt_metadata:
            state.add_trajectory(
                step="prompt_loaded",
                data={
                    "run_id": run_id,
                    "agent_version": self.agent_version,
                    **prompt_metadata,
                },
            )

        plan = self.planner.plan(request.conversation, state, intent)
        state.grounded_claims = []
        state.last_claims = []
        raw_requested_facets = plan.metadata.get("requested_facets") or extract_requested_facets(MemoryManager.last_user_message(request.conversation))
        state.requested_facets = normalize_requested_facets(raw_requested_facets)
        if prompt_metadata:
            state.add_trajectory(
                step="llm_proposal",
                data={
                    "run_id": run_id,
                    "agent_version": self.agent_version,
                    "planner_type": plan.metadata.get("planner_type", prompt_metadata.get("planner_type")),
                    "prompt_version": plan.metadata.get("prompt_version", prompt_metadata.get("prompt_version")),
                    "prompt_hash": plan.metadata.get("prompt_hash", prompt_metadata.get("prompt_hash")),
                    "proposal": plan.model_dump(mode="json"),
                },
            )
        state.add_trajectory(
            step="planner_proposal",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "intent": intent,
                "allowed_actions": state.allowed_actions,
                "proposal": plan.model_dump(mode="json"),
            },
        )

        validation_errors: List[str] = []
        ordered_tool_calls = self._ordered_tool_calls(plan.tool_calls, intent)
        if ordered_tool_calls != plan.tool_calls:
            state.add_trajectory(
                step="tool_call_order_normalized",
                data={
                    "run_id": run_id,
                    "agent_version": self.agent_version,
                    "intent": intent,
                    "original_order": [call.tool_name for call in plan.tool_calls],
                    "normalized_order": [call.tool_name for call in ordered_tool_calls],
                },
            )

        for tool_call in ordered_tool_calls:
            precondition = PreconditionValidator.validate(tool_call, state)
            state.add_trajectory(
                step="precondition_validation",
                status="success" if precondition.ok else "failed",
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
                data={
                    "run_id": run_id,
                    "agent_version": self.agent_version,
                    "intent": intent,
                    "allowed_actions": state.allowed_actions,
                    "guardrail_errors": precondition.errors,
                },
                error=";".join(precondition.errors) if not precondition.ok else None,
            )
            if not precondition.ok:
                validation_errors.extend(precondition.errors)
                continue
            result = self._execute_tool(tool_call)
            executed.append(result)
            state.add_trajectory(
                step="tool_call",
                tool_name=result.tool_name,
                arguments=result.arguments,
                status="success" if result.success else "failed",
                data={
                    "run_id": run_id,
                    "agent_version": self.agent_version,
                    "trigger_type": "conversation",
                    "intent": intent,
                    "allowed_actions": state.allowed_actions,
                    "success": result.success,
                    "error": result.error,
                    "result": result.data,
                },
                error=result.error,
            )
            BookingStateMachine.apply_tool_result(state, result, self.store)
            if result.tool_name == "book_demo" and not result.success:
                break

        self._record_handoff_failure(state, executed, run_id, intent)
        self._compensate_handoff_crm_gap(state, executed, run_id, intent)
        assistant_message = ResponseBuilder.build(plan, state, executed, validation_errors)
        postcondition = PostconditionValidator.validate(assistant_message, state, executed)
        state.add_trajectory(
            step="postcondition_validation",
            status="success" if postcondition.ok else "failed",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "intent": intent,
                "allowed_actions": state.allowed_actions,
                "guardrail_errors": postcondition.errors,
                "state_summary_after": _state_summary(state),
            },
            error=";".join(postcondition.errors) if not postcondition.ok else None,
        )
        if not postcondition.ok:
            assistant_message = FallbackPolicy.safe_message(postcondition.errors, state)

        MemoryManager.finalize_state(state)
        state.outbox = self.store.list_outbox(state.lead_id)
        state.add_trajectory(
            step="final_response",
            data={
                "assistant_message": assistant_message,
                "claims": [claim.model_dump(mode="json") for claim in state.last_claims],
                "run_id": run_id,
                "agent_version": self.agent_version,
                "state_summary_after": _state_summary(state),
            },
        )
        state.add_trajectory(
            step="state_saved",
            data={
                "run_id": run_id,
                "agent_version": self.agent_version,
                "trigger_type": "conversation",
                "session_id": session_id,
                "pending_outbox_count": len(self.store.list_pending_outbox(state.lead_id)),
            },
        )
        self.store.save_session(session_id, state)
        return AgentResponse(
            assistant_message=assistant_message,
            tool_calls=executed if debug_enabled else _public_tool_calls(executed),
            state=PublicAgentState.from_agent_state(state),
            trajectory=state.trajectory if (debug_enabled or trajectory_enabled) else None,
            run_id=run_id,
            debug_state=state if debug_enabled else None,
        )

    def handle_trigger(self, event: TriggerEvent) -> AgentResponse:
        payload = dict(event.payload)
        payload.setdefault("lead_id", event.lead_id)
        request = ConversationRequest(**payload)
        return self.handle_conversation(request)

    def retry_pending_outbox(self, session_id: Optional[str] = None) -> List[ToolResult]:
        def write_from_payload(payload: Dict) -> ToolResult:
            return tools.write_crm_note_with_store(
                self.store,
                payload.get("lead_id"),
                payload.get("summary", ""),
                payload.get("qualification_level", "unknown"),
                payload.get("next_action", ""),
                payload.get("customer_pain", ""),
                payload.get("current_stage", ""),
            )

        results = self.store.retry_pending_outbox(write_from_payload)
        target_session = session_id or self.session_id
        if target_session:
            state = self.store.get_session(target_session)
            if state is not None:
                state.outbox = self.store.list_outbox(state.lead_id)
                self.store.save_session(target_session, state)
        return results

    @staticmethod
    def _create_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"run_{stamp}_{uuid4().hex[:8]}"

    @staticmethod
    def _ordered_tool_calls(tool_calls: List[ToolCall], intent: str) -> List[ToolCall]:
        if intent != "handoff_required":
            return tool_calls
        search_calls = [call for call in tool_calls if call.tool_name == "search_knowledge_base"]
        crm_calls = [call for call in tool_calls if call.tool_name == "write_crm_note"]
        handoff_calls = [call for call in tool_calls if call.tool_name == "handoff_to_human"]
        other_calls = [call for call in tool_calls if call.tool_name not in ["search_knowledge_base", "write_crm_note", "handoff_to_human"]]
        return search_calls + crm_calls + handoff_calls + other_calls

    def _compensate_handoff_crm_gap(self, state: AgentState, tool_results: List[ToolResult], run_id: str, intent: str) -> None:
        handoff_result = _last_result(tool_results, "handoff_to_human")
        if not handoff_result or not handoff_result.success:
            return
        crm_result = _last_result(tool_results, "write_crm_note")
        if crm_result and crm_result.success:
            return

        payload = _handoff_compensation_payload(state, crm_result, handoff_result)
        tools.enqueue_crm_after_handoff(
            self.store,
            state.lead_id,
            handoff_result.data.get("ticket_id", ""),
            payload,
            crm_result.error if crm_result else "crm_write_not_executed",
        )
        state.crm.status = "PENDING"
        if "HANDOFF_WITH_CRM_PENDING" not in state.risk_flags:
            state.risk_flags.append("HANDOFF_WITH_CRM_PENDING")
        state.outbox = self.store.list_outbox(state.lead_id)
        state.add_trajectory(
            step="handoff_crm_compensation",
            status="success",
            tool_name="write_crm_note",
            arguments=payload,
            data={
                "run_id": run_id,
                "intent": intent,
                "handoff_ticket_id": handoff_result.data.get("ticket_id"),
                "crm_error": crm_result.error if crm_result else "crm_write_not_executed",
                "pending_outbox_count": len(self.store.list_pending_outbox(state.lead_id)),
            },
        )

    def _record_handoff_failure(self, state: AgentState, tool_results: List[ToolResult], run_id: str, intent: str) -> None:
        handoff_result = _last_result(tool_results, "handoff_to_human")
        if not handoff_result or handoff_result.success:
            return
        if "HANDOFF_FAILED" not in state.risk_flags:
            state.risk_flags.append("HANDOFF_FAILED")
        state.next_action = "人工转接失败，需要立即重试或通知销售负责人"
        state.add_trajectory(
            step="handoff_failure_recorded",
            status="failed",
            tool_name="handoff_to_human",
            arguments=handoff_result.arguments,
            data={"run_id": run_id, "intent": intent, "error": handoff_result.error},
            error=handoff_result.error,
        )

    def _build_planner(self, llm_client: Optional[ChatClient]):
        mode = self.planner_mode
        prompt_version = "prompt_v1" if self.agent_version == "prompt_v1" else "prompt_v2"
        if mode is None:
            if self.agent_version in ["prompt_v1", "prompt_v2"]:
                return MockLLMPlanner(prompt_version=self.agent_version)
            return RuleBasedPlanner()
        if mode == "auto":
            mode = "mock_llm"
        if mode == "llm":
            raise ValueError("planner_mode=llm is disabled in this offline mock harness; use mock_llm or rule.")
        if mode == "mock_llm":
            return MockLLMPlanner(prompt_version=prompt_version, agent_version=self.agent_version)
        if mode == "rule":
            return RuleBasedPlanner()
        raise ValueError(f"Unsupported planner_mode: {self.planner_mode}")

    def _planner_prompt_metadata(self) -> Dict:
        metadata_fn = getattr(self.planner, "prompt_metadata", None)
        if not callable(metadata_fn):
            return {}
        return metadata_fn()

    def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        args = tool_call.arguments
        if tool_call.tool_name == "search_knowledge_base":
            return tools.search_knowledge_base_with_store(self.store, args.get("query", ""))
        if tool_call.tool_name == "check_calendar":
            return tools.check_calendar_with_store(self.store, args.get("timezone"), args.get("duration_minutes", 30))
        if tool_call.tool_name == "book_demo":
            return tools.book_demo_with_store(self.store, args.get("lead_id"), args.get("slot_id"), args.get("attendee_email"), args.get("summary", ""))
        if tool_call.tool_name == "write_crm_note":
            return tools.write_crm_note_with_store(
                self.store,
                args.get("lead_id"),
                args.get("summary", ""),
                args.get("qualification_level", "unknown"),
                args.get("next_action", ""),
                args.get("customer_pain", ""),
                args.get("current_stage", ""),
            )
        if tool_call.tool_name == "handoff_to_human":
            return tools.handoff_to_human_with_store(self.store, args.get("lead_id"), args.get("reason", ""), args.get("urgency", "high"))
        if tool_call.tool_name == "get_lead_context":
            return tools.get_lead_context_with_store(self.store, args.get("lead_id"))
        raise ValueError(f"Unsupported tool: {tool_call.tool_name}")


class ResponseBuilder:
    @staticmethod
    def build(plan: ActionPlan, state: AgentState, tool_results: List[ToolResult], validation_errors: List[str]) -> str:
        kind = plan.metadata.get("response_kind")
        state.grounded_claims = []
        state.last_claims = []
        if validation_errors:
            return FallbackPolicy.safe_message(validation_errors, state)
        if kind == "pricing":
            return "公开知识库没有固定价格。具体报价需要销售同事根据你们的规模、使用场景和部署方式确认，我可以先帮你记录需求并转给同事给出准确报价。"
        if kind == "metric":
            return "公开知识库不承诺固定效果比例。实际效果会受线索质量、销售流程、团队执行和系统接入情况影响，我可以先了解你们当前流程，再判断适合从哪里介入。"
        if kind == "customer_case":
            return "公开知识库里没有可公开的客户名称。你如果需要案例材料，我可以记录这个需求，请销售同事提供可分享的版本。"
        if kind == "product":
            grounded = ResponseBuilder._grounded_product_response(plan, state, tool_results, force_handoff=False)
            state.grounded_claims = grounded.claims
            state.last_claims = grounded.claims
            return grounded.message
        if kind == "technical_or_security":
            grounded = ResponseBuilder._grounded_product_response(plan, state, tool_results, force_handoff=True)
            state.grounded_claims = grounded.claims
            state.last_claims = grounded.claims
            return grounded.message
        if kind == "handoff":
            return ResponseBuilder._handoff_message(tool_results)
        if kind == "handoff_with_product":
            return ResponseBuilder._handoff_with_product_message(plan, state, tool_results)
        if kind == "generic_clarification":
            return "可以，我先不急着推进 Demo 或报价。你们这次评估更关注行业做法、现有销售流程诊断，还是具体工具能力对比？"
        if kind == "qualification":
            return ResponseBuilder._qualification_message(state)
        if kind == "ambiguous_timezone":
            return "CST 可能指多个时区。请确认你希望按哪个城市或 IANA 时区安排，例如 Asia/Shanghai 或 America/Chicago。"
        if kind == "timezone_offset_needs_city":
            return "GMT/UTC 偏移可能对应多个城市。请告诉我你所在城市或 IANA 时区，例如 Asia/Tokyo 或 America/New_York。"
        if kind == "booking_missing_fields":
            return ResponseBuilder._booking_missing_message(plan.metadata.get("missing_fields", []))
        if kind == "offer_slots":
            return ResponseBuilder._offer_slots_message(state, tool_results)
        if kind == "waiting_slot_selection":
            return ResponseBuilder._waiting_slot_selection_message(state)
        if kind == "book_demo":
            return ResponseBuilder._book_demo_message(state, tool_results)
        return plan.assistant_message_draft or "我需要再确认一下你的需求，才能继续处理。"

    @staticmethod
    def _grounded_product_response(
        plan: ActionPlan,
        state: AgentState,
        tool_results: List[ToolResult],
        force_handoff: bool,
    ) -> GroundedResponse:
        question = last_kb_query(tool_results)
        requested_facets = normalize_requested_facets(plan.metadata.get("requested_facets") or state.requested_facets or extract_requested_facets(question))
        requested_facet_names = requested_facet_ids(requested_facets)
        state.requested_facets = requested_facets
        docs = docs_from_tool_results(tool_results)
        claims: List[Claim] = []
        missing_facets: List[str] = []

        for facet in requested_facet_names:
            evidence = find_evidence_for_facet(facet, docs)
            if not evidence:
                missing_facets.append(facet)
                continue
            claim_text = ResponseBuilder._claim_text_for_facet(facet, evidence)
            if not claim_text:
                missing_facets.append(facet)
                continue
            evidence_id = evidence.get("id")
            claims.append(
                Claim(
                    text=claim_text,
                    claim_type=claim_type_for_facet(facet),
                    facet=facet,
                    evidence_ids=[evidence_id] if evidence_id else [],
                    evidence_span=evidence.get("content") or evidence.get("title"),
                    confidence=evidence.get("confidence", "low"),
                )
            )

        if not claims:
            return GroundedResponse(
                message=ResponseBuilder._ungrounded_product_message(requested_facets, tool_results, force_handoff),
                claims=[],
            )

        message = ResponseBuilder._render_claims(claims)
        if missing_facets:
            message += f" 其中关于 {format_facets(missing_facets)}，当前知识库没有明确资料，需要销售或技术同事确认。"
        if force_handoff:
            message += " " + ResponseBuilder._technical_handoff_status(tool_results)
        return GroundedResponse(message=message, claims=claims)

    @staticmethod
    def _claim_text_for_facet(facet: str, evidence: Dict[str, object]) -> str:
        if facet == "lead_scoring":
            return "产品支持线索自动分层。"
        if facet == "follow_up_reminder":
            return "产品支持销售跟进提醒。"
        if facet == "follow_up_suggestion":
            return "产品支持跟进建议生成。"
        if facet == "crm_integration":
            return "产品支持与常见 CRM 系统集成，具体集成方式需要销售或实施同事结合客户系统确认。"
        if facet == "crm_field_sync":
            return "CRM 同步通常覆盖线索字段、跟进记录、任务状态和预约结果，字段映射需按客户系统确认。"
        return ""

    @staticmethod
    def _render_claims(claims: List[Claim]) -> str:
        lead_facets = ["lead_scoring", "follow_up_reminder", "follow_up_suggestion"]
        lead_labels = {
            "lead_scoring": "线索自动分层",
            "follow_up_reminder": "销售跟进提醒",
            "follow_up_suggestion": "跟进建议生成",
        }
        messages: List[str] = []
        lead_claims = [claim for claim in claims if claim.facet in lead_facets]
        if lead_claims:
            labels = [lead_labels[claim.facet] for claim in lead_claims if claim.facet]
            messages.append("知识库显示，产品支持" + "、".join(labels) + "。")
        for claim in claims:
            if claim.facet in lead_facets:
                continue
            messages.append("知识库显示，" + claim.text)
        return " ".join(messages)

    @staticmethod
    def _ungrounded_product_message(requested_facets: List[str], tool_results: List[ToolResult], force_handoff: bool) -> str:
        if force_handoff or has_technical_or_security_facets(requested_facets):
            scope = ResponseBuilder._technical_scope_text(requested_facets)
            return f"这些问题涉及{scope}，我不能在没有资料的情况下确认。{ResponseBuilder._technical_handoff_status(tool_results)}"
        return "我现在没有查到足够可靠的产品资料来确认这个功能细节。为避免给你不准确的信息，我可以先记录你的需求，并请销售或产品同事确认后回复。"

    @staticmethod
    def _technical_scope_text(facets: List[str]) -> str:
        facet_names = requested_facet_ids(facets)
        scopes: List[str] = []
        if any(facet in facet_names for facet in ["technical_architecture", "modeling_approach"]):
            scopes.append("技术架构")
        if "api_latency" in facet_names:
            scopes.append("性能指标")
        if "private_deployment" in facet_names:
            scopes.append("部署方式")
        if any(facet in facet_names for facet in ["data_storage", "training_data_compliance"]):
            scopes.append("数据合规")
        if "security_audit" in facet_names:
            scopes.append("安全合规")
        if not scopes:
            scopes.append("产品能力或技术细节")
        return "、".join(scopes)

    @staticmethod
    def _technical_handoff_status(tool_results: List[ToolResult]) -> str:
        handoff_result = _last_result(tool_results, "handoff_to_human")
        crm_result = _last_result(tool_results, "write_crm_note")
        if handoff_result and not handoff_result.success:
            return "我已经记录了这个问题，但人工转接失败，需要立即重试或通知销售负责人。"
        if handoff_result and handoff_result.success and crm_result and crm_result.success:
            return "我已经把问题记录到 CRM，并转给销售或技术同事确认。"
        if handoff_result and handoff_result.success:
            return "我已经先转给销售或技术同事；CRM 记录暂时待重试，系统里已保留补偿任务。"
        if crm_result and crm_result.success:
            return "我已经先把问题记录到 CRM，人工转接还需要重试。"
        return "我可以先记录下来，并请销售或技术同事确认。"

    @staticmethod
    def _qualification_message(state: AgentState) -> str:
        info = state.collected_info
        bits = []
        if info.company_size:
            bits.append(f"{info.company_size} 人公司")
        if info.industry:
            bits.append(info.industry)
        if info.sales_team_size:
            bits.append(f"{info.sales_team_size} 人销售团队")
        if info.pain_point:
            bits.append(f"主要痛点是{info.pain_point}")
        lead_desc = "，".join(bits) if bits else "你们的需求"
        budget_text = "预算还没定也没关系，" if info.budget == "not_set" else ""
        decline_text = "我不会安排 Demo；" if "BOOKING_DECLINED" in state.risk_flags else ""
        next_action_text = state.next_action if state.next_action.startswith("建议") else f"建议{state.next_action}"
        return (
            f"{lead_desc}，这个场景已经比较明确。{budget_text}{decline_text}"
            f"我会先把线索信息记录到 CRM，下一步{next_action_text}。"
        )

    @staticmethod
    def _handoff_message(tool_results: List[ToolResult]) -> str:
        handoff_result = _last_result(tool_results, "handoff_to_human")
        crm_result = _last_result(tool_results, "write_crm_note")
        if handoff_result and not handoff_result.success:
            if crm_result and crm_result.success:
                return "这类合同、法务、安全审计、DPA、SOC2、GDPR、等保、个保法或数据保护材料需要销售或安全同事提供。我已经把需求记录到 CRM，但人工转接失败，需要立即重试或通知销售负责人。"
            return "这类合同、法务、安全审计、DPA、SOC2、GDPR、等保、个保法或数据保护材料需要销售或安全同事提供。当前人工转接失败，我不能假定已经有人接手，需要立即重试。"
        if handoff_result and handoff_result.success and crm_result and crm_result.success:
            return "这类合同、法务、安全审计、DPA、SOC2、GDPR、等保、个保法或数据保护材料需要销售或安全同事提供。我已经把需求记录到 CRM，并转给人工同事继续跟进。"
        if handoff_result and handoff_result.success:
            return "这类合同、法务、安全审计、DPA、SOC2、GDPR、等保、个保法或数据保护材料需要销售或安全同事提供。我已经先转给人工同事；CRM 记录暂时待重试，系统里已保留补偿任务。"
        if crm_result and crm_result.success:
            return "这类合同、法务、安全审计、DPA、SOC2、GDPR、等保、个保法或数据保护材料需要销售或安全同事提供。我已经先把需求记录到 CRM，但人工转接还需要重试。"
        return "这类合同、法务、安全审计、DPA、SOC2、GDPR、等保、个保法或数据保护材料需要销售或安全同事提供。我会保守处理，继续补齐人工接管所需的上下文。"

    @staticmethod
    def _handoff_with_product_message(plan: ActionPlan, state: AgentState, tool_results: List[ToolResult]) -> str:
        grounded = ResponseBuilder._grounded_product_response(plan, state, tool_results, force_handoff=False)
        state.grounded_claims = grounded.claims
        state.last_claims = grounded.claims
        product_message = grounded.message
        handoff_message = ResponseBuilder._handoff_message(tool_results)
        return f"{product_message} 另外，{handoff_message}"

    @staticmethod
    def _booking_missing_message(missing_fields: List[str]) -> str:
        if "invalid_email" in missing_fields:
            other_fields = [field for field in missing_fields if field != "invalid_email"]
            labels = {"timezone": "所在时区", "demo_purpose": "这次 Demo 想看的重点"}
            rest = "、".join(labels.get(field, field) for field in other_fields)
            suffix = f"另外还需要补充：{rest}。" if rest else ""
            return f"你给的邮箱看起来不完整，请提供包含完整域名的邮箱，例如 name@example.com。{suffix}信息齐了以后我再查可选时间。"
        labels = {"email": "邮箱", "timezone": "所在时区", "demo_purpose": "这次 Demo 想看的重点"}
        missing = "、".join(labels.get(field, field) for field in missing_fields)
        return f"可以安排 Demo。为了避免约错时间，请先补充：{missing}。信息齐了以后我再查可选时间。"

    @staticmethod
    def _offer_slots_message(state: AgentState, tool_results: List[ToolResult]) -> str:
        slots = []
        for result in tool_results:
            if result.tool_name == "check_calendar" and result.success:
                slots = result.data.get("slots", [])
        if not slots:
            return "我暂时没有查到可用时间，请确认时区或换一个时间范围。"
        slot_text = "、".join(f"{slot['display']}（{slot['slot_id']}）" for slot in slots)
        place = "新加坡" if state.collected_info.timezone == "Asia/Singapore" else state.collected_info.timezone
        purpose = state.collected_info.demo_purpose or "产品能力"
        purpose_text = purpose if purpose.startswith("了解") else f"看{purpose}"
        return f"我看到你在{place}，邮箱是 {state.collected_info.email}，这次想{purpose_text}。可选时间有：{slot_text}。你选一个后我再继续预约。"

    @staticmethod
    def _waiting_slot_selection_message(state: AgentState) -> str:
        if state.booking.status == "BOOKING_FAILED":
            slot_text = "、".join(f"{slot['display']}（{slot['slot_id']}）" for slot in state.booking.offered_slots)
            if slot_text:
                return f"上次选择的时间刚刚不可用，我不会确认预约成功。你可以改选这些时间：{slot_text}，或告诉我新的时间范围。"
            return "上次选择的时间刚刚不可用，我不会确认预约成功。请告诉我新的时间范围，我再重新查询。"
        if state.booking.slot_selection_status in ["rejected", "reschedule_requested"]:
            return "好的，我不会预约这个时间。你可以从剩余可选时间里选一个，或告诉我新的时间范围。"
        if state.booking.slot_selection_status == "ambiguous":
            return "我还不能确认你选择的是哪个时间。请明确选择一个可用时间，我再继续预约。"
        slot_text = "、".join(f"{slot['display']}（{slot['slot_id']}）" for slot in state.booking.offered_slots)
        return f"当前可选时间是：{slot_text}。请告诉我你选择哪个时间。"

    @staticmethod
    def _book_demo_message(state: AgentState, tool_results: List[ToolResult]) -> str:
        book_result = _last_result(tool_results, "book_demo")
        crm_result = _last_result(tool_results, "write_crm_note")
        if book_result and not book_result.success:
            return "这个时间刚刚不可用了，我不会把它记为预约成功。你可以从其它可选时间里换一个，或告诉我新的时间范围，我再重新查询。"
        if book_result and book_result.success:
            display = book_result.data.get("display", state.booking.selected_slot_id)
            timezone = book_result.data.get("timezone", state.collected_info.timezone)
            if crm_result and crm_result.success:
                return f"Demo 已预约：{display}（{timezone}），参会邮箱是 {state.collected_info.email}。我也已经把预约和需求同步到 CRM。"
            return f"Demo 已预约：{display}（{timezone}），参会邮箱是 {state.collected_info.email}。CRM 记录还在同步，我已保留重试任务，避免丢失这次预约。"
        return "我还不能确认 Demo 已经预约成功，请先选择一个可用时间。"


def _last_result(tool_results: List[ToolResult], tool_name: str) -> Optional[ToolResult]:
    for result in reversed(tool_results):
        if result.tool_name == tool_name:
            return result
    return None


def _public_tool_calls(tool_results: List[ToolResult]) -> List[ToolCall]:
    return [ToolCall(tool_name=result.tool_name, arguments=dict(result.arguments)) for result in tool_results]


def _handoff_compensation_payload(state: AgentState, crm_result: Optional[ToolResult], handoff_result: ToolResult) -> Dict[str, object]:
    if crm_result and crm_result.error not in tools.CRM_CONTRACT_ERRORS:
        return dict(crm_result.arguments)
    return build_handoff_crm_payload(
        state,
        requested_scope=str(handoff_result.arguments.get("reason") or ""),
    )


def _has_successful_product_kb(tool_results: List[ToolResult]) -> bool:
    return any(
        result.tool_name == "search_knowledge_base"
        and result.success
        and result.data.get("topic") in ["lead_follow_up", "crm_integration"]
        and bool(result.data.get("documents"))
        for result in tool_results
    )


def _state_summary(state: AgentState) -> Dict[str, object]:
    return {
        "qualification_level": state.qualification_level,
        "missing_info": list(state.missing_info),
        "next_action": state.next_action,
        "risk_flags": list(state.risk_flags),
        "booking_status": state.booking.status,
        "crm_status": state.crm.status,
        "allowed_actions": list(state.allowed_actions),
    }
