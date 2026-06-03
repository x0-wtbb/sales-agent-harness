from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


QualificationLevel = Literal["high", "medium", "low", "unknown"]
BookingStatus = Literal[
    "NO_DEMO_INTENT",
    "DEMO_INTERESTED",
    "READY_TO_CHECK_CALENDAR",
    "SLOTS_OFFERED",
    "SLOT_SELECTED",
    "BOOKED_CRM_PENDING",
    "BOOKED_CRM_SYNCED",
    "BOOKING_FAILED",
]
CRMStatus = Literal["NOT_WRITTEN", "WRITTEN", "PENDING", "FAILED"]
SlotSelectionStatus = Literal["none", "confirmed", "rejected", "reschedule_requested", "ambiguous"]
FacetName = Literal[
    "lead_scoring",
    "follow_up_reminder",
    "follow_up_suggestion",
    "crm_integration",
    "pricing",
    "customer_case",
    "metric",
    "technical_architecture",
    "modeling_approach",
    "api_latency",
    "private_deployment",
    "data_storage",
    "training_data_compliance",
    "security_audit",
    "delivery_commitment",
    "crm_field_sync",
    "implementation_review",
    "product_capability",
]
ClaimType = Literal[
    "price",
    "customer_case",
    "metric",
    "feature",
    "technical_architecture",
    "deployment",
    "security",
    "data_compliance",
    "delivery_commitment",
    "booking",
    "crm",
    "other",
]
ToolName = Literal[
    "get_lead_context",
    "search_knowledge_base",
    "check_calendar",
    "book_demo",
    "write_crm_note",
    "handoff_to_human",
]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConversationRequest(BaseModel):
    lead_id: str
    conversation: List[ConversationTurn]


class TriggerEvent(BaseModel):
    """Conversation trigger implemented by the local harness."""

    event_type: Literal["conversation"] = "conversation"
    lead_id: str
    payload: Dict[str, Any]


class CollectedInfo(BaseModel):
    company_size: Optional[str] = None
    industry: Optional[str] = None
    pain_point: Optional[str] = None
    budget: Optional[str] = None
    budget_amount: Optional[str] = None
    budget_range: Optional[str] = None
    decision_maker: Optional[str] = None
    go_live_time: Optional[str] = None
    email: Optional[str] = None
    invalid_email: Optional[str] = None
    timezone: Optional[str] = None
    demo_purpose: Optional[str] = None
    interest_area: Optional[str] = None
    sales_team_size: Optional[str] = None


class BookingState(BaseModel):
    status: BookingStatus = "NO_DEMO_INTENT"
    email: Optional[str] = None
    timezone: Optional[str] = None
    demo_purpose: Optional[str] = None
    offered_slots: List[Dict[str, Any]] = Field(default_factory=list)
    selected_slot_id: Optional[str] = None
    selected_slot_confirmed: bool = False
    rejected_slot_ids: List[str] = Field(default_factory=list)
    slot_selection_status: SlotSelectionStatus = "none"
    book_demo_success: bool = False
    calendar_event_id: Optional[str] = None


class CRMState(BaseModel):
    status: CRMStatus = "NOT_WRITTEN"
    last_summary: Optional[str] = None
    last_qualification_level: Optional[QualificationLevel] = None
    last_next_action: Optional[str] = None
    last_current_stage: Optional[str] = None
    last_customer_pain: Optional[str] = None
    last_note_id: Optional[str] = None


class CRMNote(BaseModel):
    note_id: str
    lead_id: str
    summary: str
    pain_point: str
    customer_pain: str
    current_stage: str
    qualification_level: QualificationLevel
    next_action: str
    stage: Literal[
        "discovery",
        "qualification",
        "demo_pending",
        "demo_booked",
        "handoff_required",
        "crm_sync_pending",
        "unknown",
    ]
    source_facts: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


class EvidencePolicy(BaseModel):
    price_policy: Literal["unknown", "no_public_price", "exact_price_supported"] = "unknown"
    metric_policy: Literal["unknown", "no_guaranteed_metric", "exact_metric_supported"] = "unknown"
    customer_case_policy: Literal["unknown", "no_public_customer_name", "public_case_supported"] = "unknown"
    feature_policy: Literal["unknown", "feature_supported", "needs_confirmation"] = "unknown"
    evidence_ids: List[str] = Field(default_factory=list)
    supported_facets: Dict[str, List[str]] = Field(default_factory=dict)
    unsupported_facets: Dict[str, List[str]] = Field(default_factory=dict)
    evidence_spans: Dict[str, str] = Field(default_factory=dict)


class KBDocument(BaseModel):
    id: str
    topic: str
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    policy: Dict[str, Any] = Field(default_factory=dict)
    supported_facets: List[str] = Field(default_factory=list)
    unsupported_facets: List[str] = Field(default_factory=list)
    visibility: Literal["public", "internal", "restricted"] = "public"
    confidence: Literal["high", "medium", "low"] = "high"
    updated_at: str


class Claim(BaseModel):
    text: str
    claim_type: ClaimType
    facet: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_span: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "low"


class GroundedResponse(BaseModel):
    message: str
    claims: List[Claim] = Field(default_factory=list)


class RequestedFacet(BaseModel):
    facet: FacetName
    phrase: str


class OutboxEvent(BaseModel):
    event_id: str
    event_type: Literal["WRITE_CRM_AFTER_BOOKING", "WRITE_CRM_AFTER_HANDOFF"]
    lead_id: str
    idempotency_key: str
    calendar_event_id: Optional[str] = None
    payload: Dict[str, Any]
    status: Literal["pending", "done", "failed"]
    retry_count: int = 0
    last_error: Optional[str] = None


class TrajectoryEvent(BaseModel):
    step: str
    status: Literal["success", "failed", "skipped"] = "success"
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


class ToolCall(BaseModel):
    tool_name: ToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: ToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentState(BaseModel):
    lead_id: str
    lead_context: Dict[str, Any] = Field(default_factory=dict)
    context_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    qualification_level: QualificationLevel = "unknown"
    collected_info: CollectedInfo = Field(default_factory=CollectedInfo)
    missing_info: List[str] = Field(default_factory=list)
    next_action: str = "澄清客户当前需求"
    risk_flags: List[str] = Field(default_factory=list)
    booking: BookingState = Field(default_factory=BookingState)
    crm: CRMState = Field(default_factory=CRMState)
    trajectory: List[TrajectoryEvent] = Field(default_factory=list)
    allowed_actions: List[str] = Field(default_factory=list)
    last_kb_evidence_ids: List[str] = Field(default_factory=list)
    last_kb_evidence_topics: List[str] = Field(default_factory=list)
    last_kb_evidence_coverage: Dict[str, bool] = Field(default_factory=dict)
    last_kb_query: Optional[str] = None
    evidence_policy: EvidencePolicy = Field(default_factory=EvidencePolicy)
    requested_facets: List[RequestedFacet] = Field(default_factory=list)
    last_claims: List[Claim] = Field(default_factory=list)
    grounded_claims: List[Claim] = Field(default_factory=list)
    outbox: List[OutboxEvent] = Field(default_factory=list)
    processed_turn_count: int = 0
    last_processed_user_message: Optional[str] = None

    def add_trajectory(
        self,
        step: str,
        status: Literal["success", "failed", "skipped"] = "success",
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.trajectory.append(
            TrajectoryEvent(
                step=step,
                status=status,
                tool_name=tool_name,
                arguments=arguments or {},
                data=data or {},
                error=error,
            )
        )


class AgentResponse(BaseModel):
    assistant_message: str
    tool_calls: List[Union[ToolResult, ToolCall]] = Field(default_factory=list)
    state: "PublicAgentState"
    run_id: str
    trajectory: Optional[List[TrajectoryEvent]] = None
    debug_state: Optional[AgentState] = None


class PublicAgentState(BaseModel):
    qualification_level: QualificationLevel
    missing_info: List[str] = Field(default_factory=list)
    next_action: str
    risk_flags: List[str] = Field(default_factory=list)

    @classmethod
    def from_agent_state(cls, state: AgentState) -> "PublicAgentState":
        return cls(
            qualification_level=state.qualification_level,
            missing_info=list(state.missing_info),
            next_action=state.next_action,
            risk_flags=list(state.risk_flags),
        )


class ExtractionResult(BaseModel):
    company_size: Optional[str] = None
    industry: Optional[str] = None
    pain_point: Optional[str] = None
    budget: Optional[str] = None
    budget_amount: Optional[str] = None
    budget_range: Optional[str] = None
    decision_maker: Optional[str] = None
    go_live_time: Optional[str] = None
    email: Optional[str] = None
    invalid_email: Optional[str] = None
    timezone: Optional[str] = None
    demo_purpose: Optional[str] = None
    interest_area: Optional[str] = None
    sales_team_size: Optional[str] = None
    demo_intent: bool = False
    demo_declined: bool = False
    booking_cancelled: bool = False
    demo_cancelled: bool = False
    price_question: bool = False
    metric_question: bool = False
    customer_case_question: bool = False
    feature_question: bool = False
    technical_or_security_question: bool = False
    prompt_injection: bool = False
    ambiguous_timezone: bool = False
    risk_flags: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    ok: bool
    errors: List[str] = Field(default_factory=list)

    @classmethod
    def pass_(cls) -> "ValidationResult":
        return cls(ok=True, errors=[])

    @classmethod
    def fail(cls, *errors: str) -> "ValidationResult":
        return cls(ok=False, errors=list(errors))


class ActionPlan(BaseModel):
    intent: str
    assistant_message_draft: str = ""
    tool_calls: List[ToolCall] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    case_id: str
    lead_id: str = "L123"
    conversation: List[ConversationTurn]
    multi_turn: bool = False
    must_call: List[str] = Field(default_factory=list)
    forbidden_call: List[str] = Field(default_factory=list)
    assertions: List[str] = Field(default_factory=list)
    mock_overrides: Dict[str, Any] = Field(default_factory=dict)


class EvalCaseResult(BaseModel):
    case_id: str
    passed: bool
    failures: List[str] = Field(default_factory=list)
    p0_failures: List[str] = Field(default_factory=list)
    output: Dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    total_cases: int
    passed: int
    failed: int
    metrics: Dict[str, float]
    p0_failures: List[Dict[str, Any]] = Field(default_factory=list)
    failed_cases: List[EvalCaseResult] = Field(default_factory=list)
