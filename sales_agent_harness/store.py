from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import AgentState, OutboxEvent, ToolResult


@dataclass
class InMemoryStore:
    """Session-aware in-memory state for the local harness MVP."""

    crm_notes: List[Dict[str, Any]] = field(default_factory=list)
    handoff_log: List[Dict[str, Any]] = field(default_factory=list)
    bookings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    booked_keys: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outbox: List[OutboxEvent] = field(default_factory=list)
    call_counts: Dict[str, int] = field(default_factory=dict)
    mock_overrides: Dict[str, Any] = field(default_factory=dict)
    sessions: Dict[str, AgentState] = field(default_factory=dict)
    lead_context_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def reset(self, mock_overrides: Optional[Dict[str, Any]] = None) -> None:
        self.crm_notes.clear()
        self.handoff_log.clear()
        self.bookings.clear()
        self.booked_keys.clear()
        self.outbox.clear()
        self.call_counts.clear()
        self.mock_overrides.clear()
        self.sessions.clear()
        self.lead_context_cache.clear()
        if mock_overrides:
            self.mock_overrides.update(mock_overrides)

    def reset_runtime_logs_only(self, mock_overrides: Optional[Dict[str, Any]] = None) -> None:
        self.call_counts.clear()
        self.mock_overrides.clear()
        if mock_overrides:
            self.mock_overrides.update(mock_overrides)

    def get_session(self, session_id: str) -> Optional[AgentState]:
        stored = self.sessions.get(session_id)
        if stored is None:
            return None
        state = stored.model_copy(deep=True)
        state.trajectory = []
        return state

    def save_session(self, session_id: str, state: AgentState) -> None:
        snapshot = state.model_copy(deep=True)
        snapshot.trajectory = []
        self.sessions[session_id] = snapshot

    def add_outbox_event(self, event: OutboxEvent) -> OutboxEvent:
        existing = self._find_outbox_event(lead_id=event.lead_id, idempotency_key=event.idempotency_key)
        if existing is not None:
            return existing
        self.outbox.append(event)
        return event

    def mark_outbox_done(self, *, event_id: Optional[str] = None, calendar_event_id: Optional[str] = None) -> None:
        event = self._find_outbox_by_explicit_id(event_id=event_id, calendar_event_id=calendar_event_id)
        if event is not None and event.status == "pending":
            event.status = "done"

    def mark_outbox_failed(self, *, event_id: Optional[str] = None, calendar_event_id: Optional[str] = None, error: Optional[str]) -> None:
        event = self._find_outbox_by_explicit_id(event_id=event_id, calendar_event_id=calendar_event_id)
        if event is not None:
            event.status = "failed"
            event.retry_count += 1
            event.last_error = error

    def mark_outbox_error(self, *, event_id: Optional[str] = None, calendar_event_id: Optional[str] = None, error: Optional[str]) -> None:
        event = self._find_outbox_by_explicit_id(event_id=event_id, calendar_event_id=calendar_event_id)
        if event is not None and event.status == "pending":
            event.retry_count += 1
            event.last_error = error

    def list_pending_outbox(self, lead_id: Optional[str] = None) -> List[OutboxEvent]:
        return [
            event.model_copy(deep=True)
            for event in self.outbox
            if event.status == "pending" and (lead_id is None or event.lead_id == lead_id)
        ]

    def list_outbox(self, lead_id: Optional[str] = None) -> List[OutboxEvent]:
        return [
            event.model_copy(deep=True)
            for event in self.outbox
            if lead_id is None or event.lead_id == lead_id
        ]

    def retry_pending_outbox(self, write_crm_note: Callable[[Dict[str, Any]], ToolResult]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for event in list(self.outbox):
            if event.status != "pending":
                continue
            result = write_crm_note(dict(event.payload))
            results.append(result)
            if result.success:
                self.mark_outbox_done(event_id=event.event_id)
            else:
                self.mark_outbox_error(event_id=event.event_id, error=result.error)
        return results

    def _find_outbox_event(self, lead_id: str, idempotency_key: str) -> Optional[OutboxEvent]:
        for event in self.outbox:
            if event.lead_id == lead_id and event.idempotency_key == idempotency_key:
                return event
        return None

    def _find_outbox_by_explicit_id(
        self,
        *,
        event_id: Optional[str] = None,
        calendar_event_id: Optional[str] = None,
    ) -> Optional[OutboxEvent]:
        if (event_id is None) == (calendar_event_id is None):
            raise ValueError("exactly one of event_id or calendar_event_id is required")
        if event_id is not None:
            matches = [event for event in self.outbox if event.event_id == event_id]
        else:
            matches = [event for event in self.outbox if event.calendar_event_id == calendar_event_id]
        if len(matches) > 1:
            raise ValueError("ambiguous outbox identifier matched multiple events")
        return matches[0] if matches else None
