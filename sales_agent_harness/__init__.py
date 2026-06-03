"""Minimal Sales Operations Action Harness.

Conversation input drives a workflow that validates state, tool permissions,
business facts, CRM consistency, and evaluation outcomes.
"""

from .harness import SalesAgentHarness
from .models import AgentResponse, ConversationRequest, TriggerEvent
from .store import InMemoryStore

__all__ = ["SalesAgentHarness", "ConversationRequest", "AgentResponse", "TriggerEvent", "InMemoryStore"]
