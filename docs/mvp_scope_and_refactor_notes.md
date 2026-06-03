# MVP Scope And Refactor Notes

## This MVP Is / Is Not

This project is a local MVP harness that runs a rule/mock-LLM sales-agent planning loop with mock tools, structured output, and regression evals. A local mock LLM is within the written-exercise requirements; external LLM integration is a production-readiness question, not a current requirement miss.

Implemented:

- `conversation` trigger through `TriggerEvent(event_type="conversation")`
- deterministic `RuleBasedPlanner`
- local `MockLLMPlanner` for prompt-contract regression
- mock KB, CRM, calendar, handoff, and in-memory runtime store
- structured JSON output, debug trajectory, unit tests, and eval runner

Not implemented:

- real CRM or calendar integration
- production queue or retry worker
- persistent database
- email/form/CRM/calendar triggers with real payload contracts

## Removed / Simplified Items

- Public trigger support was narrowed to `conversation`; unimplemented trigger names are no longer in the public `Literal`.
- CLI business logic lives in root `main.py`; `sales_agent_harness.run_cli` is now a thin compatibility wrapper.
- Mock tool logic lives in `sales_agent_harness/tools.py`.
- `--planner llm` was removed from CLI choices until a real adapter exists.
- `compare-prompts` now emits one stable schema: `baseline_prompt`, `improved_prompt`, `rule_planner`, and `summary`.
- The broad product catch-all facet was removed from default extraction. Unknown broad product wording returns no facet instead of pretending to be recognized.
- Silent facet parsing failures now log warnings and catch `ValidationError` instead of swallowing all exceptions.
- Outbox updates require an explicit ID namespace: `event_id=` or `calendar_event_id=`.
- `InMemoryStore` is a dataclass service object, not a Pydantic data model.

## Facet Extraction Design

Facet keywords are a small `FACET_PATTERNS` table in `sales_agent_harness/grounding.py`. This removes the old long `if` chain while avoiding a YAML loader, registry, or inspector CLI. `extract_requested_facets(text)` loops over the table and emits `RequestedFacet` objects for matched terms.

## State Design

The latest KB lookup keeps evidence ids, topics, and `last_kb_evidence_coverage: dict[str, bool]`. This replaces separate `last_kb_evidence_for_price`, `last_kb_evidence_for_metric`, and `last_kb_evidence_for_customer_case` fields without introducing nested evidence state classes.

`AgentState` stays as the main state object. The project avoids a broad state split because this is a written-exercise MVP.

## Prompt Trust Boundary

The mock/LLM planner may propose intent, requested facets, and candidate tool calls. It does not own hard business correctness.

Deterministic code owns:

- booking preconditions and slot confirmation
- CRM write consistency
- handoff order and fallbacks
- evidence authorization and postcondition validation
- final state transitions

Prompt text may describe these boundaries, but booking, handoff, CRM, and grounding rules cannot rely on prompt compliance alone.

## Real LLM Readiness

The current mock LLM path is acceptable for this assignment. If a real LLM adapter is added later, it should be treated as a new integration surface with its own gates:

- Parse model output into `ActionPlan`; on invalid JSON or schema mismatch, attempt one repair and then fall back to the deterministic planner.
- Keep `allowed_actions`, preconditions, postconditions, booking truth, CRM consistency, and evidence validation outside the model.
- Run existing evals and hidden adversarial cases against the real model path before exposing `--planner llm`.
- Log model/provider metadata, prompt hash, repair attempts, proposed tool calls, validator errors, and fallback reasons in trajectory.
