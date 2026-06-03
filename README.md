# P-12 Sales Operations Action Harness

Minimal runnable harness for a B2B SDR workflow. It accepts conversation JSON, keeps lead/session state, proposes tool actions, validates guardrails, executes local mock tools, and returns a safe assistant message plus public state.

Scope: this is a local written-exercise harness with mock tools, a local mock LLM planner, and an in-memory store. The assignment explicitly allows a local mock LLM, so the lack of a real LLM service is an intentional offline boundary rather than a correctness gap. The project demonstrates workflow boundaries and regression coverage; it is not connected to external CRM, calendar, KB, queue, or production LLM services.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py run --input sales_agent_harness/examples/input_b.json
python3 main.py run --input sales_agent_harness/examples/input_booking.json --debug
python3 -m sales_agent_harness.run_cli run --input sales_agent_harness/examples/input_b.json
python3 run_cli.py run --input sales_agent_harness/examples/input_b.json
```

`main.py` owns the CLI implementation; `sales_agent_harness.run_cli` and root `run_cli.py` are compatibility wrappers.

Planner choices:

- default `harness_v2`: deterministic `RuleBasedPlanner`
- default `prompt_v1|prompt_v2`: local `MockLLMPlanner`
- `--planner rule`: deterministic rule planner
- `--planner mock_llm` or `--planner auto`: local mock LLM path

No `llm` CLI choice is exposed until a real external adapter is implemented and passes the same contract/eval gates.

## Eval

```bash
python3 main.py eval
python3 main.py eval --planner rule
python3 -m sales_agent_harness.eval_runner --cases sales_agent_harness/eval_cases.yaml
python3 -m pytest -q
```

The eval suite covers pricing hallucination, metric guarantees, customer cases, feature grounding, legal/security handoff, booking preconditions, invalid/corrected email, timezone ambiguity, CRM write success/failure, cross-turn outbox/idempotency, prompt injection, lead-history conflict, KB no-result/restricted cases, and high-value custom quote handoff.

## Compare Prompts

```bash
python3 main.py compare-prompts
python3 main.py compare-prompts --strict
```

This command is a local prompt-contract regression report, not a fair online A/B test. `baseline_prompt` is a deliberately minimal sanity baseline, not evidence of real historical prompt iteration. It emits one schema:

- `baseline_prompt`
- `improved_prompt`
- `rule_planner`
- `summary`

`--strict` exits non-zero only when the deterministic rule reference fails.

## Mock LLM Boundary

`MockLLMPlanner` is a local model double for this written exercise. It is useful for prompt-contract regression, schema handling, and comparing weak/strong prompt behavior, but it is not evidence that an external model provider will behave identically.

Before enabling a real LLM adapter, validate these gates:

- Schema repair: malformed or partial model JSON must be repaired once, then fall back to the deterministic planner if still invalid.
- Prompt injection: model output must remain constrained by `allowed_actions`; validators must block forbidden tools even if the model proposes them.
- Evidence grounding: high-risk claims must use evidence ids from the current KB call; restricted or mismatched evidence must trigger fallback copy.
- Eval parity: run the same eval suite plus hidden adversarial cases against the real model path before exposing `--planner llm`.
- Observability: record prompt version, model metadata, repair attempts, tool proposal, validator errors, and fallback reason in trajectory.

## Grounding

Requested-facet detection is a small keyword table in `sales_agent_harness/grounding.py`. It avoids a long `if` chain without adding a config loader or registry.

## Core Semantics

- `TriggerEvent` currently supports only `event_type="conversation"`. Other triggers should be added only when their payload shape and handlers are known.
- `allowed_actions` is the hard tool permission boundary.
- `RuleBasedPlanner` is the source of truth for deterministic business decisions. `prompt_v2.txt` describes planning boundaries but does not own booking or CRM state-machine rules.
- `book_demo.success=true` is the only source of booking truth.
- `write_crm_note` cannot record `Demo booked` unless booking succeeded.
- `InMemoryStore` is a mutable service object for local sessions, bookings, CRM notes, handoffs, call counts, and outbox events.
- Outbox updates use explicit ID namespaces: `event_id=` or `calendar_event_id=`.
- KB search returns structured documents, evidence ids, topics, visibility, and policy metadata.
- The latest KB lookup stores evidence ids, topics, and a simple claim-type coverage dict.
- Product and technical answers are grounded by requested facets and evidence ids; broad catch-all product facets are avoided.
- LLM/mock planners may propose intent, facets, and action candidates. Deterministic planners, validators, and state machines enforce booking, handoff, CRM, and evidence rules.

## Output Contract

Default output shows public tool calls and hides tool result payloads/internal state:

```json
{
  "assistant_message": "...",
  "tool_calls": [
    {
      "tool_name": "write_crm_note",
      "arguments": {
        "lead_id": "L123",
        "summary": "...",
        "qualification_level": "medium",
        "next_action": "..."
      }
    }
  ],
  "state": {
    "qualification_level": "medium",
    "missing_info": ["budget", "decision_maker"],
    "next_action": "继续确认预算和决策人",
    "risk_flags": []
  },
  "run_id": "run_..."
}
```

Default `tool_calls` matches the written-exercise contract and includes only public call shape: `tool_name` and `arguments`. Use `--debug` for executed tool results (`success`, `data`, `error`), trajectory, and full `debug_state`. Use `--include-trajectory` for trajectory without full internal state.

## Documents

- 第一部分答卷: `docs/part1_answer.md`
- AI 协作日志: `docs/ai_collaboration_log.md`
- MVP scope/refactor notes: `docs/mvp_scope_and_refactor_notes.md`
