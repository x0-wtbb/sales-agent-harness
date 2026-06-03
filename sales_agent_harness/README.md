# Sales Agent Harness Package

Package implementation for the local B2B Sales Operations Action Harness.

Run:

```bash
python3 -m sales_agent_harness.run_cli run --input sales_agent_harness/examples/input_b.json
python3 -m sales_agent_harness.eval_runner --cases sales_agent_harness/eval_cases.yaml
```

Key boundaries:

- `TriggerEvent` supports `conversation` only.
- `RuleBasedPlanner` owns deterministic business rules.
- `prompt_v1` and `prompt_v2` run through local `MockLLMPlanner`.
- Facet keywords live in a small table in `grounding.py`.
- Mock tool implementations live in `tools.py`.
- `InMemoryStore` is a mutable local service object.
- KB evidence from the latest lookup uses simple ids/topics/coverage fields on `AgentState`.

See the repository README for CLI, eval, and output details.
