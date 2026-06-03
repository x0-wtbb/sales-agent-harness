from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sales_agent_harness.harness import SalesAgentHarness
from sales_agent_harness.llm import PromptTemplateLoader
from sales_agent_harness.models import ActionPlan, AgentState, ConversationRequest, ConversationTurn
from sales_agent_harness.policy import MockLLMPlanner


ROOT = Path(__file__).resolve().parents[1]
BANNED_BASELINE_SNIPPETS = ["5 万", "20 万", "50%", "已预约"]


def test_prompt_v1_can_be_loaded_with_hash() -> None:
    prompt_text, prompt_hash = PromptTemplateLoader.load("prompt_v1")

    assert "B2B" in prompt_text
    assert len(prompt_hash) == 64
    int(prompt_hash, 16)


def test_prompt_v2_can_be_loaded_with_hash() -> None:
    prompt_text, prompt_hash = PromptTemplateLoader.load("prompt_v2")

    assert "allowed_actions" in prompt_text
    assert len(prompt_hash) == 64
    int(prompt_hash, 16)


def test_prompt_hash_is_stable() -> None:
    first = PromptTemplateLoader.load("prompt_v2")[1]
    second = PromptTemplateLoader.load("prompt_v2")[1]

    assert first == second


def test_mock_llm_planners_generate_action_plans() -> None:
    state = AgentState(lead_id="L_TEST")
    state.allowed_actions = ["search_knowledge_base", "ask_clarification"]
    conversation = [ConversationTurn(role="user", content="你们标准版一年多少钱？")]

    for version in ["prompt_v1", "prompt_v2"]:
        plan = MockLLMPlanner(version).plan(conversation, state, "pricing_question")

        assert isinstance(plan, ActionPlan)
        assert plan.intent == "pricing_question"
        assert plan.metadata["planner_type"] == "MockLLMPlanner"
        assert plan.metadata["prompt_version"] == version
        assert len(plan.metadata["prompt_hash"]) == 64


def test_prompt_v2_trajectory_contains_prompt_loaded() -> None:
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="我们想了解 AI 销售工具。")],
    )
    response = SalesAgentHarness(agent_version="prompt_v2", include_trajectory=True).handle_conversation(request)

    assert response.trajectory is not None
    prompt_events = [event for event in response.trajectory if event.step == "prompt_loaded"]
    proposal_events = [event for event in response.trajectory if event.step == "llm_proposal"]
    assert prompt_events
    assert proposal_events
    assert prompt_events[0].data["planner_type"] == "MockLLMPlanner"
    assert prompt_events[0].data["prompt_version"] == "prompt_v2"
    assert len(prompt_events[0].data["prompt_hash"]) == 64


def test_compare_prompts_default_shape_and_no_legacy_baseline() -> None:
    result = subprocess.run(
        ["python3", "main.py", "compare-prompts"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"baseline_prompt", "improved_prompt", "rule_planner", "summary"}
    assert "prompt_v1" not in payload
    assert "prompt_v2" not in payload
    assert "harness_v2_rule_based" not in payload
    assert "prompt_v1_mock" not in payload
    assert "prompt_v2_mock" not in payload
    assert "harness_v2_rule" not in payload
    assert "BaselinePlanner" not in result.stdout
    assert "LegacyFailureReplayPlanner" not in result.stdout


def test_main_run_outputs_public_schema() -> None:
    result = subprocess.run(
        ["python3", "main.py", "run", "--input", "sales_agent_harness/examples/input_b.json", "--planner", "rule"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert set(payload) == {"assistant_message", "tool_calls", "state", "run_id"}
    assert payload["tool_calls"]
    assert {"tool_name", "arguments"} == set(payload["tool_calls"][0])
    assert "success" not in payload["tool_calls"][0]
    assert "data" not in payload["tool_calls"][0]
    assert "error" not in payload["tool_calls"][0]
    assert "debug_state" not in payload
    assert "trajectory" not in payload


def test_run_cli_is_thin_wrapper() -> None:
    source = (ROOT / "sales_agent_harness" / "run_cli.py").read_text(encoding="utf-8")

    assert "argparse" not in source
    assert "json.loads" not in source
    assert "from main import main" in source


def test_llm_planner_choice_is_not_advertised_until_supported() -> None:
    result = subprocess.run(
        ["python3", "main.py", "run", "--input", "sales_agent_harness/examples/input_booking.json", "--planner", "llm"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_eval_runner_llm_planner_choice_is_not_advertised_until_supported() -> None:
    result = subprocess.run(
        ["python3", "-m", "sales_agent_harness.eval_runner", "--planner", "llm", "--allow-failures"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_prompt_v1_default_mock_output_has_no_copied_failure_phrases() -> None:
    request = ConversationRequest(
        lead_id="L_TEST",
        conversation=[ConversationTurn(role="user", content="你们标准版一年多少钱？")],
    )
    response = SalesAgentHarness(agent_version="prompt_v1", include_trajectory=True).handle_conversation(request)
    payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)

    for snippet in BANNED_BASELINE_SNIPPETS:
        assert snippet not in response.assistant_message
        assert snippet not in payload
