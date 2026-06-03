from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from sales_agent_harness.eval_runner import run_eval
from sales_agent_harness.harness import SalesAgentHarness
from sales_agent_harness.models import ConversationRequest

SUPPORTED_PLANNER_CHOICES = ["auto", "mock_llm", "rule"]


def strip_none_optional_fields(output: dict) -> dict:
    for optional_key in ["debug_state", "trajectory"]:
        if output.get(optional_key) is None:
            output.pop(optional_key, None)
    return output


def response_payload(response: Any) -> dict:
    return strip_none_optional_fields(response.model_dump(mode="json", warnings=False))


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Path to conversation JSON.")
    parser.add_argument("--agent-version", default="harness_v2", help="harness_v2, prompt_v2, or prompt_v1.")
    parser.add_argument(
        "--planner",
        choices=SUPPORTED_PLANNER_CHOICES,
        default=None,
        help="Planner backend. Default uses rule for harness_v2 and mock_llm for prompt_v1/prompt_v2.",
    )
    parser.add_argument("--debug", action="store_true", help="Include full debug_state in JSON output.")
    parser.add_argument("--include-trajectory", action="store_true", help="Include audit trajectory without full debug_state.")


def run_conversation(args: argparse.Namespace) -> dict:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    request = ConversationRequest(**payload)
    response = SalesAgentHarness(
        agent_version=args.agent_version,
        planner_mode=args.planner,
        debug=args.debug,
        include_trajectory=args.include_trajectory,
    ).handle_conversation(request)
    return response_payload(response)


def compare_prompts(cases: str, planner: Optional[str]) -> dict:
    v1 = run_eval(cases, "prompt_v1", planner)
    v2 = run_eval(cases, "prompt_v2", planner)
    rule = run_eval(cases, "harness_v2", "rule")
    v1_p0 = [item["case_id"] for item in v1.p0_failures]
    v2_p0 = [item["case_id"] for item in v2.p0_failures]
    rule_p0 = [item["case_id"] for item in rule.p0_failures]
    return {
        "baseline_prompt": {
            "total_cases": v1.total_cases,
            "passed": v1.passed,
            "failed": v1.failed,
            "p0_failures": v1_p0,
        },
        "improved_prompt": {
            "total_cases": v2.total_cases,
            "passed": v2.passed,
            "failed": v2.failed,
            "p0_failures": v2_p0,
        },
        "rule_planner": {
            "total_cases": rule.total_cases,
            "passed": rule.passed,
            "failed": rule.failed,
            "p0_failures": rule_p0,
        },
        "summary": {
            "passed_delta": v2.passed - v1.passed,
            "p0_failure_delta": len(v2_p0) - len(v1_p0),
            "improved_vs_rule_passed_delta": v2.passed - rule.passed,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Sales Agent Harness CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    add_run_arguments(run_parser)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--cases", default="sales_agent_harness/eval_cases.yaml")
    eval_parser.add_argument("--agent-version", default="harness_v2")
    eval_parser.add_argument("--planner", choices=SUPPORTED_PLANNER_CHOICES, default=None)
    eval_parser.add_argument("--allow-failures", action="store_true")

    compare_parser = subparsers.add_parser("compare-prompts")
    compare_parser.add_argument("--cases", default="sales_agent_harness/eval_cases.yaml")
    compare_parser.add_argument("--planner", choices=SUPPORTED_PLANNER_CHOICES, default=None)
    compare_parser.add_argument("--strict", action="store_true", help="Exit 1 when prompt_v2 or the rule-based reference has failures.")
    args = parser.parse_args(argv)

    if args.command == "run":
        print(json.dumps(run_conversation(args), ensure_ascii=False, indent=2))
        return
    if args.command == "eval":
        report = run_eval(args.cases, args.agent_version, args.planner)
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        if report.failed and not args.allow_failures:
            sys.exit(1)
        return
    if args.command == "compare-prompts":
        payload = compare_prompts(args.cases, args.planner)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.strict and payload["rule_planner"]["failed"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
