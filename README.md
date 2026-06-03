# P-12 Sales Operations Action Harness

代码仓库：[https://github.com/x0-wtbb/sales-agent-harness](https://github.com/x0-wtbb/sales-agent-harness)

这是一个面向 B2B SDR 场景的最小可运行 Sales Agent Harness。它接收对话 JSON，维护线索和会话状态，生成候选工具动作，执行本地 mock 工具，并通过前置/后置校验返回安全的助手回复和公开状态。

本项目是笔试题的本地 MVP 实现：包含 mock tools、本地 mock LLM planner、规则 planner、内存态 store、自动化 eval 和单元测试。项目没有连接真实 CRM、日历、知识库、队列或生产 LLM 服务；这是刻意保留的离线边界，不是运行缺口。

## 安装方式

建议使用 Python 3.10+。在项目根目录运行：

```bash
python3 -m pip install -r requirements.txt
```

主要依赖：

- `pydantic>=2.0`
- `pyyaml>=6.0`
- `requests>=2.31`
- `pytest>=8.0`

## 运行方式

运行示例输入：

```bash
python3 main.py run --input sales_agent_harness/examples/input_b.json
python3 main.py run --input sales_agent_harness/examples/input_booking.json --debug
```

也可以使用兼容入口：

```bash
python3 -m sales_agent_harness.run_cli run --input sales_agent_harness/examples/input_b.json
python3 run_cli.py run --input sales_agent_harness/examples/input_b.json
```

`main.py` 是 CLI 主入口；`sales_agent_harness.run_cli` 和根目录 `run_cli.py` 只是兼容包装。

Planner 选项：

- 默认 `harness_v2`：使用确定性的 `RuleBasedPlanner`
- 默认 `prompt_v1` / `prompt_v2`：使用本地 `MockLLMPlanner`
- `--planner rule`：强制使用规则 planner
- `--planner mock_llm` 或 `--planner auto`：使用本地 mock LLM 路径

当前不暴露 `--planner llm`，因为真实外部 LLM adapter 尚未实现，也没有通过相同的 contract/eval gates。

## 如何运行 Eval

运行自动化 eval：

```bash
python3 main.py eval
python3 main.py eval --planner rule
python3 -m sales_agent_harness.eval_runner --cases sales_agent_harness/eval_cases.yaml
```

运行单元测试：

```bash
python3 -m pytest -q
```

当前 eval suite 覆盖以下风险：

- 价格幻觉和用户锚定价格
- 效果指标保证
- 客户案例和客户名称编造
- 产品能力 grounding
- 法务、安全、采购、定制报价 handoff
- Demo 预约前置条件和 slot 确认
- 无效邮箱、更正邮箱、歧义时区
- CRM 写入成功/失败
- 跨轮 outbox、idempotency 和部分成功补偿
- prompt injection
- lead history 与当前用户信息冲突
- KB no-result / restricted 文档
- 高价值线索和高风险人工接管

## 架构说明

项目刻意实现为 Agent Harness，而不是只写一个销售 prompt。核心原则是：

> LLM 或 mock LLM 只提出候选 intent、facet 和 tool calls；确定性代码负责判断动作是否允许、执行工具、校验回复、推进状态。

主要模块：

- `main.py`：CLI 入口，支持 `run`、`eval`、`compare-prompts`。
- `sales_agent_harness/harness.py`：编排层，负责加载状态、调用 planner、前置校验、执行工具、状态迁移、fallback 和输出整形。
- `sales_agent_harness/policy.py`：规则 planner、mock LLM planner 包装、intent routing 和 `allowed_actions` 计算。
- `sales_agent_harness/state.py`：线索记忆、资格评分、booking 状态机、CRM payload 构造和 evidence policy 更新。
- `sales_agent_harness/validators.py`：动作前置条件和回复后置条件校验，包括工具权限、预约事实、CRM 一致性和 unsupported claim 拦截。
- `sales_agent_harness/tools.py`：本地 mock 工具，包括 lead context、KB search、calendar、book demo、CRM note、handoff 和 outbox。
- `sales_agent_harness/grounding.py`：requested facet 抽取和 evidence 匹配，用于约束价格、指标、客户案例、产品能力、技术/安全声明。
- `sales_agent_harness/eval_runner.py` 和 `sales_agent_harness/eval_cases.yaml`：确定性 eval runner 和回归 case。
- `tests/`：工具、validator、grounding、normalizer、prompt 集成和 CLI 输出测试。

### 核心语义

- `TriggerEvent` 当前只支持 `event_type="conversation"`。
- `allowed_actions` 是硬权限边界，模型不能绕过。
- `RuleBasedPlanner` 是确定性业务决策的参考实现。
- `book_demo.success=true` 是 Demo 已预约的唯一事实来源。
- `write_crm_note` 不能在 booking 未成功时记录 `Demo booked`。
- `InMemoryStore` 管理本地 session、booking、CRM notes、handoff、call counts 和 outbox。
- Outbox 更新必须使用显式 ID 命名空间：`event_id=` 或 `calendar_event_id=`。
- KB search 返回结构化文档、evidence ids、topics、visibility 和 policy metadata。
- 价格、效果指标、客户案例、产品能力、技术/安全声明必须由 evidence ids 支撑。
- Prompt 可以描述规则，但 booking、handoff、CRM 和 grounding 的硬约束必须落在代码里。

## 输出格式

默认输出只暴露公开 tool call 形状，不暴露工具结果 payload 或内部状态：

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

调试参数：

- `--debug`：输出工具执行结果、trajectory 和完整 `debug_state`
- `--include-trajectory`：输出 trajectory，但不输出完整内部状态

## Prompt 对比

```bash
python3 main.py compare-prompts
python3 main.py compare-prompts --strict
```

该命令是本地 prompt-contract regression report，不是线上 A/B test。输出包含：

- `baseline_prompt`
- `improved_prompt`
- `rule_planner`
- `summary`

`--strict` 只在确定性的 rule reference 失败时返回非 0。

## 已知问题

- 没有启用真实生产 LLM adapter；`MockLLMPlanner` 只是本地模型替身，用于 prompt contract regression。
- 没有连接真实 CRM、Calendar、KB、Queue 或 Handoff 服务；所有工具均为本地 mock。
- 状态存储在 `InMemoryStore`，只适合本地运行，不具备跨进程/跨重启持久化能力。
- `TriggerEvent` 当前只支持 `conversation`，没有实现 email、form、CRM event、calendar event 等触发器。
- `compare-prompts` 只是本地对比报告，不代表真实模型提供商表现。
- KB 是小型手写 fixture，真实 KB 接入后需要补充 hidden/adversarial eval。
- 真实 LLM 接入前，需要额外验证 JSON schema repair、prompt injection、evidence grounding、observability 和 hidden eval。

## 文档

- 第一部分答卷：`docs/part1_answer.md`
- AI 协作日志：`docs/ai_collaboration_log.md`
- MVP scope/refactor notes：`docs/mvp_scope_and_refactor_notes.md`
