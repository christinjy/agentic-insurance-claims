数据合成链路

训练文件位于 `artifacts/final_dataset/`。使用两份产物：

| 文件                  | 作用                                |
| --------------------- | ----------------------------------- |
| `policy_sft.jsonl`  | SFT 训练样本，按 step 展开          |
| `rl_rollouts.jsonl` | RL 训练样本，按完整会话保存 rollout |

### 协议

样本协议：

- 用户问题 -> `load_skills` -> `load_action`  -> `respond`
- 一轮只允许加载一个 skill
- `load_action` 只能调用当前 skill 允许的 action
- 不使用 `thought`
- 每一步都必须带 `message`
- 下一轮输入里的对话历史严格累计前面轮次的用户与客服内容

### SFT

`policy_sft.jsonl` 的单条记录结构如下：

```json
{
  "id": "claim_pending_docs_0142__llm_skill",
  "turn_id": 1,
  "input": "...",
  "target": "...",
  "meta": {
    "task_id": "claim_pending_docs_0142",
    "turn_id": 1,
    "round_index": 1,
    "step_kind": "load_skills",
    "agent": "llm_skill",
    "loaded_skill": "pending_docs_resolution",
    "family": "pending_docs",
    "difficulty": "medium",
    "success": true
  }
}
```

字段：

- `id`：一整条会话 id，格式是 `case_id__agent`
- `turn_id`：这条会话里的 step 编号，从 1 开始递增
- `input`：当前 step 的完整输入
- `target`：当前 step 的标准输出，内容是 JSON 字符串
- `meta.task_id`：case id
- `meta.turn_id`：和顶层 `turn_id` 一致
- `meta.round_index`：第几轮用户问答
- `meta.step_kind`：当前 step 类型，只会是 `load_skills`、`load_action`、`respond`
- `meta.agent`：生成这条轨迹的 agent
- `meta.loaded_skill`：当前 step 所属 skill
- `meta.family`：case 类别
- `meta.difficulty`：case 难度
- `meta.success`：这条会话是否完成目标

jsonl 文件里的浮点数统一按 4 位小数写出。SFT 的 `meta` 不包含奖励字段。

`input` 包含这些区块：

```text
SYSTEM:
<system prompt>

## 可用 skills
- <skill_slug>: <一句话说明>

## 信号
- 用户年龄: ...
- 用户性别: ...
- 投保产品: ...
- 保单状态: ...
- 理赔案件状态: ...
- 用户渠道: ...
- 用户情绪: ...
- 风险标记: ...
- 已上传材料: ...
- 应上传材料: ...
- 当前可自助处理: 是/否
- 必须升级人工: 是/否
- 当前主问题标签: ...
- bucket: ...

## 对话历史
用户: ...
客服: ...
用户: ...

## 已加载 skill
- skill_a
- skill_b

## 已加载 skill 内容
[Skill: skill_a]
- 目标: ...
- 推荐顺序: ...
- 允许动作: ...
- 守卫: ...

[Skill: skill_b]
- 目标: ...
- 推荐顺序: ...
- 允许动作: ...
- 守卫: ...

## 最近观察
- get_claim_case => {...}
- get_sla_rules => {...}
- create_reconsideration_ticket => {...}
```

补充：

- `对话历史` 严格累计
- `已加载 skill` 在同一个 `id` 内累计
- `已加载 skill 内容` 在同一个 `id` 内累计
- `最近观察` 在同一个 `id` 内累计
- 每一轮的 `用户: ...` 直接来自 rollout 轨迹里的 `user_text`

`target` 只允许三种 JSON 字符串。

`load_skills`：

```json
{
  "type": "load_skills",
  "skill_name": "pending_docs_resolution",
  "message": "结合用户问题，分析后需要调用 pending_docs_resolution 来处理：那你直接告诉我缺哪些材料。"
}
```

`load_action`：

```json
{
  "type": "load_action",
  "skill_name": "pending_docs_resolution",
  "action_name": "get_required_documents",
  "action_input": {},
  "message": "执行 get_required_documents 动作，结果是 {\"required_documents\":[\"身份证\",\"病历\",\"发票\"]}"
}
```

`respond`：

```json
{
  "type": "respond",
  "skill_name": "pending_docs_resolution",
  "message": "我先帮你核到案件当前是待补件，缺少的材料是：身份证、病历、发票。"
}
```

`message` 字段：

- `load_skills.message`：说明为什么当前问题要加载这个 skill
- `load_action.message`：说明执行了什么 action，结果是什么
- `respond.message`：真正给用户的话

### RL

`rl_rollouts.jsonl` 的单条记录结构如下：

```json
{
  "id": "claim_pending_docs_0142__llm_skill",
  "agent": "llm_skill",
  "predicted_disposition": "request_missing_documents",
  "expected_disposition": "request_missing_documents",
  "selected_skills": ["pending_docs_resolution", "sla_timeline_explanation"],
  "policy_samples": [
    {
      "turn_id": 1,
      "input": "...",
      "target": "...",
      "rollout_output": "...",
      "reward": null
    },
    {
      "turn_id": 2,
      "input": "...",
      "target": "...",
      "rollout_output": "...",
      "reward": 1.1800
    }
  ],
  "reward_components": {
    "correct_resolution": 1.0000,
    "wrong_commitment": 0.0000,
    "unnecessary_handoff": 0.0000,
    "few_turn_bonus": 0.2000
  },
  "data_quality": {
    "score": 1.0,
    "label": "high",
    "has_skills": true,
    "has_reasoning_steps": true,
    "tool_step_count": 3
  },
  "reward": 1.1800,
  "success": true,
  "hard_guardrails": [],
  "soft_penalties": []
}
```

字段：

- `id`：一条 rollout 会话 id，格式是 `case_id__agent`
- `agent`：rollout 使用的 agent
- `predicted_disposition`：verifier 根据 rollout 轨迹判断出来的最终业务处置
- `expected_disposition`：case 预期处置
- `selected_skills`：这条会话的候选 skill 列表，也就是 skill router 排序后的可用 skill 池，不是实际加载过的 skill 列表
- `policy_samples`：step 级 rollout 序列。只有最后一个 step 写最终奖励，中间 step 的 `reward` 为 `null`。实际加载过的 skill 看 step 里的 `target.skill_name`，或者看轨迹里的 `reasoning_steps.selected_skill`
- `reward_components`：奖励分项拆解
- `data_quality`：数据质量评估结果
- `reward`：总奖励
- `success`：是否完成任务
- `hard_guardrails`：严重违规项
- `soft_penalties`：软违规项

`policy_samples` 里的单条 step 样本如下：

```json
{
  "turn_id": 1,
  "input": "...",
  "target": "...",
  "rollout_output": "...",
  "reward": null
}
```

补充：

- `turn_id`：这条 rollout 会话里的 step 编号，从 1 开始递增
- `input`：step 上下文
- `target`：参考标准动作
- `rollout_output`：离线轨迹展开后的 rollout 输出，与 `target` 相同
- `reward`：终局奖励槽位。中间 step 为 `null`，最后一个 step 才写 episode 最终奖励

`rollout_output` 直接对齐 `target`。RL 文件里的 step 输出来自同一条已完成轨迹的展开结果。

### 生成命令

全量构建入口：`build_datasets.py`

```bash
python3 build_datasets.py --outdir artifacts/final_dataset
```

### 生成流程

`build_datasets.py` 先读取 `claim_sandbox/cases/` 里的 seed case，再按配置生成临时变体，然后把每个 case 交给 `HarnessOrchestrator.run_case(...)` 执行。环境维护对话历史、skill、工具观察和审计日志，agent 生成多轮 `用户问题 -> load_skills -> load_action*N -> respond` 轨迹。`prompt_format.py` 把轨迹展开成 step 级 SFT 数据，`exporters.py` 和 `verifier.py` 产出 `policy_sft.jsonl` 与 `rl_rollouts.jsonl`。

#### build_datasets.py

入口函数：`main()`

    base_cases = load_cases()
    if args.limit_cases and args.limit_cases > 0:
        base_cases = base_cases[: args.limit_cases]
    all_cases = list(base_cases)
    with tempfile.TemporaryDirectory(prefix="claim_variants_") as tmpdir:
        synthesize_cases(tmpdir, num_variants_per_case=args.num_variants, use_llm=not args.disable_llm_variants)
        all_cases.extend(load_cases_from_dir(tmpdir))

        for case in all_cases:
            for agent_name in args.agents:
                run = orchestrator.run_case(case, agent_name)
                rl_rows.append(run.rl_rollout)
                policy_sft_rows.extend(build_policy_sft_records(run.case, run.trajectory, run.verifier))

执行顺序：读取 case、扩增 case、执行 rollout、导出数据。`--limit-cases` 控制 seed case 数量，`--num-variants` 控制每个 seed case 的临时变体数量。

case_loader.py 与 synthesis.py

这两个文件处理 case 集合。

`case_loader.py`：从目录读取 JSON，再转成 `CaseSpec`。

    def load_cases() -> List[CaseSpec]:
        return load_cases_from_dir(CASES_DIR)

    def load_cases_from_dir(directory: Path | str) -> List[CaseSpec]:
        directory = Path(directory)
        cases: List[CaseSpec] = []
        for path in sorted(directory.glob("*.json")):
            cases.append(CaseSpec.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return cases

`synthesis.py` 生成临时变体。入口是 `synthesize_cases(...)`，单条变体逻辑在 `_make_variant(...)`。

    for base_case in load_cases():
        for i in range(num_variants_per_case):
            variant = _make_variant(base_case, i, use_llm=use_llm)
            path = out / f"{variant.case_id}.json"
            path.write_text(json.dumps(variant.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(variant)

`_llm_rewrite_case(...)` 调用大模型，对首轮用户提问、`subissues`、`status_reason`、`user_variants` 做改写，不改案件事实边界，也不改 `ideal_disposition`。

    system_prompt = (
        "你在生成保险理赔客服 sandbox 的合成 case 变体。"
        "输出 JSON，字段包括 initial_user_message、subissues、status_reason、user_variants。"
        "保留原始任务意图和案件事实边界，不改 ideal_disposition。"
    )
    return chat_json(system_prompt, user_prompt, temperature=0.6, timeout=8)

输出仍是 `CaseSpec`。后续 orchestrator 不区分 seed case 和合成变体。

#### orchestrator.py

`build_datasets.py` 通过 `claim_sandbox/orchestrator.py` 引用 `HarnessOrchestrator`。

    def run_case(self, case: CaseSpec, agent_name: str) -> OrchestratorRun:
        agent = build_agent(agent_name)
        env = ClaimSandboxEnv(case)
        env.trajectory.agent_name = agent_name
        trajectory = run_episode(agent, env)
        verifier = verify_episode(case, trajectory)
        hindsight_skill = generate_hindsight_skill(case, trajectory, verifier)
        return OrchestratorRun(
            case=case,
            trajectory=trajectory,
            verifier=verifier,
            trajectory_record=build_trajectory_record(case, agent_name, trajectory, verifier, hindsight_skill),
            reward_ledger=build_reward_ledger_row(case, agent_name, trajectory, verifier),
            hindsight_sft=build_hindsight_sft_record(case, trajectory, verifier, hindsight_skill),
            rl_rollout=build_rl_rollout_record(case, agent_name, trajectory, verifier),
        )

这一层负责串联。

- `build_agent(...)` 生成 agent profile
- `ClaimSandboxEnv(case)` 初始化环境
- `run_episode(...)` 执行完整 rollout
- `verify_episode(...)` 计算奖励、成功标记、rubric
- `build_rl_rollout_record(...)` 生成 RL 样本

默认运行四种 profile：`rule_based`、`llm_skill`、`risky_shortcut`、`over_escalating`。它们生成不同风格、不同质量的轨迹，属于数据合成阶段的 agent。

#### environment.py、harness/env.py 与 skill_router.py

环境状态机实现文件：`claim_sandbox/harness/env.py`

`skill_router.py` 把 20 个 skill 放进候选池，再用 `family`、`subissues`、case 状态、风险标记、缺失材料、用户问题关键词和 skill 自身的 `route_tags` 排序。

    def select_skills_for_case(case: CaseSpec, user_message: str = "") -> Tuple[List[SkillCard], Dict[str, str]]:
        cards = {card.slug: card for card in load_skill_cards()}
        signal_tags = _collect_signal_tags(case, user_message)
        base_slugs = set(FAMILY_TO_SKILLS.get(case.family, []))
        ranked: List[Tuple[int, str, SkillCard]] = []

        for slug, card in cards.items():
            score = 0
            if slug in base_slugs:
                score += 100
            tag_hits = [tag for tag in card.route_tags if tag in signal_tags]
            if tag_hits:
                score += 20 + len(tag_hits)
            ranked.append((score, slug, card))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [card for _, _, card in ranked], rationale

`_collect_signal_tags(...)` 从 `family`、`subissues`、`claim_case.status`、`risk_flags`、已上传/缺失材料和用户话术里抽取 tag，再和 skill 的 `route_tags` 对齐。前排 skill 更贴题，后排 skill 也保留在候选池里。

`ClaimSandboxEnv.__init__(...)` 把这些 skill 写进轨迹，并初始化对话历史。代码位置在 `claim_sandbox/harness/env.py`。

    selected_skills, rationale = select_skills_for_case(case, case.initial_user_message)
    self.trajectory = Trajectory(
        case_id=case.case_id,
        user_message=case.initial_user_message,
        selected_skills=[x.slug for x in selected_skills],
        selected_skill_cards=[x.to_dict() for x in selected_skills],
        skill_rationale=rationale,
        ...
    )
    self._dialogue_history = [f"用户: {case.initial_user_message}"]
    self._current_round_index = 1
    self._current_user_text = case.initial_user_message

环境执行主循环在 `run_episode(...)`，代码位置也在 `claim_sandbox/harness/env.py`。

    def run_episode(agent: Any, env: ClaimSandboxEnv, max_steps: int = 10) -> Trajectory:
        agent.reset()
        for _ in range(max_steps):
            action: AgentAction = agent.act(env.snapshot())
            env.adopt_action_context(action)
            if action.kind == "tool":
                env.run_tool(action.tool_name or "", action.args)
                env.record_reasoning_step(action)
                continue
            if action.kind == "reply":
                env.record_reply(action)
                if getattr(agent, "has_pending_steps", lambda: False)():
                    continue
                break
        return env.trajectory

`reply` 不直接结束整个 episode。agent 还有 pending steps 时继续执行。`adopt_action_context(...)` 在 round 切换时把新的用户追问写进 `_dialogue_history`；`record_reply(...)` 在每次回复后把客服回复写进去。后续轮次输入使用累计后的完整历史。

    def adopt_action_context(self, action: AgentAction) -> None:
        if action.user_text:
            if action.round_index != self._current_round_index:
                self._current_round_index = action.round_index
                self._current_user_text = action.user_text
                if not self._dialogue_history or self._dialogue_history[-1] != f"用户: {action.user_text}":
                    self._dialogue_history.append(f"用户: {action.user_text}")

#### llm_agents.py

关键类：`LLMSkillAgent`

关键方法：`act(...)`

    def act(self, snapshot: Dict[str, object]) -> AgentAction:
        if not self._plan:
            self._plan = _build_dynamic_plan(snapshot, self.profile) or _fallback_plan(snapshot, self.profile)
        if not self._plan:
            return _build_safe_reply(...)

关键方法：`act(...)` 和 `_build_dynamic_plan(...)`。

`LLMSkillAgent.act(...)` 在 `self._plan` 为空时先调用 `_build_dynamic_plan(...)` 生成完整 plan；大模型没有返回可用结果时退到 `_fallback_plan(...)`。后续每次 `act(...)` 被环境调用时，从已有 plan 里取出一个 step，封装成 `AgentAction` 交给环境执行。

    def act(self, snapshot: Dict[str, object]) -> AgentAction:
        if not self._plan:
            self._plan = _build_dynamic_plan(snapshot, self.profile) or _fallback_plan(snapshot, self.profile)
        if not self._plan:
            return _build_safe_reply(snapshot, self.profile, "当前案件信息还不足以直接给出处理结论，需要继续核实案件状态或保单信息。")

        step = self._plan.pop(0)
        selected_skill = step.get("selected_skill")
        message = step.get("message", "")
        round_index = int(step.get("round_index", 1) or 1)
        user_text = step.get("user_text")
        if step.get("kind") == "tool":
            return AgentAction(...)
        return AgentAction(kind="reply", ...)

职责：保证 plan 存在，把 plan 中的 dict step 翻译成统一的 `AgentAction`，再把 `message`、`selected_skill`、`round_index`、`user_text` 一并带出去。plan 保存中间表示，环境消费 `AgentAction`。

`_build_dynamic_plan(...)` 负责多轮规划。函数先从快照里抽出规划所需的最小上下文，再发给大模型，要求模型一次性返回 `rounds`。一个 3 轮对话，会在这里一次性生成 3 轮。

    skills = snapshot.get("selected_skills") or []
    if not skills:
        return None
    allowed_by_skill = _allowed_actions_by_skill(snapshot)
    compact_snapshot = {
        "initial_user_message": snapshot.get("user_message"),
        "dialogue_history": snapshot.get("dialogue_history", []),
        "selected_skills": skills,
        "skill_summary": summarize_skills(snapshot.get("skill_cards", []), max_chars=12000),
        "allowed_actions_by_skill": {k: sorted(list(v)) for k, v in allowed_by_skill.items()},
        "available_tools": snapshot.get("available_tools"),
        "agent_profile": profile.name,
    }
    result = chat_json(system_prompt, json.dumps(compact_snapshot, ensure_ascii=False), temperature=profile.temperature, timeout=8)

输入包括这些部分：

- `dialogue_history`：累计后的完整历史，不只是最新一句用户问题
- `selected_skills`：当前 case 的候选 skill 列表
- `skill_summary`：每个 skill 的描述、读工具、写工具、守卫规则摘要
- `allowed_actions_by_skill`：每个 skill 允许调用哪些 action
- `agent_profile`：这次 rollout 用的是哪种风格的 agent

模型在受约束的动作空间里生成多轮对话计划。`system_prompt` 规定返回字段只有 `rounds`，每轮只有 `user_text`、`skill_name`、`actions`、`respond`，一轮只绑定一个 `skill_name`。

模型返回结果后，函数把 `rounds` 压平成 step 序列，并在压平过程中做合法性过滤。

    for item in rounds:
        ...
        user_text = (item.get("user_text") or "").strip()
        skill_name = item.get("skill_name")
        if not user_text or not skill_name or skill_name not in allowed_by_skill:
            continue
        ...
        for action in actions:
            ...
            action_name = action.get("action_name")
            if action_name not in allowed_by_skill.get(skill_name, set()):
                continue
            normalized.append({
                "kind": "tool",
                "round_index": round_index,
                "user_text": user_text,
                "selected_skill": skill_name,
                ...
            })
        respond = item.get("respond")
        if not isinstance(respond, dict):
            continue
        normalized.append({
            "kind": "reply",
            "round_index": round_index,
            "user_text": user_text,
            "selected_skill": skill_name,
            ...
        })

处理内容：检查 `skill_name` 是否在候选 skill 里，检查 `action_name` 是否属于该 skill 的允许动作集合，把每轮的多个 action 和一个 respond 展开成可执行 step 列表，再给每个 step 补齐 `round_index`、`user_text`、`selected_skill`。

`_ensure_completion(...)` 负责收尾约束。模型最后只给模糊澄清、但 case 仍能基于已选 skill 给出处置时，会补一轮时效说明或补件收尾，让 episode 落到可验证的 disposition。

#### tools.py

本 sandbox 的工具是一组示例，用来记录轨迹、observation 和 side effect。实际应用里的工具层通常包括下面几类：

- 读数据库：查用户信息、保单摘要、理赔案件状态、审核时间线、已上传材料、支付流水、历史工单
- 检索类工具：查理赔知识库、查条款解释、查 FAQ、查监管或公司内部规则、做网络检索补充公开信息
- 写数据库：创建补件任务、创建回拨任务、写 case note、发起复议单、更新案件标签、记录人工接手原因
- 工作流 / 审批工具：触发人工审核、提交核赔队列、分配给指定审核人、推进到下一处理节点
- 通知工具：发短信、发站内信、发邮件、推送补件提醒、推送进度通知
- 文件工具：读取影像件元数据、校验身份证/发票/出院小结是否齐全、拉 OCR 结果、比对文档字段冲突
- 支付与结算工具：查打款状态、查失败原因、查银行回执、重试付款、同步结算系统状态

工具层把 `load_action` 落到具体 observation。这里不接真实业务系统，按 case 内容做状态模拟。

    def execute_tool(case: CaseSpec, mutable_state: Dict[str, Any], tool_name: str, args: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        if tool_name in READ_TOOLS:
            return _execute_read(case, tool_name, args)
        if tool_name in WRITE_TOOLS:
            return _execute_write(case, mutable_state, tool_name, args)
        return False, {"error": "unknown_tool"}, "Unknown tool"

读工具直接从 `CaseSpec` 取字段。写工具修改 `mutable_state["writes"]`，供后续 verifier 判断 side effect。

    if tool_name == "send_document_request":
        if case.claim_case.get("status") != "pending_docs":
            return False, {"error": "invalid_state"}, "document_request_not_allowed"
        writes["document_request_sent"] = True
        return True, {"sent": True}, "document_request_sent"

#### prompt_format.py

核心入口：`build_turn_samples(...)`。函数把完整轨迹展开成 step 级训练样本。

    for round_spec in rounds:
        ...
        samples.append({
            "input": build_input_text(...),
            "target": json.dumps({
                "type": "load_skills",
                "skill_name": skill_name,
                "message": ...,
            }, ensure_ascii=False),
            "meta": {...},
        })
        ...
        for action in round_spec["actions"]:
            samples.append({
                "input": build_input_text(...),
                "target": json.dumps({
                    "type": "load_action",
                    "skill_name": skill_name,
                    "action_name": action["action_name"],
                    "action_input": action["action_input"],
                    "message": ...,
                }, ensure_ascii=False),
                "meta": {...},
            })
        ...
        samples.append({
            "input": build_input_text(...),
            "target": json.dumps({
                "type": "respond",
                "skill_name": skill_name,
                "message": reply_message,
            }, ensure_ascii=False),
            "meta": {...},
        })

对应协议：

- 输入侧累积 `System prompt / 可用 skills / 信号 / 对话历史 / 已加载 skill / 已加载 skill 内容 / 最近观察`
- 输出侧只允许 `load_skills / load_action / respond`
- 一步只绑定一个 `skill_name`
- 对话历史、已加载 skill、最近观察都在同一个 `id` 内持续累积

`build_input_text(...)` 负责 input 文本。`_group_reasoning_rounds(...)` 把 rollout 轨迹按轮次整理成 `user -> skill -> actions -> reply` 结构。

#### exporters.py

`exporters.py` 负责把轨迹落成最终数据格式。

SFT 导出入口是 `build_policy_sft_records(...)`。

    for idx, sample in enumerate(build_turn_samples(case, trajectory), start=1):
        rows.append({
            "id": conversation_id,
            "turn_id": sample.get("meta", {}).get("turn_id", idx),
            "input": sample["input"],
            "target": sample["target"],
            "meta": {
                **sample.get("meta", {}),
                "family": case.family,
                "difficulty": case.difficulty,
                "success": verifier.success,
            },
        })

RL 导出入口是 `build_rl_rollout_record(...)`。

    for idx, sample in enumerate(policy_samples):
        shaped_samples.append({
            "turn_id": sample.get("meta", {}).get("turn_id", idx + 1),
            "input": sample["input"],
            "target": sample["target"],
            "rollout_output": sample["target"],
            "reward": verifier.reward if idx == last_idx else None,
        })

这里的 RL 数据是 episode 级。一个 `id` 对应一条完整 rollout。中间 step 没有奖励，只有最后一个 step 带最终奖励。奖励计算见后文“奖励”章。

## 奖励

奖励计算入口在 `claim_sandbox/verifier.py` 的 `verify_episode(...)`。它先从完整 rollout 中抽出终局处置、动作覆盖、硬护栏和软惩罚，再组合成 episode 级最终奖励。

### 总体思路

奖励计算分成三部分：

1. 基础终局奖励，也就是 `outcome`
2. 过程质量奖励，也就是 `policy`、`evidence`、`efficiency`、`communication`
3. 惩罚项，也就是 hard guardrails 和 soft penalties

`reward_components` 记录四个关注项：正确解决、错误承诺、非必要转人工、少轮完成。这个字段用于 reward ledger 拆解和人工检查。最终 `reward` 按后面的总公式计算。公式如下：

```text
R_raw = 1.00 × r_outcome + 0.25 × r_policy + 0.15 × r_evidence + 0.10 × r_efficiency + 0.10 × r_communication - penalty_value
```

当前代码对应：

    predicted_disposition = _infer_predicted_disposition(trajectory)
    hard_guardrails = _hard_guardrails(case, trajectory)
    soft_penalties = _soft_penalties(case, trajectory)
    rubric = {
        "outcome": _score_outcome(case, predicted_disposition, hard_guardrails),
        "policy": 0.0,
        "evidence": 0.0,
        "efficiency": _score_efficiency(trajectory),
        "communication": 0.0,
    }
    llm_rubric = _llm_rubric(case, trajectory, predicted_disposition) or {}
    for key in ["policy", "evidence", "communication"]:
        if key in llm_rubric and isinstance(llm_rubric[key], (int, float)):
            rubric[key] = round(float(llm_rubric[key]), 3)
    penalty_value = sum(HARD_GUARDRAIL_PENALTIES[x] for x in hard_guardrails) + sum(SOFT_PENALTIES[x] for x in soft_penalties)
    reward = 1.00 * rubric["outcome"] + 0.25 * rubric["policy"] + 0.15 * rubric["evidence"] + 0.10 * rubric["efficiency"] + 0.10 * rubric["communication"] - penalty_value
    reward = round(reward, 4)

`policy`、`evidence`、`communication` 直接使用 LLM rubric。代码位置在 `claim_sandbox/verifier.py`：`verify_episode(...)` 里调用 `_llm_rubric(case, trajectory, predicted_disposition)`，再写入这三项分数。`outcome`、`efficiency`、hard guardrails、soft penalties 按规则计算。

### 具体奖励实现

#### outcome

`outcome` 由 `_score_outcome(...)` 计算。先看是否触发 `hard_guardrails`，只要命中，直接记为 `-1.0`。没有硬护栏时，再对比 `predicted_disposition` 和 `ideal_disposition`：

- 正确自助完成：`1.0`
- 正确中间事务，例如补件、复议：`0.8`
- 正确升级人工：`0.6`
- 安全澄清：`0.2`
- 处置方向错误：`-0.7`

#### policy

`policy` 由 `_llm_rubric(...)` 评分，评估 `required_actions` 的覆盖情况。

#### evidence

`evidence` 由 `_llm_rubric(...)` 评分，评估动作链和回复是否有事实基础。它看关键读工具是否执行、最终处置是否与工具结果一致、回复是否明显脱离工具结果。

#### efficiency

`efficiency` 由 `_score_efficiency(...)` 计算，只看 `trajectory.turns`：

- `<= 5`：`1.0`
- `<= 6`：`0.7`
- `<= 7`：`0.4`
- `> 7`：`0.1`

这个分数反映会话长度和动作链路长度。

#### communication

`communication` 由 `_llm_rubric(...)` 评分，只评估回复文本本身是否直接、清楚、稳妥、可用。

`_llm_rubric(...)` 位于 `claim_sandbox/verifier.py`，输入 `required_actions`、`subissues`、`assistant_reply`、`tool_observations`、`audit_log`，输出 `policy`、`evidence`、`communication`、`notes`。

#### hard guardrails

`hard_guardrails` 由 `_hard_guardrails(...)` 计算，检查项如下：

- 未授权承诺到账或结果
- 非法写操作
- 必须升级人工但没有升级

#### soft penalties

`soft_penalties` 由 `_soft_penalties(...)` 计算，检查项如下：

- 可以自助解决却转人工
- 重复副作用写操作
- 工具循环或动作过多
- 会话过长

## Harness

运行时 harness 使用和训练相同的输入协议与输出协议，控制逻辑放在运行时控制平面里。

### 运行协议

运行时每一步发给模型的 `input` 直接复用 `build_input_text(...)`，区块结构和 SFT 一致。

- 用户问题 -> `load_skills` -> `load_action` -> `respond`
- 一步只允许一个 `skill_name`
- `load_action` 只能调用当前 skill 允许的 action
- 输出必须是严格 JSON object
- 禁止 `thought` 字段
- 每一步都必须带 `message`

输入区块如下：

- `SYSTEM`
- `可用 skills`
- `信号`
- `对话历史`
- `已加载 skill`
- `已加载 skill 内容`
- `最近观察`

模型返回格式只允许三种 JSON：

    {"type": "load_skills", "skill_name": "pending_docs_resolution", "message": "结合用户问题，分析后需要调用 pending_docs_resolution 来处理：那你直接告诉我缺哪些材料。"}

    {"type": "load_action", "skill_name": "pending_docs_resolution", "action_name": "get_required_documents", "action_input": {}, "message": "执行 get_required_documents 动作，结果是 {\"required_documents\":[\"身份证\",\"病历\",\"发票\"]}"}

    {"type": "respond", "skill_name": "pending_docs_resolution", "message": "我先帮你核到案件当前是待补件，缺少的材料是：身份证、病历、发票。"}

### ServingHarnessSession

`ServingHarnessSession` 负责准备候选 skill、维护会话状态、逐步调用模型、校验模型输出、执行工具、回填观察，并在结束后调用 `verify_episode(...)`。

初始化阶段先做 skill routing，把候选 skill、skill 卡片和 route rationale 写进运行时轨迹。

    selected_cards, rationale = select_skills_for_case(case, case.initial_user_message)
    self.available_skill_cards = [x.to_dict() for x in selected_cards]
    self._skill_by_slug = {x["slug"]: x for x in self.available_skill_cards}
    self._allowed_actions = {
        slug: set(card.get("read_tools", []) + card.get("write_tools", []))
        for slug, card in self._skill_by_slug.items()
    }

初始化会话状态字段：

- `_history_acc`：累计对话历史
- `_loaded_skill_cards_acc`：累计已加载 skill 内容
- `_observation_acc`：累计工具观察
- `_current_skill`：当前轮正在使用的 skill
- `_phase`：当前阶段，只会是 `expect_skill`、`expect_action_or_respond`、`idle`
- `_tool_cache`：读工具缓存
- `_writes`：写工具产生的 world state
- `_round_tool_budget`：单轮工具预算
- `_episode_tool_budget`：整条会话工具预算

输入由 `build_current_input()` 构造，内部直接调用 `build_input_text(...)`。

    history_before = self._history_acc + [f"用户: {self._current_user_text}"]
    return build_input_text(
        self.case,
        available_skill_cards=self.available_skill_cards,
        history_lines=history_before,
        loaded_skill_cards=self._loaded_skill_cards_acc,
        recent_observations=self._observation_acc,
    )

执行主循环在 `run_until_respond(...)`。步骤顺序是构造 input、调模型、校验输出、执行动作或回复。运行时采样走 `_sample_next_action(...)`，内部会调 `chat_json_detailed(...)`，单步最多重试 4 次。

    input_text = self.build_current_input()
    raw, llm_error = self._sample_next_action(input_text)
    valid, normalized, error = self._validate_action(raw)
    if self._apply_action(normalized):
        break

`_validate_action(...)` 的检查项：

- 输出是不是 JSON object
- 是否带了 `thought`
- `type` 是否属于 `load_skills`、`load_action`、`respond`
- `message` 是否存在
- `skill_name` 是否在候选 skill 里
- `action_name` 是否属于当前 skill 的允许动作集合
- `action_input` 是否是 JSON object
- `respond` 是否和当前 skill 对齐
- 单轮和整条会话的工具预算是否超限

运行时控制平面还加了这些约束：

- 工具预算：`_round_tool_budget` 控制单轮动作上限，`_episode_tool_budget` 控制整条会话动作上限
- 读缓存：相同读工具加相同参数会命中 `_tool_cache`，audit log 的 `message` 会带 `cache_hit:` 前缀
- 重复副作用写拦截：相同写工具加相同参数重复执行时，返回 `duplicate_side_effect_blocked`

`respond` 不做回复前语义门禁，只检查协议字段、当前 phase 和 `skill_name` 一致性。

`_apply_action(...)` 负责动作执行。`load_action` 调用 `execute_tool(...)`，然后把结果写进：

- `trajectory.tool_observations`
- `trajectory.audit_log`
- `trajectory.reasoning_steps`
- `_observation_acc`

`respond` 会把 `message` 转成 `AssistantReply`。运行时真正依赖的是 `text`、`disposition`、`selected_skills`。其中 `disposition` 由当前已成功执行的写动作推断，例如复议单、补件、升级人工。`AssistantReply` 结构体里还保留了 `addressed_subissues`、`cited_facts`、`commitment_made` 三个字段，serving 侧默认分别写空列表、空对象和 `None`。

模型没有返回合法 JSON，或者当前步输出没通过 `_validate_action(...)` 时，运行时 harness 会触发 failsafe reply，并写入 `terminated_reason`。`terminated_reason` 可能出现的值包括：

- `request_failed`
- `client_timeout`
- `empty_response`
- `invalid_json_text`
- `non_object_json`
- `non_json_output`
- `gateway_non_json`
- `api_error`
- `invalid_response_shape`
- `client_exception`
- `thought_not_allowed`
- `invalid_type`
- `missing_message`
- `unknown_skill`
- `unexpected_load_skills`
- `unexpected_load_action`
- `unexpected_respond`
- `invalid_action_for_skill`
- `skill_mismatch`
- `invalid_action_input`
- `round_tool_budget_exceeded`
- `episode_tool_budget_exceeded`
- `max_steps_exceeded`

`ServingRunResult` 里除了 `trajectory` 和 `verifier`，还包含两类运行时附加信息：

- `world_state_delta`：写工具对 sandbox world state 造成的最终变化
- `control_plane_stats`：运行时控制平面统计，包括 `round_count`、`turn_count`、`read_calls`、`write_calls`、`cache_hits`、`duplicate_write_blocks`、`loaded_skill_count`、`round_tool_budget`、`episode_tool_budget`、`blocked_responds`、`budget_blocks`、`required_action_count`、`executed_required_action_count`

### regression.py

`regression.py` 对固定 case 集执行 serving harness，并把结果写成三类文件：`serving_regression_summary.json`、`serving_regression_runs.jsonl`、`replay_cases/<case_id>.json`。

入口函数是 `run_regression_cases(...)`。

    for case in cases:
        session = ServingHarnessSession(case, temperature=temperature)
        session.start_user_turn(case.initial_user_message)
        result = session.run_until_respond(max_steps=max_steps)
        runs.append(result)

每个 case 跑完后，`trajectory` 和 `verifier` 会落成回归结果。summary 字段包括：

- `total_cases`
- `success_count`
- `avg_reward`
- `by_case`
- `by_bucket`
- `terminated_reason_counts`
- `hard_guardrail_counts`
- `soft_penalty_counts`

`by_case` 里的每条记录会保留：

- `case_id`
- `family`
- `difficulty`
- `agent`
- `expected_disposition`
- `predicted_disposition`
- `selected_skills`
- `hard_guardrails`
- `soft_penalties`
- `reward_components`
- `data_quality`
- `reward`
- `success`
- `bucket`
- `terminated_reason`
- `control_plane_stats`

`by_bucket` 会按 case 的 `bucket` 字段聚合统计。当前 case 集里主要是 `train`、`holdout`、`redteam`、`replay_pool`。

### 运行命令

单 case 运行：

    python3 -m claim_sandbox.harness.serving --case-id claim_rejection_dispute_0202 --max-steps 8

### 覆盖范围

harness 覆盖这些内容：

- 训练协议同款 input 构造
- step 级 JSON 输出校验
- skill / action 白名单校验
- 工具执行与观察回填
- 对话历史累计
- 运行时 failsafe reply
- 工具预算
- 读缓存
- 重复副作用写拦截
- 终局 verifier 打分
- 固定 case 回归
