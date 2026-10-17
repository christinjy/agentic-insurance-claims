from __future__ import annotations

import json
from typing import Any, Dict, List

from .schemas import CaseSpec, Trajectory


SYSTEM_PROMPT = """你是保险理赔客服智能体。
你每次只输出下一步动作，且只能输出一个 JSON object。
除这个 JSON object 外，不允许输出任何别的字符，不允许解释，不允许 Markdown，不允许代码块。

固定流程：用户问题 -> load_skills -> load_action 若干步 -> respond。
下一轮开始时，对话历史必须累计上一轮客服回复和这一轮用户新问题。
每一步只能绑定一个 skill。

只允许三种 type：load_skills、load_action、respond。

输出 schema：
1. load_skills 时只能输出这 3 个字段：type, skill_name, message
   示例：{"type":"load_skills","skill_name":"pending_docs_resolution","message":"结合当前用户问题，需要先加载 pending_docs_resolution。"}
2. load_action 时只能输出这 5 个字段：type, skill_name, action_name, action_input, message
   示例：{"type":"load_action","skill_name":"pending_docs_resolution","action_name":"get_required_documents","action_input":{},"message":"执行 get_required_documents，结果显示缺少出院小结和发票原件。"}
3. respond 时只能输出这 3 个字段：type, skill_name, message
   示例：{"type":"respond","skill_name":"pending_docs_resolution","message":"我先帮你核到案件当前处于待补件状态，缺少出院小结和发票原件。"}

硬性约束：
- 禁止输出 thought 字段
- 禁止输出任何 schema 之外的字段
- message 必须是字符串
- action_input 必须是 JSON object
- action_name 必须来自当前 skill 的允许动作
- 禁止编造条款、赔付金额、到账时间、案件状态、系统处理结果
- 禁止向用户暴露内部字段名、内部工具名

先保证输出格式完全合法，再考虑业务内容。格式不合法视为失败。"""


def build_turn_samples(case: CaseSpec, trajectory: Trajectory) -> List[Dict[str, Any]]:
    rounds = _group_reasoning_rounds(trajectory)
    samples: List[Dict[str, Any]] = []
    history_acc: List[str] = []
    loaded_skill_cards_acc: List[Dict[str, Any]] = []
    loaded_skill_slugs_acc = set()
    observation_acc: List[Dict[str, Any]] = []
    turn_id = 0
    skill_cards = {card["slug"]: card for card in trajectory.selected_skill_cards}

    for round_spec in rounds:
        round_index = round_spec["round_index"]
        user_text = round_spec["user_text"]
        skill_name = round_spec["skill_name"]
        skill_card = skill_cards.get(skill_name)
        if not skill_card:
            continue
        history_before = history_acc + [f"用户: {user_text}"]

        samples.append(
            {
                "input": build_input_text(
                    case,
                    available_skill_cards=trajectory.selected_skill_cards,
                    history_lines=history_before,
                    loaded_skill_cards=loaded_skill_cards_acc,
                    recent_observations=observation_acc,
                ),
                "target": json.dumps(
                    {
                        "type": "load_skills",
                        "skill_name": skill_name,
                        "message": _build_load_skill_message(user_text, skill_name, round_spec.get("load_skill_reasoning", "")),
                    },
                    ensure_ascii=False,
                ),
                "meta": {
                    "task_id": case.case_id,
                    "turn_id": turn_id + 1,
                    "round_index": round_index,
                    "step_kind": "load_skills",
                    "agent": _infer_agent_name(trajectory),
                    "loaded_skill": skill_name,
                },
            }
        )
        turn_id += 1

        if skill_name not in loaded_skill_slugs_acc:
            loaded_skill_cards_acc.append(skill_card)
            loaded_skill_slugs_acc.add(skill_name)

        step_index = 1
        for action in round_spec["actions"]:
            samples.append(
                {
                    "input": build_input_text(
                        case,
                        available_skill_cards=trajectory.selected_skill_cards,
                        history_lines=history_before,
                        loaded_skill_cards=loaded_skill_cards_acc,
                        recent_observations=observation_acc,
                    ),
                    "target": json.dumps(
                        {
                            "type": "load_action",
                            "skill_name": skill_name,
                            "action_name": action["action_name"],
                            "action_input": action["action_input"],
                            "message": _build_load_action_message(action["action_name"], action["observation"], action.get("reasoning", "")),
                        },
                        ensure_ascii=False,
                    ),
                    "meta": {
                        "task_id": case.case_id,
                        "turn_id": turn_id + 1,
                        "round_index": round_index,
                        "step_index": step_index,
                        "step_kind": "load_action",
                        "agent": _infer_agent_name(trajectory),
                        "loaded_skill": skill_name,
                    },
                }
            )
            turn_id += 1
            step_index += 1
            observation_acc.append({"tool_name": action["action_name"], "output": action["observation"]})

        reply_message = round_spec["reply_text"]
        samples.append(
            {
                "input": build_input_text(
                    case,
                    available_skill_cards=trajectory.selected_skill_cards,
                    history_lines=history_before,
                    loaded_skill_cards=loaded_skill_cards_acc,
                    recent_observations=observation_acc,
                ),
                "target": json.dumps(
                    {
                        "type": "respond",
                        "skill_name": skill_name,
                        "message": reply_message,
                    },
                    ensure_ascii=False,
                ),
                "meta": {
                    "task_id": case.case_id,
                    "turn_id": turn_id + 1,
                    "round_index": round_index,
                    "step_index": step_index,
                    "step_kind": "respond",
                    "agent": _infer_agent_name(trajectory),
                    "loaded_skill": skill_name,
                },
            }
        )
        turn_id += 1
        history_acc = history_before + [f"客服: {reply_message}"]
    return samples


def build_input_text(
    case: CaseSpec,
    available_skill_cards: List[Dict[str, Any]],
    history_lines: List[str],
    loaded_skill_cards: List[Dict[str, Any]],
    recent_observations: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("SYSTEM:")
    lines.append(SYSTEM_PROMPT)
    lines.append("")
    lines.append("## 可用 skills")
    for card in available_skill_cards:
        lines.append(f"- {card['slug']}: {_compact_description(card)}")
    lines.append("")
    lines.append("## 信号")
    for key, value in _build_signals(case).items():
        lines.append(f"- {key}: {_render_value(value)}")
    lines.append("")
    lines.append("## 对话历史")
    lines.extend(history_lines)
    lines.append("")
    lines.append("## 已加载 skill")
    if loaded_skill_cards:
        for card in loaded_skill_cards:
            lines.append(f"- {card['slug']}")
    else:
        lines.append("[]")
    lines.append("")
    lines.append("## 已加载 skill 内容")
    if loaded_skill_cards:
        first = True
        for card in loaded_skill_cards:
            if not first:
                lines.append("")
            lines.extend(_render_loaded_skill(card))
            first = False
    else:
        lines.append("无")
    lines.append("")
    lines.append("## 最近观察")
    if recent_observations:
        for obs in recent_observations:
            lines.append(f"- {obs['tool_name']} => {json.dumps(obs['output'], ensure_ascii=False)}")
    else:
        lines.append("无")
    return "\n".join(lines).strip()


def allowed_actions(skill_card: Dict[str, Any]) -> List[str]:
    return list(skill_card.get("read_tools", [])) + list(skill_card.get("write_tools", []))


def _group_reasoning_rounds(trajectory: Trajectory) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for step in trajectory.reasoning_steps:
        round_index = int(getattr(step, "round_index", 1) or 1)
        user_text = getattr(step, "user_text", None) or trajectory.user_message
        skill_name = step.selected_skill or (trajectory.selected_skills[0] if trajectory.selected_skills else "")
        bucket = grouped.setdefault(round_index, {
            "round_index": round_index,
            "user_text": user_text,
            "skill_name": skill_name,
            "load_skill_reasoning": step.thought or "",
            "actions": [],
            "reply_text": "",
        })
        if step.action_type == "tool" and step.tool_name:
            bucket["actions"].append({
                "action_name": step.tool_name,
                "action_input": step.tool_args or {},
                "observation": (step.observation or {}).get("output", step.observation or {}),
                "reasoning": step.thought or "",
            })
        elif step.action_type == "reply" and step.response_text is not None:
            bucket["reply_text"] = step.response_text
    rounds = [grouped[k] for k in sorted(grouped)]
    return [item for item in rounds if item.get("skill_name") and item.get("reply_text")]


def _compact_description(card: Dict[str, Any]) -> str:
    route_tags = card.get("route_tags", [])
    if route_tags:
        return f"处理标签: {', '.join(route_tags[:3])}"
    desc = card.get("description", "")
    return desc[:80]


def _build_signals(case: CaseSpec) -> Dict[str, Any]:
    return {
        "用户年龄": case.customer_profile.get("age", "未知"),
        "用户性别": case.customer_profile.get("gender", "未知"),
        "投保产品": case.policy_summary.get("product"),
        "保单状态": case.policy_summary.get("status", "有效"),
        "理赔案件状态": case.claim_case.get("status"),
        "用户渠道": case.customer_profile.get("channel"),
        "用户情绪": case.customer_profile.get("emotion"),
        "风险标记": case.claim_case.get("risk_flags", []),
        "已上传材料": case.uploaded_documents,
        "应上传材料": case.required_documents,
        "当前可自助处理": "是" if case.safe_to_self_serve else "否",
        "必须升级人工": "是" if case.must_escalate else "否",
        "当前主问题标签": case.subissues,
        "bucket": getattr(case, "bucket", "train"),
    }


def _render_loaded_skill(card: Dict[str, Any]) -> List[str]:
    lines = [f"[Skill: {card['slug']}]"]
    lines.append(f"- 目标: {_compact_description(card)}")
    actions = allowed_actions(card)
    if actions:
        lines.append(f"- 允许动作: {', '.join(actions)}")
    guardrails = card.get("guardrails", [])
    if guardrails:
        lines.append(f"- 守卫: {'；'.join(guardrails[:3])}")
    return lines


def _render_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _infer_agent_name(trajectory: Trajectory) -> str:
    return getattr(trajectory, "agent_name", "unknown")


def _build_load_skill_message(user_text: str, skill_name: str, thought: str) -> str:
    thought = (thought or "").strip()
    if thought:
        return f"结合用户问题，分析后需要调用 {skill_name} 来处理：{user_text}。依据：{thought}"
    return f"结合用户问题，分析后需要调用 {skill_name} 来处理：{user_text}"


def _build_load_action_message(action_name: str, observation: Dict[str, Any], thought: str) -> str:
    prefix = f"执行 {action_name} 动作，结果是 {json.dumps(observation, ensure_ascii=False)}"
    thought = (thought or "").strip()
    if thought:
        return f"{prefix}。处理依据：{thought}"
    return prefix
