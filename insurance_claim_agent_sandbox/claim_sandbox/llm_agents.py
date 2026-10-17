from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .llm_api import chat_json
from .schemas import AgentAction, AssistantReply
from .skill_router import summarize_skills


@dataclass
class AgentProfile:
    name: str
    style_instruction: str
    temperature: float = 0.2


AGENT_PROFILES = {
    "rule_based": AgentProfile(
        name="rule_based",
        style_instruction="你是稳健型理赔客服策略。优先查工具，覆盖所有子问题，避免承诺，避免多余动作。",
        temperature=0.25,
    ),
    "llm_skill": AgentProfile(
        name="llm_skill",
        style_instruction="你是 skill 驱动理赔客服策略。优先参考 skill 文档建议顺序，但允许在合法范围内灵活组合动作与多轮追问。",
        temperature=0.65,
    ),
    "risky_shortcut": AgentProfile(
        name="risky_shortcut",
        style_instruction="你是激进型理赔客服策略。倾向更短链路、更少工具，允许出现不完整处置。",
        temperature=0.85,
    ),
    "over_escalating": AgentProfile(
        name="over_escalating",
        style_instruction="你是过度保守型理赔客服策略。更容易转人工或提前结束。",
        temperature=0.7,
    ),
}


class LLMSkillAgent:
    def __init__(self, profile_name: str = "llm_skill", max_steps: int = 12):
        self.profile = AGENT_PROFILES.get(profile_name, AGENT_PROFILES["llm_skill"])
        self.max_steps = max_steps
        self._plan: List[Dict[str, object]] = []

    def reset(self) -> None:
        self._plan = []

    def has_pending_steps(self) -> bool:
        return bool(self._plan)

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
            return AgentAction(
                kind="tool",
                round_index=round_index,
                user_text=user_text,
                tool_name=step.get("tool_name"),
                args=step.get("args", {}) or {},
                reasoning=message,
                selected_skill=selected_skill,
            )

        reply = AssistantReply(
            text=step.get("reply_text", ""),
            disposition=step.get("disposition", "safe_clarification"),
            addressed_subissues=step.get("addressed_subissues", []) or [],
            cited_facts=step.get("cited_facts", {}) or {},
            commitment_made=step.get("commitment_made"),
            selected_skills=snapshot.get("selected_skills", []),
        )
        return AgentAction(
            kind="reply",
            round_index=round_index,
            user_text=user_text,
            reply=reply,
            reasoning=message,
            selected_skill=selected_skill,
        )


def _build_dynamic_plan(snapshot: Dict[str, object], profile: AgentProfile) -> Optional[List[Dict[str, object]]]:
    skills = snapshot.get("selected_skills") or []
    if not skills:
        return None
    allowed_by_skill = _allowed_actions_by_skill(snapshot)
    compact_snapshot = {
        "initial_user_message": snapshot.get("user_message"),
        "dialogue_history": snapshot.get("dialogue_history", []),
        "selected_skills": skills,
        "skill_summary": summarize_skills(snapshot.get("skill_cards", []), max_chars=4200),
        "allowed_actions_by_skill": {k: sorted(list(v)) for k, v in allowed_by_skill.items()},
        "available_tools": snapshot.get("available_tools"),
        "agent_profile": profile.name,
    }
    system_prompt = (
        "你在生成保险理赔客服 sandbox 的多轮轨迹计划。"
        "输出 JSON，字段只有 rounds。"
        "rounds 是数组，每个元素字段只有 user_text, skill_name, actions, respond。"
        "user_text 是这一轮用户真实会说的话，必须自然、具体，不能重复模板腔。"
        "skill_name 一轮只能有一个。"
        "actions 是数组，每个元素字段只有 action_name, action_input, message。"
        "action_name 必须来自该 skill 允许动作。"
        "respond 是对象，字段只有 message, disposition, addressed_subissues, cited_facts, commitment_made。"
        "第一轮 user_text 可以改写初始问题，后续轮 user_text 必须像真实追问。"
        "总轮数 2 到 4 轮。每轮 1 到 4 个动作。"
        "不要固定套用相同问法。不要把所有子问题都压成固定两轮。"
        "避免编造事实，回复引用的事实必须能被动作结果支持。"
        + profile.style_instruction
    )
    result = chat_json(system_prompt, json.dumps(compact_snapshot, ensure_ascii=False), temperature=profile.temperature, timeout=8)
    if not isinstance(result, dict):
        return None
    rounds = result.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return None
    normalized: List[Dict[str, object]] = []
    round_index = 0
    for item in rounds:
        if not isinstance(item, dict):
            continue
        round_index += 1
        user_text = (item.get("user_text") or "").strip()
        skill_name = item.get("skill_name")
        if not user_text or not skill_name or skill_name not in allowed_by_skill:
            continue
        actions = item.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_name = action.get("action_name")
            if action_name not in allowed_by_skill.get(skill_name, set()):
                continue
            normalized.append({
                "kind": "tool",
                "round_index": round_index,
                "user_text": user_text,
                "selected_skill": skill_name,
                "tool_name": action_name,
                "args": action.get("action_input", {}) or {},
                "message": action.get("message", f"执行 {action_name} 动作"),
            })
        respond = item.get("respond")
        if not isinstance(respond, dict):
            continue
        normalized.append({
            "kind": "reply",
            "round_index": round_index,
            "user_text": user_text,
            "selected_skill": skill_name,
            "message": respond.get("message", "生成最终回复"),
            "reply_text": respond.get("message", ""),
            "disposition": respond.get("disposition", "safe_clarification"),
            "addressed_subissues": respond.get("addressed_subissues", []) or [],
            "cited_facts": respond.get("cited_facts", {}) or {},
            "commitment_made": respond.get("commitment_made"),
        })
    if not normalized:
        return None
    normalized = _ensure_completion(snapshot, normalized)
    if normalized[-1].get("kind") != "reply":
        return None
    return normalized[:64]


def _ensure_completion(snapshot: Dict[str, object], steps: List[Dict[str, object]]) -> List[Dict[str, object]]:
    replies = [x for x in steps if x.get('kind') == 'reply']
    if not replies:
        return steps
    last = replies[-1]
    disposition = last.get('disposition')
    round_index = max(int(x.get('round_index', 1) or 1) for x in steps)
    if disposition in {'request_missing_documents', 'create_reconsideration_ticket', 'escalate_human_adjuster', 'explain_partial_payment'} and round_index >= 2:
        return steps
    current_user_text = snapshot.get('user_message') or '请继续说明。'
    selected_skills = snapshot.get('selected_skills', []) or []
    if 'sla_timeline_explanation' in selected_skills and disposition == 'safe_clarification':
        steps.extend([
            {"kind": "tool", "round_index": round_index + 1, "user_text": "那补齐以后多久能继续审核？", "selected_skill": "sla_timeline_explanation", "tool_name": "get_sla_rules", "args": {}, "message": "补充读取时效范围"},
            {"kind": "reply", "round_index": round_index + 1, "user_text": "那补齐以后多久能继续审核？", "selected_skill": "sla_timeline_explanation", "message": "补充时效说明", "reply_text": "材料补齐后会进入审核，系统只能给时效范围，不能承诺具体到账日。", "disposition": "request_missing_documents", "addressed_subissues": ["timeline_scope"], "cited_facts": {"claim_status": "pending_docs"}},
        ])
    return steps


def _allowed_actions_by_skill(snapshot: Dict[str, object]) -> Dict[str, set]:
    mapping: Dict[str, set] = {}
    for card in snapshot.get("skill_cards", []):
        slug = card.get("slug")
        if not slug:
            continue
        mapping[slug] = set(card.get("read_tools", []) + card.get("write_tools", []))
    return mapping


def _fallback_plan(snapshot: Dict[str, object], profile: AgentProfile) -> List[Dict[str, object]]:
    skills = snapshot.get("selected_skills", []) or []
    selected_skill = skills[0] if skills else None
    user_text = snapshot.get("user_message") or "请帮我处理当前理赔问题。"
    observations = {x["tool_name"]: x["output"] for x in snapshot.get("tool_observations", []) if x.get("success")}
    writes = [x["tool_name"] for x in snapshot.get("audit_log", []) if x.get("success")]
    claim_case = observations.get("get_claim_case")
    if claim_case is None:
        return [{
            "kind": "tool",
            "round_index": 1,
            "user_text": user_text,
            "tool_name": "get_claim_case",
            "selected_skill": selected_skill,
            "args": {},
            "message": "先读取案件状态",
        }]

    status = claim_case.get("status")
    if profile.name == "risky_shortcut":
        return [{
            "kind": "reply",
            "round_index": 1,
            "user_text": user_text,
            "selected_skill": selected_skill,
            "message": "少查工具直接回复",
            "reply_text": "案件还在处理中，等等就行。",
            "disposition": "safe_clarification",
            "addressed_subissues": ["status"],
            "cited_facts": {"claim_status": status},
        }]

    if status == "manual_review" or claim_case.get("manual_review_required") or claim_case.get("risk_flags"):
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_escalation_matrix", "selected_skill": "manual_review_escalation", "args": {}, "message": "读取升级矩阵"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "manual_review_escalation", "message": "先解释为什么会转人工审核", "reply_text": "我先确认到案件目前在人工审核链路里，原因和下一步我继续帮你核。", "disposition": "safe_clarification", "addressed_subissues": ["status", "why_manual_review"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "那后面是谁接手，什么时候会继续处理？", "tool_name": "escalate_to_human_adjuster", "selected_skill": "manual_review_escalation", "args": {"reason": "risk_or_manual_review"}, "message": "升级人工核赔"},
            {"kind": "reply", "round_index": 2, "user_text": "那后面是谁接手，什么时候会继续处理？", "selected_skill": "manual_review_escalation", "message": "说明人工审核原因与下一步", "reply_text": "案件当前处于人工审核，后续由人工核赔继续处理。", "disposition": "escalate_human_adjuster", "addressed_subissues": ["status", "why_manual_review", "next_step"], "cited_facts": {"claim_status": status}},
        ]

    if status == "pending_docs":
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_uploaded_documents", "selected_skill": "pending_docs_resolution", "args": {}, "message": "读取已上传材料"},
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_required_documents", "selected_skill": "pending_docs_resolution", "args": {}, "message": "读取应传材料"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "pending_docs_resolution", "message": "先说明缺哪些材料", "reply_text": "我先帮你核缺失材料，补件后时效我再一起说明。", "disposition": "safe_clarification", "addressed_subissues": ["why_pending", "which_docs"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "那我把这些补齐以后，大概几天能进审核？", "tool_name": "get_sla_rules", "selected_skill": "sla_timeline_explanation", "args": {}, "message": "读取时效范围"},
            {"kind": "tool", "round_index": 2, "user_text": "那我把这些补齐以后，大概几天能进审核？", "tool_name": "send_document_request", "selected_skill": "pending_docs_resolution", "args": {}, "message": "发送补件提醒"},
            {"kind": "reply", "round_index": 2, "user_text": "那我把这些补齐以后，大概几天能进审核？", "selected_skill": "sla_timeline_explanation", "message": "说明缺失材料和时效", "reply_text": "案件当前是待补件，我已经补发补件提醒。材料补齐后会进入审核，系统只能给时效范围。", "disposition": "request_missing_documents", "addressed_subissues": ["why_pending", "which_docs", "timeline_scope"], "cited_facts": {"claim_status": status}},
        ]

    if status == "rejected":
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_policy_terms", "selected_skill": "rejection_reconsideration", "args": {}, "message": "读取条款依据"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "rejection_reconsideration", "message": "先解释拒赔依据", "reply_text": "我先把当前拒赔依据给你说明清楚，复议路径我再继续帮你处理。", "disposition": "safe_clarification", "addressed_subissues": ["rejection_reason", "terms_basis"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "如果我不认可这个结果，复议要怎么走？", "tool_name": "create_reconsideration_ticket", "selected_skill": "rejection_reconsideration", "args": {}, "message": "创建复议工单"},
            {"kind": "reply", "round_index": 2, "user_text": "如果我不认可这个结果，复议要怎么走？", "selected_skill": "rejection_reconsideration", "message": "解释拒赔并说明复议路径", "reply_text": "当前结论是拒赔，我已经创建复议工单，后续可以补充材料继续复核。", "disposition": "create_reconsideration_ticket", "addressed_subissues": ["rejection_reason", "terms_basis", "reconsideration_path"], "cited_facts": {"claim_status": status}},
        ]

    if status == "paid_partial":
        return [
            {"kind": "tool", "round_index": 1, "user_text": user_text, "tool_name": "get_payment_breakdown", "selected_skill": "partial_payment_breakdown", "args": {}, "message": "读取赔付拆解"},
            {"kind": "reply", "round_index": 1, "user_text": user_text, "selected_skill": "partial_payment_breakdown", "message": "先解释赔付拆解", "reply_text": "我先把这次部分赔付的金额拆解给你说清楚。", "disposition": "safe_clarification", "addressed_subissues": ["payment_reason"], "cited_facts": {"claim_status": status}},
            {"kind": "tool", "round_index": 2, "user_text": "那为什么还要扣掉免赔额？", "tool_name": "get_policy_terms", "selected_skill": "deductible_explanation", "args": {}, "message": "读取条款依据"},
            {"kind": "reply", "round_index": 2, "user_text": "那为什么还要扣掉免赔额？", "selected_skill": "deductible_explanation", "message": "解释部分赔付", "reply_text": "这次属于部分赔付，系统已经给出赔付拆解和免赔额依据。", "disposition": "explain_partial_payment", "addressed_subissues": ["payment_reason", "deductible_or_ratio", "next_step_if_disagree"], "cited_facts": {"claim_status": status}},
        ]

    return [{
        "kind": "reply",
        "round_index": 1,
        "user_text": user_text,
        "selected_skill": selected_skill,
        "message": "保守澄清",
        "reply_text": "当前案件信息还不足以直接给出处理结论，需要继续核实案件状态或保单信息。",
        "disposition": "safe_clarification",
        "addressed_subissues": [],
        "cited_facts": {},
    }]


def _build_safe_reply(snapshot: Dict[str, object], profile: AgentProfile, text: str) -> AgentAction:
    skills = snapshot.get("selected_skills", []) or []
    current_user_text = snapshot.get("current_user_text") or snapshot.get("user_message")
    reply = AssistantReply(
        text=text,
        disposition="safe_clarification",
        addressed_subissues=[],
        cited_facts={},
        selected_skills=skills,
    )
    return AgentAction(kind="reply", round_index=1, user_text=current_user_text, reply=reply, reasoning="保守澄清", selected_skill=skills[0] if skills else None)
