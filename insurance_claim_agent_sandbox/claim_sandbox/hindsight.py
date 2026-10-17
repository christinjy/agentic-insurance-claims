from __future__ import annotations

import time

from .exporters import serialize_trajectory_analysis_input
from .llm_api import chat_json
from .schemas import CaseSpec, Trajectory, VerifierResult


HINDSIGHT_SYSTEM_PROMPT = (
    "你在为保险理赔客服智能体生成 hindsight skill。"
    "输入是一条已完成 episode 的任务、工具轨迹、最终回复和 verifier 结果。"
    "输出 JSON，字段只有 skill。"
    "skill 是一条可复用的经验规则，中文一到三句话，围绕以下三类内容之一："
    "workflow skill（应该按什么流程做）、"
    "decisive observation skill（哪些观察决定了后续动作）、"
    "failure-avoidance skill（哪些行为会导致风险或失败）。"
    "只基于输入中的事实，不要编造轨迹里不存在的细节。"
)


def generate_hindsight_skill(case: CaseSpec, trajectory: Trajectory, verifier: VerifierResult) -> str:
    return _fallback_hindsight_skill(case, verifier)


def _llm_hindsight_skill(case: CaseSpec, trajectory: Trajectory, verifier: VerifierResult) -> str | None:
    analysis_input = serialize_trajectory_analysis_input(case, trajectory, verifier)
    for attempt in range(3):
        result = chat_json(HINDSIGHT_SYSTEM_PROMPT, analysis_input, temperature=0.3, timeout=30)
        if isinstance(result, dict):
            skill = result.get("skill")
            if isinstance(skill, str) and skill.strip():
                return skill.strip()
        if attempt < 2:
            time.sleep(1)
    return None


def _fallback_hindsight_skill(case: CaseSpec, verifier: VerifierResult) -> str:
    if verifier.success:
        if case.family == "pending_docs":
            return "先查案件状态，再核对已传材料与缺失材料，随后解释审核时效范围；在未核验支付节点前，不要直接承诺到账时间。"
        if case.family == "rejection_dispute":
            return "遇到拒赔争议，先调取条款依据，再解释核心限制条件；若用户明确表示不认可，应同步提供复议路径并创建复议工单。"
        if case.family == "high_risk_review":
            return "当案件存在资料冲突、人工复核或高风险标记时，优先参考升级矩阵，将案件升级到人工核赔，而不是继续做自助承诺。"
        if case.family == "partial_payment":
            return "解释部分赔付前，应先查询赔付拆分和条款依据，再说明免赔额、比例或责任限额对赔付金额的影响。"
    if "unauthorized_commitment" in verifier.hard_guardrails:
        return "未核验 SLA 和支付节点前，禁止承诺具体到账时间；用时效范围替代确定性承诺。"
    if "missed_mandatory_escalation" in verifier.hard_guardrails:
        return "只要出现人工复核、高风险标记或资料冲突，就必须按升级矩阵转人工，不应继续自助结案。"
    if "subissue_omission" in verifier.soft_penalties:
        return "用户一次提出多个问题时，先拆分为子问题逐项覆盖，避免只回答其中一个。"
    return "完成轨迹后要复盘：这次案件中哪些查询动作提供了关键证据，哪些结论没有依据，下次应先查再答。"
