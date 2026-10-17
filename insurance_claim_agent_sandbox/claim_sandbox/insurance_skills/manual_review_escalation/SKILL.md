---
name: manual_review_escalation
description: 人工审核升级。当用户出现以下情形时触发：案件处于 manual_review 或用户追问为什么卡住、什么时候能结案。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 人工审核升级

## 触发条件
- 案件处于 manual_review 或用户追问为什么卡住、什么时候能结案。
- 路由标签：manual_review, why_manual_review, next_step

## 需要先观察的内容
- get_claim_case
- get_escalation_matrix
- get_claim_timeline

## 推荐动作顺序
1. 先确认 manual_review_required 或 risk_flags。
2. 说明进入人工审核的直接原因。
3. 读取升级矩阵。
4. 执行升级或确认已在人工审核链路。

## 可执行写动作
- escalate_to_human_adjuster
- append_case_note

## 守卫规则
- 禁止在人工审核前承诺结案时间。
- 禁止把人工审核说成系统故障。

## 输出约束
- 回复要有当前状态、触发原因、下一步和人工接手信息。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

