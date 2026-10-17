---
name: rejection_reconsideration
description: 拒赔复议。当用户出现以下情形时触发：用户表示不认可拒赔、要求申诉或复议。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 拒赔复议

## 触发条件
- 用户表示不认可拒赔、要求申诉或复议。
- 路由标签：reconsideration_path, rejection_dispute

## 需要先观察的内容
- get_claim_case
- get_policy_terms
- get_escalation_matrix

## 推荐动作顺序
1. 先确认当前为 rejected。
2. 说明现有结论依据。
3. 判断是否可发起复议。
4. 创建复议单并记录需要补充的新证据。

## 可执行写动作
- create_reconsideration_ticket
- append_case_note

## 守卫规则
- 禁止劝阻用户复议。
- 禁止未读案件状态就创建复议单。

## 输出约束
- 回复写清复议入口、需要补充的材料、后续流转。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

