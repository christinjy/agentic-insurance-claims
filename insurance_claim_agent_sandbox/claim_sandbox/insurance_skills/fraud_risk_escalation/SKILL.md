---
name: fraud_risk_escalation
description: 风险件升级。当用户出现以下情形时触发：案件有票据冲突、资料冲突、疑似骗赔等风险标记。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 风险件升级

## 触发条件
- 案件有票据冲突、资料冲突、疑似骗赔等风险标记。
- 路由标签：risk_flags, discrepancy

## 需要先观察的内容
- get_claim_case
- get_escalation_matrix
- search_claim_kb

## 推荐动作顺序
1. 识别 risk_flags。
2. 调用升级矩阵。
3. 说明存在资料冲突，需要人工核赔。
4. 写入案件备注。

## 可执行写动作
- escalate_to_human_adjuster
- append_case_note

## 守卫规则
- 禁止直接向用户下定性说欺诈。
- 禁止私自继续自助处理高风险案件。

## 输出约束
- 回复只说资料冲突/风险复核，不做法律定性。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

