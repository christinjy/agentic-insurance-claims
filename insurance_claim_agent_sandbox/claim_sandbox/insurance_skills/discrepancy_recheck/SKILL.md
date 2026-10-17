---
name: discrepancy_recheck
description: 资料冲突复核。当用户出现以下情形时触发：票据日期、住院日期、金额等存在冲突。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 资料冲突复核

## 触发条件
- 票据日期、住院日期、金额等存在冲突。
- 路由标签：document_conflict

## 需要先观察的内容
- get_claim_case
- get_claim_timeline
- get_uploaded_documents

## 推荐动作顺序
1. 先识别冲突。
2. 说明冲突点来自资料不一致。
3. 需要时升级人工复核。

## 可执行写动作
- append_case_note
- escalate_to_human_adjuster

## 守卫规则
- 禁止在冲突未解前给赔付结论。

## 输出约束
- 回复只说明冲突和复核路径。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

