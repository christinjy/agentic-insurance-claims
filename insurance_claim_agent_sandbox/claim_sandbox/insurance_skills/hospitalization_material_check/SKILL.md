---
name: hospitalization_material_check
description: 住院材料核对。当用户出现以下情形时触发：案件是住院理赔，用户问为什么不能过审或材料是否齐全。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 住院材料核对

## 触发条件
- 案件是住院理赔，用户问为什么不能过审或材料是否齐全。
- 路由标签：hospitalization, pending_docs

## 需要先观察的内容
- get_claim_case
- get_required_documents
- get_uploaded_documents
- get_policy_summary

## 推荐动作顺序
1. 先识别是住院险案件。
2. 列出住院场景常见关键材料：出院小结、发票、身份材料。
3. 结合当前 case 判断哪些已齐、哪些未齐。
4. 未齐时触发补件。

## 可执行写动作
- send_document_request

## 守卫规则
- 禁止把门诊材料当住院核心材料。
- 禁止脱离具体案件泛泛输出清单。

## 输出约束
- 回复要给出已齐和未齐两部分。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

