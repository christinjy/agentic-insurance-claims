---
name: pending_docs_resolution
description: 待补件处理。当用户出现以下情形时触发：用户问待补件、缺什么材料、补件后多久进入审核或到账。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 待补件处理

## 触发条件
- 用户问待补件、缺什么材料、补件后多久进入审核或到账。
- 路由标签：pending_docs, why_pending, which_docs, timeline_scope

## 需要先观察的内容
- get_claim_case
- get_uploaded_documents
- get_required_documents
- get_sla_rules

## 推荐动作顺序
1. 先读取案件状态，确认确实是 pending_docs。
2. 对比 required_documents 与 uploaded_documents，给出缺失清单。
3. 读取 SLA，只能说时效范围，不能承诺精确到账日。
4. 必要时发送补件提醒。

## 可执行写动作
- send_document_request

## 守卫规则
- 禁止在未读 SLA 前承诺到账日。
- 禁止把已上传材料说成缺失。
- 禁止跳过缺失清单直接让用户等待。

## 输出约束
- 回复必须包含当前状态、缺失材料、时效范围、下一步。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

