---
name: partial_payment_breakdown
description: 部分赔付拆解。当用户出现以下情形时触发：用户问为什么只赔了一部分，要求看计算方式。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 部分赔付拆解

## 触发条件
- 用户问为什么只赔了一部分，要求看计算方式。
- 路由标签：payment_reason, payment_breakdown

## 需要先观察的内容
- get_claim_case
- get_payment_breakdown
- get_policy_terms

## 推荐动作顺序
1. 读取赔付状态和 payment breakdown。
2. 说明可赔金额、免赔额、超限部分。
3. 把结果映射回用户提到的总花费与实赔金额差。

## 可执行写动作
- append_case_note

## 守卫规则
- 禁止未读 breakdown 就给金额解释。
- 禁止把不赔原因说成系统故障。

## 输出约束
- 回复要出现总额、可赔额、免赔额或超限原因。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

