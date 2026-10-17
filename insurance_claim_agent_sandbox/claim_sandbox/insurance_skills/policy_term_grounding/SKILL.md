---
name: policy_term_grounding
description: 条款依据定位。当用户出现以下情形时触发：用户质疑赔付、拒赔、时效或处理规则依据。 强制先核实案件事实，再依据技能步骤组织回复。
---

# 条款依据定位

## 触发条件
- 用户质疑赔付、拒赔、时效或处理规则依据。
- 路由标签：terms_basis, coverage_basis

## 需要先观察的内容
- get_policy_terms
- get_policy_summary
- search_claim_kb

## 推荐动作顺序
1. 先读 policy terms。
2. 把条款规则转成客服可说的人话。
3. 若条款解释可能引发争议，追加 case note 记录已解释依据。

## 可执行写动作
- append_case_note

## 守卫规则
- 禁止创造条款编号或不存在的责任定义。

## 输出约束
- 回复必须出现条款依据和通俗解释。
- 回复必须基于已观察事实，不补不存在的条款、时效、金额、案件节点。
- 如果用户同时问多个问题，先拆问题，再逐项覆盖。

