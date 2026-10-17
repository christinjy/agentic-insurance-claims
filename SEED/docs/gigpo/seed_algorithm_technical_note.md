# SEED 算法技术说明

## 摘要

本文档根据论文 `ICLR_27_SEED (3).pdf` 更新当前仓库中 SEED 的算法说明。
论文版 SEED 全称为 **SElf-Evolving On-Policy Distillation**，核心目标是在
长程 agentic RL 中，把完整轨迹结束后才能看到的 hindsight 信息转化为
token-level 训练信号，同时避免在推理阶段依赖额外 skill prompt、检索库或外部
分析器。

SEED 有两个阶段：

1. **Hindsight Skill SFT**：先收集普通 agent 轨迹，用外部 analyzer 为完整轨迹
   生成 episode-level hindsight skill，再 SFT 当前 backbone，使其具备“读完整
   轨迹并总结 skill”的能力。
2. **Self-Evolving OPD**：RL 时，冻结的当前策略快照既负责采样 on-policy 轨迹，
   也作为同步 analyzer 总结这些轨迹。随后训练中的策略在普通上下文和
   skill-augmented 上下文下对同一批 sampled action tokens 重新打分，用
   skill-induced log-prob shift 构造 gated OPD loss，并与 GRPO loss 联合优化。

最重要的推理侧结论是：**SEED 的 skill 只在训练时作为 privileged supervision
使用，推理时只部署普通 policy**。

## 1. 论文核心 Idea

SEED 针对长程 agentic RL 的三个问题设计：

| 需求 | SEED 的对应机制 |
| --- | --- |
| on-policy | 轨迹由当前策略快照采样，skill 也由同一快照分析得到 |
| dense | 同一 action token 在普通上下文和 skill 上下文下重新打分，形成 token-level OPD 信号 |
| self-evolving | 每轮更新后，actor 和 analyzer 都随最新 checkpoint 同步刷新 |

直观地说，episode reward 只能告诉我们一条轨迹最后成功或失败；完整轨迹中的
hindsight skill 则能指出“成功 workflow”“关键观察”“失败规避规则”。SEED 不把
这些 skill 当作推理 prompt，而是把 skill 对当前策略行为概率的影响蒸馏进模型
参数。

## 2. 当前代码路径说明

当前仓库保留论文使用的 SFT + self-evolving OPD 主路径：

| 路径 | 典型脚本 | 说明 |
| --- | --- | --- |
| 论文主路径 | `examples/seed_trainer/run_*_sft*.sh` | 从 hindsight-skill SFT checkpoint 初始化 policy 和 `policy_vllm` analyzer，用 `opd_loss_coef` 启用 gated OPD loss |

启动脚本名已省略历史字段 `episode_no_skill_loss`。它表示默认不启用额外的
skill-generation LM auxiliary loss，并不表示关闭论文式 OPD loss；OPD 仍通过
`actor_rollout_ref.actor.opd_loss_coef=0.01` 启用。

## 3. Stage 1: Hindsight Skill SFT

论文中每个 benchmark 选择 180 个 SFT 任务，每个任务采样 8 条 rollout，总计
1,440 条完整轨迹：

$$
B_j = \{\tau_{j,k}\}_{k=1}^{K_0}, \quad K_0=8.
$$

每条轨迹包含 task description、observations、actions、rewards 和 final outcome。
外部 analyzer 对完整轨迹生成 episode-level skill：

$$
s_\tau = A_{\text{ext}}(\tau).
$$

成功轨迹的 skill 应总结可复用 workflow；失败轨迹的 skill 应总结 avoidance
rule。论文 prompt 要求输出合法 JSON：

```json
{
  "episode_summary": "string",
  "episode_skill": "string"
}
```

通过格式校验后的样本组成：

$$
D_{\text{sft}} = \{(x_\tau, s_\tau): v_\tau=1\},
$$

其中 \(x_\tau\) 是序列化后的 trajectory-analysis input。SFT 目标是标准
negative log-likelihood：

$$
L_{\text{sft}}(\theta)
= -\mathbb{E}_{(x_\tau,s_\tau)\sim D_{\text{sft}}}
\sum_\ell \log \pi_\theta(s_{\tau,\ell}\mid x_\tau,s_{\tau,<\ell}).
$$

SFT 后的 checkpoint 既作为后续 RL actor，也作为后续同步 trajectory analyzer
的初始化。

对应脚本：

```bash
# ALFWorld
bash scripts/sft/alfworld/prepare_data.sh
bash scripts/sft/alfworld/train_sft.sh

# WebShop
bash scripts/sft/webshop/prepare_data.sh
bash scripts/sft/webshop/train_sft.sh

# Search-based QA
bash scripts/sft/search/prepare_data.sh
bash scripts/sft/search/train_sft.sh

# EZPoints
bash scripts/sft/ezpoints/prepare_data.sh
bash scripts/sft/ezpoints/train_sft.sh

# Sokoban
bash scripts/sft/sokoban/prepare_data.sh
bash scripts/sft/sokoban/train_sft.sh
```

## 4. Stage 2: Self-Evolving OPD

在第 \(k\) 轮 policy update 开始时，SEED 冻结当前策略为
\(\pi_{\theta_{\text{old}}}\)。这个快照承担两个角色：

1. actor：在环境中采样 on-policy trajectories；
2. analyzer：读取完整轨迹并生成 hindsight skill。

对于任务 \(q\)，采样 \(N\) 条轨迹：

$$
G_q = \{\tau_q^{(1)},\ldots,\tau_q^{(N)}\},\quad
\tau_q^{(n)} \sim \pi_{\theta_{\text{old}}}(\cdot\mid q).
$$

论文和主要脚本均使用 rollout group size \(N=8\)。同步 analyzer 生成：

$$
s_q^{(n)} = A_{\theta_{\text{old}}}(x_{\tau_q^{(n)}}).
$$

这形成 self-evolving loop：策略越强，采样到的轨迹分布会变化；同一个 checkpoint
的分析能力也会随训练一起变化，因此 hindsight supervision 不会长期停留在旧
策略或静态 skill 库上。

## 5. Skill-Augmented Re-Scoring

令 \(h_{q,n,t}\) 表示第 \(n\) 条轨迹第 \(t\) 步的普通 interaction history，
\(a_{q,n,t}\) 是已采样出的 action token 序列。SEED 不重新采样动作，而是把
episode skill 插入上下文：

$$
\tilde{h}_{q,n,t}=H(h_{q,n,t},s_q^{(n)}).
$$

训练中的当前策略 \(\pi_\theta\) 对同一批 sampled action tokens 计算两种
log-prob：

$$
\ell^{\text{skill}}_{q,n,t,\ell}
=\log\pi_\theta(a_{q,n,t,\ell}\mid \tilde{h}_{q,n,t},a_{q,n,t,<\ell}),
$$

$$
\ell^\theta_{q,n,t,\ell}
=\log\pi_\theta(a_{q,n,t,\ell}\mid h_{q,n,t},a_{q,n,t,<\ell}).
$$

两条分支共享同一个模型参数，但 teacher branch 看到 hindsight skill，student
branch 只看到普通上下文。梯度只通过普通 student branch。

skill-induced log-prob shift 定义为：

$$
\Delta_{q,n,t,\ell}
=\operatorname{sg}\left[
\ell^{\text{skill}}_{q,n,t,\ell}
-\ell^\theta_{q,n,t,\ell}
\right],
$$

其中 \(\operatorname{sg}\) 表示 stop-gradient。再用 sigmoid gate 控制 OPD
强度：

$$
g_{q,n,t,\ell}=\sigma(\beta_{\text{opd}}\Delta_{q,n,t,\ell}).
$$

论文默认 \(\beta_{\text{opd}}=5.0\)。

OPD loss 为：

$$
L_{\text{opd}}(\theta)
=
\mathbb{E}_{q,n,t,\ell}
\left[
m_{q,n,t,\ell}\,
g_{q,n,t,\ell}\,
\left(
\operatorname{sg}[\ell^{\text{skill}}_{q,n,t,\ell}]
-\ell^\theta_{q,n,t,\ell}
\right)
\right].
$$

由于 gate 和 teacher log-prob 都 detached，该目标等价于 gate-weighted NLL：
skill 支持的 token 会得到更强蒸馏，skill 不支持的 token 影响会被 attenuate。

对应实现入口：

- [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py) 的
  `compute_opd_loss`
- [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py) 的
  actor update
- [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py) 的
  SEED analysis、teacher/OPD signal 构造

## 6. 与 GRPO 的联合目标

SEED 保留 group-relative RL objective。对同一任务组 \(G_q\)，计算 trajectory
outcome 的均值和标准差：

$$
\mu_q=\frac{1}{N}\sum_{n=1}^N R(\tau_q^{(n)}),\quad
\sigma_q=
\sqrt{
\frac{1}{N}\sum_{n=1}^N
\left(R(\tau_q^{(n)})-\mu_q\right)^2
}.
$$

trajectory-level advantage 为：

$$
A^{\text{rl}}_{q,n}
=
\frac{R(\tau_q^{(n)})-\mu_q}{\sigma_q+\epsilon}.
$$

该 advantage broadcast 到该 trajectory 的有效 action tokens。PPO/GRPO ratio 为：

$$
\rho_{q,n,t,\ell}(\theta)
=
\exp\left(
\ell^\theta_{q,n,t,\ell}
-\ell^{\text{old}}_{q,n,t,\ell}
\right).
$$

最终目标：

$$
L_{\text{SEED}}(\theta)
=
L_{\text{rl}}(\theta)
+
\lambda_{\text{opd}} L_{\text{opd}}(\theta).
$$

论文默认 \(\lambda_{\text{opd}}=0.01\)，KL coefficient 为 \(0.01\)。

## 7. 推理阶段

SEED 推理时只使用普通交互历史：

$$
a_t \sim \pi_\theta(\cdot\mid h_t).
$$

不需要：

- trajectory analyzer；
- skill bank；
- skill retrieval；
- 额外 skill prompt；
- privileged context。

这也是 SEED 与 Skill-Prompt、Skill-GRPO* 等方法的关键区别：后者在评测时仍
依赖 skill context，而 SEED 将 skill 的行为作用内化到参数中。

## 8. 主要实验数据

下表汇总论文 Table 1 中 GRPO 与 SEED 的 aggregate 对比。每个单元格为
`GRPO -> SEED (+gain)`。

| Backbone | ALFWorld Avg. | Search-QA Avg. | WebShop Score | WebShop Succ. |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 75.0 -> 91.8 (+16.8) | 36.4 -> 45.7 (+9.3) | 79.8 -> 88.5 (+8.7) | 63.3 -> 78.9 (+15.6) |
| Qwen2.5-7B-Instruct | 81.2 -> 96.1 (+14.9) | 42.0 -> 48.6 (+6.6) | 80.9 -> 89.7 (+8.8) | 72.6 -> 78.1 (+5.5) |
| Qwen3-1.7B-Instruct | 46.1 -> 92.0 (+45.9) | 40.8 -> 42.2 (+1.4) | 67.3 -> 87.1 (+19.8) | 38.3 -> 77.3 (+39.0) |

论文还报告：

- 相比 GRPO，SEED 在三种 backbone 上分别带来 ALFWorld +14.9 到 +45.9、
  Search-QA +1.4 到 +9.3、WebShop Score +8.7 到 +19.8、WebShop Success
  +5.5 到 +39.0 的提升。
- Skill-Prompt 和 Skill-GRPO* 在评测时插入 skill，但 SEED 在不使用推理
  skill prompt 的情况下，仍在绝大多数 aggregate 指标上更强。
- 在 ALFWorld，Qwen2.5-3B 的 SEED 为 91.8，高于 SDAR 的 84.4 和
  GRPO+OPSD 的 81.2。

## 9. 样本效率、泛化与消融

### 样本效率

论文 Table 6 显示，SEED 在不同训练数据比例下均超过 GRPO：

| Benchmark | Data | GRPO | SEED | Gain |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 20% | 27.3 | 40.7 | +13.4 |
| ALFWorld | 40% | 42.2 | 58.9 | +16.7 |
| ALFWorld | 60% | 56.3 | 80.7 | +24.4 |
| ALFWorld | 80% | 58.6 | 88.8 | +30.2 |
| ALFWorld | 100% | 75.0 | 91.8 | +16.8 |
| WebShop | 20% | 31.3 | 37.5 | +6.2 |
| WebShop | 40% | 45.3 | 53.1 | +7.8 |
| WebShop | 60% | 57.0 | 62.5 | +5.5 |
| WebShop | 80% | 63.6 | 75.0 | +11.4 |
| WebShop | 100% | 63.3 | 78.9 | +15.6 |

关键结论：ALFWorld 上 SEED 用 60% 数据达到 80.7，已经超过 full-data GRPO
的 75.0。

### ALFWorld Unseen 泛化

论文 Table 7 中，Qwen2.5-3B 在 ALFWorld unseen split 上：

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO | 73.9 | 60.0 | 82.4 | 59.3 | 72.7 | 76.9 | 70.9 |
| SEED | 90.4 | 78.3 | 79.5 | 94.3 | 86.2 | 88.2 | 86.2 |
| Gain | +16.5 | +18.3 | -2.9 | +35.0 | +13.5 | +11.3 | +15.3 |

SEED 在 6 个 unseen task family 中 5 个超过 GRPO，平均提升 15.3。

### 消融

论文 Table 2 的 ALFWorld 消融：

| Variant | Avg. | Drop |
| --- | ---: | ---: |
| SEED | 91.8 | 0.0 |
| w/o Hindsight Skill SFT | 86.0 | -5.8 |
| w/o Self-Evolving OPD | 87.0 | -4.8 |
| w/o On-Policy Skill | 84.4 | -7.4 |

这说明三点都重要：SFT 初始化、持续 OPD 蒸馏、以及从当前策略轨迹中生成
on-policy skill。

### 视觉 agent 扩展

论文 Table 8 在 Qwen2.5-VL-3B-Instruct 上报告：

| Method | Sokoban 6x6 | EZPoints |
| --- | ---: | ---: |
| GRPO | 67.1 | 86.9 |
| SEED | 82.0 | 100.0 |
| Gain | +14.9 | +13.1 |

## 10. 关键配置

论文和当前 paper-style 脚本的关键配置如下：

| 配置项 | 值 |
| --- | --- |
| SFT tasks | 180 per benchmark |
| SFT rollouts per task | 8 |
| SFT trajectories | 1,440 |
| SFT epochs | 3 |
| RL updates | 150 in paper; current scripts commonly set `trainer.total_epochs=160` |
| Rollout group size | `env.rollout.n=8` |
| OPD coefficient | `actor_rollout_ref.actor.opd_loss_coef=0.01` |
| OPD gate sharpness | `actor_rollout_ref.actor.opd_gate_beta=5.0` |
| KL coefficient | `actor_rollout_ref.actor.kl_loss_coef=0.01` |
| Learning rate | `actor_rollout_ref.actor.optim.lr=1e-6` |
| Analyzer backend | `algorithm.seed.analysis_backend=policy_vllm` for paper-style SEED |
| Skill mode | `algorithm.seed.skill_mode=episode_only` for paper-style episode-skill OPD |

## 11. 实现映射

| 功能 | 主要实现位置 |
| --- | --- |
| SFT 数据构建：ALFWorld | [`scripts/sft/alfworld`](../../scripts/sft/alfworld) |
| SFT 数据构建：WebShop | [`scripts/sft/webshop`](../../scripts/sft/webshop) |
| SFT 数据构建：Search-QA | [`scripts/sft/search`](../../scripts/sft/search) |
| SFT 数据构建：EZPoints | [`scripts/sft/ezpoints`](../../scripts/sft/ezpoints) |
| SFT 数据构建：Sokoban | [`scripts/sft/sokoban`](../../scripts/sft/sokoban) |
| SFT 训练 | [`verl/trainer/fsdp_sft_trainer.py`](../../verl/trainer/fsdp_sft_trainer.py) |
| 多步 rollout | [`agent_system/multi_turn_rollout/rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py) |
| SEED 训练器集成 | [`verl/trainer/ppo/ray_trainer.py`](../../verl/trainer/ppo/ray_trainer.py) |
| 轨迹分析 prompt 与 JSON 解析 | [`seed/analysis.py`](../../seed/analysis.py) |
| skill 注入 observation | [`seed/prompting.py`](../../seed/prompting.py) |
| gated OPD loss | [`verl/trainer/ppo/core_algos.py`](../../verl/trainer/ppo/core_algos.py) |
| actor update | [`verl/workers/actor/dp_actor.py`](../../verl/workers/actor/dp_actor.py) |
| SEED 配置 | [`verl/trainer/config/ppo_trainer.yaml`](../../verl/trainer/config/ppo_trainer.yaml) |

## 12. 与 Legacy Teacher-Advantage 路径的关系

当前代码仍支持一种早期路径：计算 enhanced prompt log-prob 与 ordinary prompt
log-prob 的差值，并把这个 delta 按权重融入 PPO advantage：

$$
A^{\text{SEED}}
=
A^{\text{ep}}
+w_{\text{ep}}\Delta^{\text{ep}}
+w_{\text{step}}\Delta^{\text{step}}.
$$

这个路径由 `episode_skill_teacher_advantage_w` 和
`step_skill_teacher_advantage_w` 控制；当
`actor_rollout_ref.actor.opd_loss_coef > 0` 时，代码会优先使用论文式的
auxiliary OPD loss，这些 teacher-advantage 权重会被忽略或置为 0。

因此：

- 复现实验和论文主线时，使用 SFT checkpoint + `policy_vllm` analyzer +
  `opd_loss_coef=0.01`；
- 做直接 ablation 或快速实验时，在对应的 `run_*_sft*.sh` 上通过环境变量或
  Hydra 参数覆盖 analyzer、loss 和 teacher-advantage 配置，不再维护独立的
  legacy 启动脚本。

## 13. 结论

论文版 SEED 可以概括为：

$$
\text{SEED}
=
\text{GRPO outcome optimization}
+
\text{self-evolving hindsight-skill OPD}.
$$

它不是在推理阶段“多给一个 skill prompt”，也不是用静态 skill 库做检索，而是
在训练时让当前策略从自己的完整轨迹中产生 hindsight skill，再把这些 skill 对
行为概率的影响蒸馏回普通策略。这样，决策能力和轨迹分析能力随训练共同演化，
最终得到一个推理时不依赖额外上下文的 agent policy。
