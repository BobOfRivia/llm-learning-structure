# 11.4 Reasoning 训练（RLVR / GRPO / DAPO / GSPO / 推理蒸馏）

[← 返回框架](../../README.md) · [📎 materials.md → §11.4](../../materials.md)

---

## 〇、本节回答什么

> o1 / R1 这条 "reasoning model" 路线到底训了什么？RLVR 和经典 RLHF 区别是什么？GRPO 怎么省掉 critic？DAPO / GSPO 又改了什么？推理能力可以蒸馏吗？

2024 年 9 月 OpenAI o1 发布、2025 年 1 月 DeepSeek-R1 复现并开源——**Reasoning RL** 成为后训练新主线。核心范式：

```
SFT base
  ↓
RLVR (Reinforcement Learning with Verifiable Rewards)
  ↓
长 CoT、能"想"几千 token 再答的模型
```

不同于 §11.2-3 的"偏好对齐"，这里的 reward 是**可程序化校验**（数学答案对/错、代码 unit test 通过/失败）。

---

## 一、为什么是 "Verifiable Reward"

经典 RLHF：reward 来自 RM（学习的）→ 易被 hack。

Reasoning 任务有天然 ground-truth：
```
数学题:    答案是不是 42 ？   verifier(answer) → 1 or 0
代码题:    unit test 通过否 ？  pytest → pass/fail
逻辑题:    形式化验证          z3/lean → ok/fail
事实题:    检索 + 校验         retrieval verifier
```

→ Reward 是**布尔/稀疏标量**，不靠学习 RM、不会被 hack（除非 reward 设计有漏洞，比如做题作弊）。

这一类 reward 设定叫 **RLVR (RL with Verifiable Rewards)**——Tülu-3 (AllenAI 2024) 正式命名。

---

## 二、Reasoning RL 的本质

### 2.1 "让模型把思考显式写出来"

```
不会 reasoning 的模型:
  问题 → "答案是 X" (一蹴而就)

reasoning 模型:
  问题 → "<think>让我先...再...然后...</think> 答案是 X"
```

RL 训练的关键奖励信号：**长 CoT 通向正确答案 → 奖励**。
模型学到："多想几步，正确率更高" → 自发延长 CoT。

### 2.2 R1-zero 的现象级发现（2024）

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 报告：
- **从 base model（无 SFT）直接 RL**
- 仅靠 verifiable reward + GRPO
- 模型在数学/代码上**自发**出现：长 CoT、反思（"wait, let me reconsider"）、自我修正

→ "Aha moment"：reasoning 是 **RL 涌现** 出来的，不必先 SFT 教 think 格式。

### 2.3 完整 R1 pipeline

```
DeepSeek-V3-Base
   ↓  纯 RL (GRPO + verifiable)
DeepSeek-R1-Zero (可读性差但能推理)
   ↓  cold-start SFT (少量长 CoT 数据，让输出可读)
DeepSeek-R1-SFT
   ↓  RL (reasoning + helpfulness 混合 reward)
DeepSeek-R1 (生产形态)
   ↓  rejection sampling + SFT
DeepSeek-R1-Distill (小模型继承推理)
```

---

## 三、GRPO（Group Relative Policy Optimization）

DeepSeek 2024 提出，R1 训练主力算法。

### 3.1 动机：去掉 critic

PPO 要训 critic 估 V(s) → 多一个 70B 模型。
GRPO：用 **group baseline** 代替 critic。

### 3.2 算法

对每个 prompt $x$，**采样 G 条 response** $\{y_1, ..., y_G\}$（典型 G=4-64）：

1. 计算 reward $\{r_1, ..., r_G\}$
2. 组内归一化得到 advantage：

$$
A_i = \frac{r_i - \mathrm{mean}(r_1, ..., r_G)}{\mathrm{std}(r_1, ..., r_G)}
$$

3. PPO-clip 形式更新：

$$
\mathcal{L}_{\text{GRPO}} = - \mathbb{E}\!\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_t \min\!\big( \rho_t^i A_i, \mathrm{clip}(\rho_t^i, 1\!-\!\epsilon, 1\!+\!\epsilon) A_i \big) \right] + \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

其中 $\rho_t^i = \pi_\theta / \pi_{\theta_{\text{old}}}$。

### 3.3 GRPO 的好处

- **省 critic** = 省一半显存
- 同 prompt G 条之间天然 baseline，方差低
- 对 verifiable reward（0/1）极友好（同组有对有错→ 强信号）

### 3.4 GRPO 实战超参（R1 设定）

| 超参 | 值 |
|------|---|
| Group size G | 16-64 |
| Rollout batch | 256-1024 prompts |
| Clip ε | 0.2 |
| KL β | 0.001 - 0.01 (R1 用很小 KL) |
| LR | 1e-6 ~ 5e-6 |
| Max gen length | 8K-32K（reasoning 输出很长） |
| Temperature | 1.0 |

### 3.5 GRPO 注意点

- Reward 必须 normalize 到组内（不然不同 prompt 难度不可比）
- 组内若全对或全错 → advantage = 0，浪费 → 难度分桶采样
- Generation 长度爆炸（>32K）→ 必须 cap

---

## 四、DAPO（Decoupled Clip and Dynamic Sampling, 2025）

[ByteDance 2025](https://arxiv.org/abs/2503.14476)：GRPO 的工程改进。

### 4.1 改进 1：Clip-Higher（非对称 clip）

发现：PPO clip 对**低概率 token**（探索）过于压制，长 CoT 时尤甚。

解：把上下 clip 解耦：
$$
\mathrm{clip}(\rho, 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}})
$$
$\epsilon_{\text{high}} > \epsilon_{\text{low}}$（典型 0.28 vs 0.2）→ 允许 policy 更激进地"放大新模式"。

### 4.2 改进 2：Dynamic Sampling

如果一个 prompt 的 G 条全对或全错 → advantage 全 0 → 浪费 rollout。
DAPO 动态过滤：**只保留 group 内有 reward 多样性的 prompt**，重新采样直到拿到有效样本。

### 4.3 改进 3：Token-Level Loss

GRPO 对每条 response 取 mean log-prob → 长 response 被稀释。
DAPO 直接对**所有 token** 求和（"flat token-level loss"）→ 长 CoT token 权重不被压。

### 4.4 改进 4：Overlong Reward Shaping

response 截断到 max-len 时，简单给 0 不太合理（被截断 ≠ 错）。
DAPO 给"软惩罚"：奖励 = 1 - length/max_length 之类，鼓励适度长度。

### 4.5 DAPO 工业实战

ByteDance 用 DAPO 训出 Qwen-32B-DAPO，在 AIME 上超过 R1-Zero。完整开源。

---

## 五、GSPO（Generative Sequence Policy Optimization，2025）

[Qwen 2025](https://arxiv.org/abs/2507.18071)：GRPO 的另一支改进，重写 importance ratio。

### 5.1 问题：token-level ratio 与 sequence-level reward 不匹配

GRPO 每个 token 上算 $\rho_t = \pi/\pi_{\text{old}}$，但 reward 是 sequence-level → 信号粒度不一致。

### 5.2 GSPO 的改动

用 **sequence-level importance ratio**：

$$
\rho_{\text{seq}} = \exp\!\left( \frac{1}{|y|} \sum_t \log \frac{\pi_\theta(y_t | \cdot)}{\pi_{\theta_{\text{old}}}(y_t | \cdot)} \right)
$$

然后整条 sequence 共享一个 clip 与 advantage。

### 5.3 GSPO 的好处

- 与 sequence-level reward 完美匹配
- Long-CoT 训练更稳定（变长 KV 缓存 + token-level ratio 易 NaN）
- Qwen-32B 后期训练（reasoning 阶段）改用 GSPO

→ **2025 趋势**：GSPO 在长 CoT、agentic 场景逐渐取代 GRPO。

---

## 六、其他 reasoning RL 算法

| 算法 | 来源 | 关键改进 |
|------|------|---------|
| **RLOO** | Cohere 2024 | Leave-One-Out baseline，类似 GRPO 单条版 |
| **REINFORCE++** | 2024 | 无 critic 的 REINFORCE，多个稳定技巧 |
| **GRPO++** | 社区 | GRPO + entropy bonus + advantage shaping |
| **DAPO** | ByteDance 2025 | 见 §四 |
| **GSPO** | Qwen 2025 | 见 §五 |
| **VAPO** | 2025 | Variance-aware PO |
| **VC-PPO** | 2024 | Value-Calibrated PPO，专攻 long CoT |
| **PRM** + RL | OpenAI 2023+ | Process Reward Model：每步打分而非只看终值 |

---

## 七、Process Reward Model（PRM）

**Outcome Reward Model (ORM)**：只看最终答案对错。
**Process Reward Model (PRM)**：给 CoT 每一步打分。

```
Q: 求 2x + 3 = 7 的 x
Step 1: 移项 → 2x = 4    [PRM: 0.95]
Step 2: 除 2 → x = 2     [PRM: 0.98]
答案: x = 2              [ORM: 1.0]
```

### 7.1 PRM 数据获取

- **人工标**（昂贵）：[PRM800K (OpenAI 2023)](https://arxiv.org/abs/2305.20050)
- **自动标**：Math-Shepherd（rollout 多条，统计完成率回填 step 标签）
- **MCTS 标**：树搜索探索每步价值

### 7.2 PRM 用法

1. **训练 reward**：每 step reward + 末尾 outcome reward
2. **Best-of-N reranking**：推理时 sample N 条，PRM 选最佳
3. **MCTS guidance**：PRM 当 value head

### 7.3 R1 vs o1 在 PRM 上的分歧

- **o1 (OpenAI)**：据称用 PRM + MCTS（PRM800K 之后）
- **DeepSeek-R1**：**否定** PRM 路线——发现 PRM 易被 hack 且训练复杂；R1 用纯 outcome reward + GRPO 反而更好

→ 2025 工业共识**逐渐偏向 outcome-only**（GRPO 简化 + verifiable）。

---

## 八、推理蒸馏（R1-Distill 范式）

R1 现象级发现：**大 reasoning 模型生成的长 CoT 可被小模型 SFT 学到**。

### 8.1 R1-Distill 做法

```
1. 用 DeepSeek-R1 (671B MoE) 生成 80w 条 (问题, 长 CoT 解答)
2. 在 Qwen-7B/14B/32B base 上做 SFT（不需 RL）
3. 得到 R1-Distill-Qwen-{7B/14B/32B}
```

结果：
- R1-Distill-32B 在 AIME 上 **接近 o1-mini**（小模型史上第一次）
- 不需要任何 RL，纯 SFT
- 训练时间 / 算力比从头 RL 少 1-2 个量级

### 8.2 为什么蒸馏比 RL 还有效（在小模型上）

- 小模型 RL 信号太稀疏（reward 多为 0）
- 但模仿大模型已找到的"好路径"，等于把 RL 探索过的 trajectory 直接灌进去
- → **大模型 RL → 小模型 distill** 成为 2025 标准 pipeline

详见 §10.5 知识蒸馏的 R1-Distill 部分。

### 8.3 与思维链蒸馏的对比

- 经典 CoT distill (Orca, 2023)：让小模型学短 CoT，效果有限
- R1-Distill：长 CoT（reasoning trace 几千 token） + outcome 同时正确 → 质变

---

## 九、Reasoning RL 工程实战

### 9.1 Verifier 设计

```
数学题:   sympy 等价校验（不只字符串匹配）
代码题:   sandbox 执行 + unit test
通用 QA: 自动 grading prompt + LLM-as-judge
逻辑:    Z3 / Lean / Prover9
```

Verifier 的**误判率**直接是 RL 的 noise floor → 必须可靠。

### 9.2 数据集（公开）

- **MATH-500 / AIME / OlympiadBench**：竞赛级数学
- **GSM8K**：小学数学（已饱和）
- **CodeContests / LiveCodeBench**：代码
- **OpenR1-Math** / **AceCoder**：开源 RL 数据
- **Skywork-OR1** / **RUC-OpenR1**：完整 R1 复现链

### 9.3 训练成本

- R1 完整训练：估计**数百万美元** GPU（DeepSeek 未公开精确数）
- R1-Distill-Qwen-7B：单 8×H100 几天即可
- 中小团队普遍走 distill 路线

---

## 十、关键问答

**Q1**：为什么 RLVR 比 RLHF 稳？
- Reward 是程序化的 0/1，没有 RM 被 hack 风险
- 缺点：只能用于"有 ground-truth 的任务"（数学/代码/形式逻辑）
- 通用 chat 仍需 RM 或 LLM-judge

**Q2**：GRPO 与 PPO 的差距？
- GRPO 无 critic、用 group baseline
- 显存省 ~30-40%、训练快 1.5-2×
- 推理任务上效果 ≥ PPO（R1 实测）
- 通用 chat 上略逊（baseline noise 大）

**Q3**：R1-Zero 真的没 SFT 吗？
- 论文报告"直接 base model + RL"
- 实际 base model 已有大量代码/数学 pretrain → 称为"无后训练 SFT"更准确
- 但确实没用 instruction-tuning SFT 步

**Q4**：长 CoT 输出会拖慢推理吗？
- 是。reasoning 模型输出 1k-32k thinking tokens
- 推理成本：thinking token 数 × token cost
- 部署需考虑：是否让用户看 think 内容 / latency 预算

**Q5**：PRM 还有用吗？
- 在 search-based 推理（MCTS、Best-of-N）里仍重要
- 在 RL 训练阶段，2025 工业主流是 **outcome-only**（GRPO）
- PRM 的训练成本与 hack 风险让大家暂时回避

**Q6**：DAPO 和 GRPO 我应该选哪个？
- 复现 R1 / 数学 RL：GRPO 是默认起点
- Long CoT 不稳 / 想刷 leaderboard：DAPO
- Qwen 系生态：GSPO

**Q7**：reasoning 能力可以蒸馏到 7B 吗？
- 可以，R1-Distill-Qwen-7B 在 MATH 上接近 o1-mini
- 但 1-3B 蒸馏效果显著下降（容量不足）
- 7B 是当前 reasoning distill 的甜点

---

## 十一、本节与其他节关系

```
§11.1 SFT
   ↓
§11.2 PPO    ─┐
§11.3 DPO    ─┴── 经典对齐
   ↓
§11.4 RLVR / GRPO (本节) ── reasoning 新主线
   ↓
§11.5 Agentic RL          ── 多轮工具调用
   ↓
§10.5 蒸馏 (R1-Distill)    ── 把大模型推理能力压到小模型
```

---

## 十二、参考资料

- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) ⭐⭐（reasoning RL 现象级）
- [DeepSeekMath — GRPO 原始论文 (2024)](https://arxiv.org/abs/2402.03300) ⭐⭐
- [OpenAI o1 system card](https://openai.com/index/learning-to-reason-with-llms/) ⭐
- [Let's Verify Step by Step — PRM (OpenAI 2023)](https://arxiv.org/abs/2305.20050) ⭐⭐
- [Tülu-3 — RLVR 命名 (AllenAI 2024)](https://arxiv.org/abs/2411.15124) ⭐⭐
- [DAPO (ByteDance 2025)](https://arxiv.org/abs/2503.14476) ⭐
- [GSPO (Qwen 2025)](https://arxiv.org/abs/2507.18071) ⭐
- [Math-Shepherd — 自动 PRM 数据 (2024)](https://arxiv.org/abs/2312.08935)
- [VinePPO / VC-PPO long CoT](https://arxiv.org/abs/2410.01679)
- [Skywork-OR1 (开源 R1 复现)](https://github.com/SkyworkAI/Skywork-OR1)
- [HuggingFace open-r1 项目](https://github.com/huggingface/open-r1) ⭐⭐
- [REINFORCE++ (2024)](https://arxiv.org/abs/2501.03262)
- [The Bitter Lesson Strikes Reasoning — Sutton blog 风 (Lilian Weng 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [Reward Hacking in RL (Lilian Weng 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) ⭐⭐
