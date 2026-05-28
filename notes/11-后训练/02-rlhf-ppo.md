# 11.2 RLHF（RM + PPO）

[← 返回框架](../../README.md) · [📎 materials.md → §11.2](../../materials.md)

---

## 〇、本节回答什么

> RLHF 的三阶段流水是什么？Reward Model 怎么训？为什么用 PPO 而不是直接监督？KL 控制、reward hacking 怎么处理？2024 后 PPO 还是不是主流？

RLHF = **从人类偏好反向恢复奖励函数 → 用 RL 把策略优化到该奖励上**。

```
SFT model (§11.1)
    ↓
Reward Model (RM) ← 偏好数据训练
    ↓
PPO loop (policy ← 策略梯度更新，受 KL 锚定到 ref)
    ↓
aligned model
```

---

## 一、为什么需要 RLHF（不是 SFT 就够了）

SFT 教**"复述示范"**，但无法表达：
- "A 比 B 好"这种**比较**信号
- "不要说脏话"这种**禁止**信号
- 人类常见**模糊偏好**（更礼貌、更详细、更诚实）

人类标比较 (pairwise) 比标完美回答容易得多：
```
SFT 数据:   prompt → 写一条最优 response (难标，标注一致性低)
RLHF 数据:  prompt → 两条 response，标"A好/B好/差不多" (易标)
```

→ RLHF 的核心思想：**人类不擅长写正确答案，但擅长比较哪个更好**。

---

## 二、RLHF 三阶段流水（InstructGPT 2022 范式）

```
Stage 1: SFT (见 §11.1)
   base → SFT model π^SFT

Stage 2: Reward Modeling
   收集偏好 (prompt, y_w, y_l) → 训练 RM r_φ
   r_φ(y_w | x) > r_φ(y_l | x)

Stage 3: RL (PPO)
   π_θ ← argmax  E[r_φ(y|x)] - β·KL(π_θ || π^SFT)
```

每个阶段都依赖上一个。

---

## 三、Reward Model 训练

### 3.1 偏好数据格式

```
prompt:    "解释牛顿第二定律"
chosen:    "F=ma，意思是..."     ← 人类选的
rejected:  "牛顿很厉害..."        ← 人类没选的
```

来源：
- **公开**：Anthropic HH-RLHF、UltraFeedback、HelpSteer2
- **合成**：用更强模型 (GPT-4) 当"AI annotator"打分

### 3.2 Bradley-Terry 模型

假设人类偏好概率与"reward 差"成 sigmoid 关系：

$$
P(y_w \succ y_l \mid x) = \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big)
$$

训练 RM 用对数似然（即逐对的 sigmoid loss）：

$$
\mathcal{L}_{\text{RM}} = - \mathbb{E}_{(x, y_w, y_l)} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]
$$

### 3.3 RM 架构

```
input:  [prompt + response]
        ↓
   transformer backbone (用 SFT 模型初始化)
        ↓
   last hidden state of <|eos|>
        ↓
   linear projection → scalar reward
```

- 通常用 SFT 同尺寸甚至更大的模型（reward 模型必须比 policy "懂")
- 输出单标量 reward（per-sequence）
- 也可输出 multi-objective（HelpSteer2：helpfulness/correctness/coherence/...）

### 3.4 RM 训练要点

- LR 比 SFT 小（1e-6 ~ 5e-6）
- 1 epoch 即可，多了会过拟合
- 重复 pair 数据 + diverse prompts 关键
- 评估指标：**accuracy on holdout pairs**（>60% 实用，>70% 优）

### 3.5 RM 的失败模式

| 问题 | 表现 | 应对 |
|------|------|------|
| 过拟合到注释员风格 | reward 与真实质量背离 | 多注释员 + 共识过滤 |
| Length bias | 偏爱长回答 | length normalization / debias |
| 分布偏移 | 训练 prompt 与 RL prompt 分布不同 | online RM / RM ensemble |
| Reward hacking | 模型找漏洞刷 reward | KL 控制 + reward shaping |

---

## 四、PPO（Proximal Policy Optimization）

### 4.1 目标函数

RL 目标：

$$
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \big[ r_\phi(x, y) \big] - \beta \cdot D_{\text{KL}}\big( \pi_\theta(\cdot|x) \,\big\|\, \pi_{\text{ref}}(\cdot|x) \big)
$$

- $r_\phi$：reward model 输出（per-sequence reward）
- $\pi_{\text{ref}}$：SFT 模型（**固定不动**），作为 KL anchor
- $\beta$：KL 系数（典型 0.01-0.1）

### 4.2 token-level reward 分解

per-sequence reward 拆到每个 token：

$$
R_t = - \beta \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} + \mathbb{1}[t = T] \cdot r_\phi(x, y)
$$

只有最后一个 token 拿到 RM reward，前面 token 只有 KL 惩罚。

### 4.3 PPO 核心：clipped surrogate

设重要性比率 $\rho_t = \pi_\theta / \pi_{\theta_{\text{old}}}$：

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \Big[ \min\big( \rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \big) \Big]
$$

- $A_t$：advantage（GAE 估计）
- $\epsilon$：clip 阈值（典型 0.2）
- clip 防止策略更新太激进

### 4.4 PPO 的四个模型

```
π_θ         actor（策略，要训练）            ← 与 ref 同初始化
π_ref       reference（KL anchor，冻结）     ← SFT model
V_ψ         critic（value head，估算 V(x,y_<t)）
r_φ         reward model（冻结）
```

→ **同时驻留 4 个模型** = RLHF 工程最贵的部分。70B 训练单机放不下，需要分布式。

### 4.5 PPO 循环

```
for iter in 1..N:
    1. rollout:   π_θ 生成 K 条回答 / 一个 batch
    2. score:     用 r_φ 给每条打分
    3. compute:   token-level R, advantage A (GAE)
    4. update:    multi-epoch（4-8 epoch）on rollout batch
                  loss = -PPO_clip(ρ, A) + c_v·V_loss - c_e·entropy
```

每个 RL "step" 实际上是一整个 rollout-and-update cycle。

---

## 五、PPO 实战工程

### 5.1 KL 控制

KL 太低：模型偏离 ref 太远，质量崩
KL 太高：进步太慢，几乎在 SFT 附近

工程经验：
- 静态 β：0.01 - 0.05（初始）
- 自适应 β：[Adaptive KL controller](https://arxiv.org/abs/1707.06347)（KL 超过 target 就调大 β）
- **目标 KL** 一般 5-15 nats（per sequence）

### 5.2 Reward Hacking

policy 找到 RM 的漏洞，刷 reward 但实际质量下降。

例子：
- 输出超长 → length-biased RM 给高分（实际啰嗦）
- 重复套话 → RM 觉得"礼貌"
- 输出格式化 markdown → RM 偏爱

→ **KL 控制是主防御**。其他：reward ensemble、长度归一化、reward clipping、early stopping。

### 5.3 Critic（V-head）初始化

通常用 RM **复制一份** + 把 reward head 换成 value head。
- 也可以从 SFT 直接加一个 linear head 训
- Critic 与 policy 同步训练

### 5.4 训练超参

| 超参 | 典型值 |
|------|--------|
| Rollout batch | 512-2048 prompts |
| Multi-epoch (PPO) | 2-4 |
| Mini-batch | 64-128 |
| LR (policy) | 1e-6 ~ 5e-6 |
| LR (critic) | 5e-6 ~ 1e-5 |
| Clip ε | 0.1 - 0.2 |
| GAE λ | 0.95 |
| γ (discount) | 1.0（一次性 reward，无需折扣） |
| KL β | 0.01 - 0.05 |
| Gen temperature | 0.7 - 1.0 |
| Max gen length | 1024 - 4096 |

### 5.5 工程开销

70B PPO 训练**显存预算**（大致）：
```
policy (BF16 mixed):    140 GB × 2 (master)
critic (BF16 mixed):    140 GB × 2
reference (BF16):       140 GB (frozen)
reward (BF16):          140 GB (frozen)
                        ─────────
                        ~ 700 GB + activation + KV cache
```
→ 单机 8×H100 (640 GB) 都吃力。常用方案：critic/RM 共享底座、PPO + LoRA、训练阶段 ref 卸载。

---

## 六、PPO 的工程坑

1. **采样和训练数据分布漂移**：rollout 后第 2-4 个 PPO epoch 已偏离原分布 → 不要 multi-epoch 太多
2. **Advantage 标准化**：必须做 (A - mean) / std，否则方差炸
3. **梯度范围异常**：PPO 比 SFT 更易 NaN，grad clip 必须严
4. **Generation 长度不稳定**：先长后短或先短后长，要监控
5. **EOS 不出现**：用截断 + EOS reward shaping
6. **训练发散**：90% 是 KL 没控好 + reward hacking

---

## 七、PPO 之外的 on-policy RL（2024-2025 演进）

PPO 太重 → 出现一系列**轻量化变体**：

| 方法 | 关键改进 | 备注 |
|------|---------|------|
| **REINFORCE++** | 去掉 critic，用 batch baseline | Anthropic / OpenAI 内部用 |
| **RLOO** (2024) | Leave-One-Out baseline | Cohere 提出 |
| **GRPO** (DeepSeek 2024) | Group-relative advantage | R1 主力（详见 §11.4） |
| **DPO 家族** | 完全跳过 RL（offline） | 见 §11.3 |
| **PPO + reference-free** | 移除 ref model | 省显存 |

**2025 趋势**：PPO 在通用 RLHF 退场，**GRPO** 在 reasoning RL 当道，**DPO** 在偏好对齐当道。

---

## 八、关键问答

**Q1**：为什么 PPO 用 clip 而不是 trust-region？
- TRPO 用 conjugate gradient + line search 实现 trust region，太复杂
- PPO clip 是简化的"软 trust region"——梯度被 clip 限制
- 工程简单，效果接近 TRPO

**Q2**：β（KL 系数）怎么调？
- 先小（0.01）跑两轮，看 KL 是否 < target（10 左右）
- 太低 → policy 偏离 ref 太远 → 加大
- 自适应控制器更稳，但需要 target KL 经验

**Q3**：reward model 多大合适？
- 至少和 policy 同尺寸；最好更大（比如 policy 7B + RM 13B）
- 小 RM 容易被 hack；大 RM 推理慢
- Llama-3：使用与 policy 同等规模的 RM

**Q4**：value head 一定要单独的 critic 吗？
- 经典 PPO：单独 critic（独立模型）
- 简化版：actor + value head 共享 backbone（省一半显存）
- 工业实战：分离 critic 更稳，但贵

**Q5**：PPO vs DPO 现在怎么选？
- PPO：上限高，可继续 online 改进，但工程复杂、需 RM
- DPO：简单稳定、不需 RM，但只能用 offline 数据
- 大厂（OpenAI/Anthropic）仍 PPO；中小团队 DPO 居多

**Q6**：reward hacking 怎么早期识别？
- 监控生成长度、重复率、特定 token 频率
- holdout RM（不同初始化）打分对比
- 人工抽检 100 条 generation

**Q7**：PPO 后还要做什么？
- 迭代式：再 SFT (rejection sampling) → 再 PPO → 直到收敛
- Llama-3：6 轮 SFT-DPO 迭代
- 也可接 reasoning RL（GRPO + verifiable reward）

---

## 九、本节与其他节关系

```
§11.1 SFT ──────────────→ §11.2 PPO (本节) ←── reward model
                                ↓
                          §11.3 DPO 家族 (简化的 offline 方案)
                                ↓
                          §11.4 RLVR / GRPO (verifiable reward 替代 RM)
                                ↓
                          §11.5 Agentic RL (长程多轮)
                                ↓
                          §11.6 Constitutional / RLAIF (规模化 reward)
```

---

## 十、参考资料

- [InstructGPT (Ouyang 2022)](https://arxiv.org/abs/2203.02155) ⭐⭐（RLHF 开山）
- [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347) ⭐⭐
- [Anthropic HH-RLHF (Bai 2022)](https://arxiv.org/abs/2204.05862) ⭐
- [Llama-2 — RLHF §](https://arxiv.org/abs/2307.09288) ⭐⭐（最详细的工业 PPO）
- [Secrets of RLHF in LLMs Part I (Zheng 2023)](https://arxiv.org/abs/2307.04964) ⭐（工程坑详解）
- [HuggingFace TRL 库](https://github.com/huggingface/trl) ⭐⭐（PPO/DPO/GRPO 标准实现）
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) ⭐
- [N Implementation Details of RLHF with PPO (Huang 2024)](https://arxiv.org/abs/2403.17031) ⭐（PPO 工程 37 个细节）
- [HelpSteer2 — Open RM dataset (NVIDIA 2024)](https://arxiv.org/abs/2406.08673)
- [REINFORCE++ (Ahmadian 2024)](https://arxiv.org/abs/2402.14740) ⭐（去 critic 替代）
- [RLOO (Ahmadian 2024)](https://arxiv.org/abs/2402.14740)
