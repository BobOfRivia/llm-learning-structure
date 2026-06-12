# 11.3 RLHF（RM + PPO）

[← 返回框架](../../README.md) · [📎 materials.md → §11.3](../../materials.md)

---

## 〇、本节回答什么

> RLHF 的三阶段流水到底在做什么？Reward Model 的损失为什么用 Bradley-Terry sigmoid？PPO 从 TRPO 演化到 clipped surrogate 经过了什么取舍？token-level reward 是怎么从 sequence-level 拆出来的？KL 控制、reward hacking 到底是什么机制？2024+ PPO 是否被 DPO/GRPO 取代了？

RLHF = **从人类偏好反向恢复奖励函数 → 用 RL 把策略优化到该奖励上，同时用 KL 锚定到 SFT 不让模型崩**。

```
SFT model (§11.1)
    ↓                            ┌──── 偏好数据 (Bradley-Terry) ────┐
Reward Model (RM)  ←──────────── │                                  │
    ↓                            ↓                                  │
PPO loop (policy ← clipped surrogate，KL anchor 到 ref)            │
    ↓                                                               │
aligned model                                                       │
```

本节按这个顺序展开：

1. **为什么不直接 SFT**——比较信号比示范信号容易标、信息密度高（一段概率论的论证）。
2. **Reward Model**：Bradley-Terry 假设、为什么用 sigmoid、length bias / reward hacking 的根因。
3. **PPO**：从 policy gradient → TRPO → clipped surrogate 的演化，KL 控制为什么必要，token-level reward 怎么拆。
4. **工程**：4 模型显存预算细算、训练循环每一步、典型坑。
5. **2024+ 现状**：PPO 已不是主流的偏好对齐方法，但仍是 reasoning RL（GRPO）的母算法，理解 PPO 是理解后续所有 on-policy RL 的前提。

---

## 一、为什么 SFT 不够：示范信号 vs 比较信号

### 1.1 三种信号的信息密度

- **示范（demonstration）**：人写一条"完美"回答。SFT 用这种。
- **比较（comparison）**：人在两条回答里选好的。RLHF 用这种。
- **打分（rating）**：人给单条回答 1-5 星。HelpSteer / 早期工作用。

为什么 OpenAI 在 [InstructGPT](https://arxiv.org/abs/2203.02155) 选择比较：**比较的标注一致性远高于示范和打分**，单 token 信息密度更高。

形式化：假设 ground-truth quality $q(y) \in \mathbb{R}$，标注员观察值带噪声 $\hat q(y) = q(y) + \epsilon$（$\epsilon$ 标注员个体偏差）。

- 打分：标注员直接给 $\hat q(y)$，**偏差 $\epsilon$ 直接进入信号**。
- 比较：标注员给 $\text{sign}(\hat q(y_a) - \hat q(y_b))$，**只要 $|q(y_a) - q(y_b)|$ 比噪声大就标对**，且 $\epsilon$ 的"绝对偏移"被减号消掉。

→ 比较对噪声鲁棒得多。这是为什么"两个标注员独立给单条打分一致性 ~50%，给两条比较一致性 ~75%"。

### 1.2 SFT 表达不出来的三类信号

SFT 教"复述示范"，但表达不出：

- "A 比 B 好"这种**相对比较**信号（即使 A、B 都不完美）；
- "不要说脏话"这种**禁止**信号（SFT 只能给正例，不能给反例）；
- "更礼貌、更详细、更诚实"这种**模糊偏好**（很难写出完美示范）。

RLHF 的核心思想：

> **人类不擅长写正确答案，但擅长比较哪个更好**——把比较信号转化成奖励函数，再用 RL 把策略往奖励方向推。

---

## 二、RLHF 三阶段流水（InstructGPT 2022 范式）

```
Stage 1: SFT (§11.1)
   base → SFT model π^SFT

Stage 2: Reward Modeling
   收集偏好 (x, y_w, y_l) → 训 RM r_φ
   要求: r_φ(x, y_w) > r_φ(x, y_l)

Stage 3: RL (PPO)
   π_θ ← argmax  E[r_φ(x, y) - β·KL(π_θ || π^SFT)]
        s.t.  π_θ 不偏离 π^SFT 过远（KL 锚定）
```

三个阶段顺序依赖：SFT 给一个"差不多能用"的起点；RM 从偏好数据学到奖励函数；PPO 把 policy 优化到该奖励。任何一阶段失败都会传播——RM 错了 PPO 就在错的奖励上爬山。

---

## 三、Reward Model

### 3.1 偏好数据形态

```
prompt:    "解释牛顿第二定律"
chosen:    "F=ma，意思是物体加速度与受力成正比、与质量成反比 ..."   ← 标为更好
rejected:  "牛顿是个很厉害的物理学家，他提出了很多定律 ..."          ← 标为较差
```

数据来源：

- **公开**：Anthropic HH-RLHF（160k 条 helpful + harmless）、UltraFeedback（GPT-4 生成的合成偏好）、HelpSteer2（NVIDIA 开源，多维度）。
- **合成**：用更强模型当 "AI annotator" 打分（详见 §11.7 RLAIF）。
- **真实日志**：用户 thumbs up/down 信号（噪声大但分布最真）。

### 3.2 Bradley-Terry 模型：从偏好概率推 reward

我们想从一堆 "A 比 B 好" 的标注里反推一个奖励函数 $r$。**Bradley-Terry 模型**是这种"成对比较"问题的经典假设：

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma\big( r(x, y_w) - r(x, y_l) \big)
$$

其中 $\sigma(z) = 1/(1 + e^{-z})$ 是 sigmoid。

**为什么用 sigmoid（指数族）？** 三种等价视角：

1. **公理化**（Bradley & Terry 1952）：假设偏好概率只依赖 $r$ 的差，且满足传递性 + scale invariance，唯一满足的形式就是 logistic。
2. **信息论**：sigmoid 是 binary 分类的最大熵分布——在只知道一阶矩信息时，最不强加假设。
3. **统计物理**：与 Boltzmann 分布同构，$r$ 像"负能量"，温度参数被吸收进 reward scale。

### 3.3 RM 损失

给定偏好数据 $\mathcal{D} = \{(x_i, y_w^i, y_l^i)\}$，最大化对数似然：

$$
\boxed{
\mathcal{L}_{\text{RM}}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]
}
$$

这就是一个 binary cross-entropy 损失，目标变量是"$y_w$ 比 $y_l$ 好的概率应该接近 1"。

**注意**：$r_\phi$ 是 sequence-level 标量（一段输出一个分数），不是 token-level。

### 3.4 RM 架构

```
input:  [prompt + response]
        ↓
   transformer backbone   ← 用 SFT 模型初始化（或更大）
        ↓
   取最后一层最后一个 <|eos|> 位置的 hidden state
        ↓
   linear projection → scalar reward r_φ(x, y) ∈ ℝ
```

设计细节：

- **初始化**：用 SFT model（或更大模型），把 LM head 换成 reward head（一个 `Linear(d, 1)`）。这样 RM 已经"懂任务"，只需学打分。
- **冻结策略**：通常 freeze embedding + 前若干层，只 fine-tune 后半段 + reward head。更快、过拟合更少。
- **多维 reward**（HelpSteer2 / Nemotron）：输出 5 维（helpfulness / correctness / coherence / complexity / verbosity），训练时每维用独立标注，inference 时加权融合。

### 3.5 RM 训练超参

| 超参 | 推荐 |
|---|---|
| LR | 1e-6 ~ 5e-6 (比 SFT 小一个量级) |
| Epochs | 1（多了过拟合到注释员偏差） |
| Batch | 32-128 pairs |
| Loss type | Bradley-Terry sigmoid loss |
| Eval metric | Accuracy on holdout pairs（>65% 实用，>72% 优秀） |

RM accuracy 的现实区间：人类标注一致性约 70-75%（同一对偏好两个人独立标的一致率），所以 RM 准确率达到 ~72% 就接近"人类水平"。再往上 hard ceiling 是噪声。

### 3.6 RM 的失败模式与机制

**(1) Length bias**

**现象**：长 response 几乎总被 RM 打高分。

**机制**：

- 训练数据里 chosen response 平均略长（标注员"看起来认真"的偏见）；
- RM 学到一个简单的代理特征——长度 → 高分（少量信号即可被捕获，因为 length 是非常容易提取的特征）；
- PPO 阶段 policy 立刻发现"输出更长 → reward 更高"，于是输出越来越长。

**解药**：

- **Length-debias**：把训练数据按长度对齐（chosen 和 rejected 同长度桶）。
- **Length penalty in PPO**：reward 减去 $\alpha \cdot |y|$。
- **Two-stage RM**：先训一个 RM 预测 length，从 main RM 中"减掉"长度方向的预测。

**(2) Style bias**

**现象**：RM 偏爱有 markdown / bullet point / "Sure!" 开头的回答。

**机制**：同样是简单特征绑架——这些表面 style 是数据里高频共现的，RM 学到它们当代理。

**(3) Reward hacking（PPO 阶段才出现）**

**现象**：PPO 跑久后，policy 输出怪异格式但 RM 给极高分。

**机制**：Goodhart's Law——"任何被当成 metric 的 measure 都不再是好 measure"。RM 是 quality 的代理，policy 找到 RM 的漏洞（OOD 输入下 RM 不可靠的方向）来刷分。这是 RL with learned reward 的根本难题。

**根本解药**：

- **KL anchor**（§4 详述）：限制 policy 不能离 SFT 太远；
- **RM ensemble**：多个独立初始化的 RM 平均，OOD 不一致就当作 noise；
- **iterative RM**：每轮 PPO 后用新数据重训 RM，让它跟上 policy 分布；
- **Reward 上限**：clip reward 到训练分布的 99 分位。

**(4) Annotator bias**

**现象**：RM 学到的偏好其实是少数活跃注释员的偏好。

**机制**：偏好标注成本高，少数标注员标了大量数据，他们的个人风格主导了 RM。

**解药**：多注释员 + 共识过滤 + demographic 平衡。

---

## 四、PPO（Proximal Policy Optimization）

### 4.1 从 policy gradient 到 PPO 的演化

**Vanilla policy gradient**：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a) \big]
$$

问题：每次更新都需要新的 rollout（on-policy），样本效率极低。

**Importance sampling**：用老策略 $\pi_{\theta_{\text{old}}}$ 采集数据，新策略上算梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_{\theta_{\text{old}}}}\Big[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a) \Big]
$$

代价：当 $\pi_\theta$ 偏离 $\pi_{\theta_{\text{old}}}$ 太远，重要性比 $\rho = \pi_\theta / \pi_{\theta_{\text{old}}}$ 方差爆炸。

**TRPO（Trust Region Policy Optimization, Schulman 2015）**：加 KL constraint：

$$
\max_\theta \mathbb{E}\big[ \rho \cdot A \big] \quad \text{s.t.} \quad \mathbb{E}\big[ \text{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \big] \le \delta
$$

解法用 conjugate gradient + line search，**数学优雅但工程复杂**。

**PPO（Schulman 2017）**：用 **clipped surrogate** 近似 trust region：

$$
\boxed{
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( \rho_t \cdot A_t, \, \text{clip}(\rho_t, 1\!-\!\epsilon, 1\!+\!\epsilon) \cdot A_t \big) \Big]
}
$$

直觉：

- 当 $A_t > 0$（action 好）：希望 $\rho_t$ 增大，但 clip 到 $1 + \epsilon$ 后梯度变 0——不再激励"过激进的增大"；
- 当 $A_t < 0$（action 差）：希望 $\rho_t$ 减小，clip 到 $1 - \epsilon$ 后梯度变 0；
- **min** 操作保证 clip 永远朝"更保守"那一侧生效——这是 "pessimistic" 替身函数。

$\epsilon$ 典型 0.2，工程上是软 trust region。

→ 工程简单（只需 clip + 普通 SGD），效果接近 TRPO。**这是 PPO 取胜的关键**。

### 4.2 RLHF 的 RL 目标

$$
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \big[ r_\phi(x, y) \big] - \beta \cdot \mathbb{E}_x\big[ D_{\text{KL}}\big( \pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x) \big) \big]
$$

- $r_\phi$：reward model 输出（sequence-level）；
- $\pi_{\text{ref}}$：SFT 模型，**完全冻结**，作为 KL anchor；
- $\beta$：KL 系数（典型 0.01-0.1）。

**为什么需要 KL 项？** 没有 KL，PPO 会快速找到 RM 的 OOD 漏洞（reward hacking）。KL 把 policy 钉在"SFT 附近"，hacking 路径被剪掉。形式化：

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp(r_\phi(x, y) / \beta)
$$

（这是带 KL 约束的最优策略，下一节 DPO 推导会用到。）

$\beta$ 越大 → policy 越贴近 ref；$\beta \to 0$ → 纯 reward 最大化（崩）。

### 4.3 Token-level reward 分解

RM 是 sequence-level（整段一个分数），但 PPO 的 $A_t$ 是 token-level——怎么把 sequence reward 摊到每个 token？

InstructGPT 的做法：

$$
R_t = - \beta \cdot \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} + \mathbb{1}[t = T] \cdot r_\phi(x, y)
$$

- **每个 token 都有 KL 项**（这是 per-token 的 KL 估计，工程上叫 "token-level KL penalty"）；
- **只有最后一个 token 拿到 RM reward**（稀疏 reward）。

为什么这种拆法在数学上"对的"？因为：

- KL 的 per-token 拆法是恒等式（链式法则）：$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \sum_t \mathbb{E}[\log \pi_\theta / \pi_{\text{ref}}]$。
- Sequence reward 只能在 sequence 结束后给（中间没有 ground-truth），所以放最后一个 token。
- 用 GAE 算 advantage 时，最后的 reward 会沿 trajectory 向前传播——前面的 token "拿到"自己的贡献。

### 4.4 GAE（Generalized Advantage Estimation）

PPO 需要 token-level advantage $A_t$。直接用蒙特卡洛 return $G_t = \sum_{k \ge t} \gamma^{k-t} R_k$ 方差大；用 $V(s_t)$ 估计偏差大。GAE（Schulman 2015）做权衡：

$$
A_t^{\text{GAE}}(\lambda) = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \text{where} \quad \delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)
$$

- $\lambda \to 0$：纯 TD（低方差高偏差）；
- $\lambda \to 1$：蒙特卡洛（高方差低偏差）；
- 典型 $\lambda = 0.95$。

$V(s_t)$ 由 **critic**（value head）给出，critic 与 actor 并行训练。

### 4.5 PPO 的四个模型

```
π_θ         actor（策略，要训练）            ← 与 ref 同初始化
π_ref       reference（KL anchor，冻结）     ← SFT model
V_ψ         critic（value head，估算 V_t）    ← 通常用 RM 初始化
r_φ         reward model（冻结）
```

→ **同时驻留 4 个模型**——这是 RLHF 工程最贵的部分。70B PPO 在单机 8×H100 (640 GB) 上几乎跑不动，需要：

- 16-32 张 H100 + ZeRO-3；
- 或 actor + critic 共享 backbone（省一半）；
- 或 ref offload 到 CPU（带宽痛）；
- 或用 GRPO / DPO 等省 critic 的算法。

显存预算（70B BF16 mixed precision，含 master + grad + Adam state）：

```
actor:       70B × 16 bytes/param ≈ 1120 GB（参数 + master + grad + Adam state）
critic:      ≈ 1120 GB
reference:   140 GB（只前向，BF16）
reward:      140 GB（只前向）
activation:  500-1000 GB（rollout 阶段，long gen）
            ────────────
            ~ 3000-3500 GB total
```

需要 40+ 张 H100 才舒服。所以工业 PPO 训 70B 是个"非平凡的工程项目"。

### 4.6 PPO 训练循环

```python
for iter in range(N_iter):
    # 1. rollout（用 π_θ 生成 K 条 response / prompt batch）
    prompts = sample_prompts(M)
    responses = π_θ.generate(prompts, max_len=L)

    # 2. score（每条算 reward）
    rewards = r_φ(prompts, responses)            # sequence-level
    logprobs_θ = π_θ.logprob(responses)          # token-level
    logprobs_ref = π_ref.logprob(responses)
    values = V_ψ(prompts, responses)             # token-level

    # 3. compute token-level R_t with KL penalty
    R_t = - β * (logprobs_θ - logprobs_ref)
    R_T += rewards   # 最后一个 token 加 sequence reward

    # 4. GAE
    advantages, returns = compute_gae(R_t, values, λ=0.95, γ=1.0)
    advantages = normalize(advantages)            # 关键！

    # 5. multi-epoch PPO update on the rollout batch
    for epoch in range(4):
        for minibatch in split(prompts, responses, ...):
            ratio = exp(π_θ.logprob(...) - logprobs_θ_old)
            actor_loss = - min(ratio * A, clip(ratio, 1-ε, 1+ε) * A).mean()
            value_loss = (V_ψ(...) - returns).pow(2).mean()
            entropy_bonus = π_θ.entropy().mean()
            loss = actor_loss + c_v * value_loss - c_e * entropy_bonus
            loss.backward(); optimizer.step()
```

每个 RL "iter" 是一整套 rollout-and-update cycle，通常 N_iter ≈ 200-1000。

---

## 五、PPO 实战工程

### 5.1 KL 控制：β 不是固定常数

**KL 太低**（policy 离 ref 太远）：

- 现象：质量崩，输出胡乱化、emoji 满天飞；
- 通常是 $\beta$ 太小或忘了加 KL 项。

**KL 太高**（policy 几乎不动）：

- 现象：reward 不涨，policy ≈ SFT；
- 通常是 $\beta$ 太大。

工程经验：

- **静态 $\beta$**：0.01-0.05 起步；
- **自适应 KL controller**（PPO 原论文）：

  $$
  \beta \leftarrow \begin{cases} \beta \cdot 1.5 & \text{if KL} > 1.5 \cdot \text{KL}_{\text{target}} \\ \beta / 1.5 & \text{if KL} < 0.5 \cdot \text{KL}_{\text{target}} \\ \beta & \text{otherwise} \end{cases}
  $$
  
  目标 KL 一般 5-15 nats（per sequence）。
  
- **截断 KL（Anthropic 流派）**：只算正向 KL，避免反向爆炸。

### 5.2 Reward Hacking 防御

监控指标（出现以下任一即怀疑 reward hacking）：

- **生成长度突然飙升**（length-biased RM 被刷）；
- **重复 n-gram 飙升**（policy 学会刷套话）；
- **熵骤降**（policy 退化到几乎确定性输出）；
- **某些 token 频率异常**（"Sure!"、"Certainly!"、"In conclusion" 飙升）；
- **holdout RM 与 main RM 打分发散**（main RM 给高分，holdout 给低分）。

防御组合拳：

1. **KL 控制要严**（β 调大 / 自适应 controller）；
2. **Length normalization**：reward 除以长度，或减去长度先验；
3. **Reward ensemble**：3-5 个独立 RM，取最小值（pessimistic）；
4. **Early stopping**：固定每 N step 在 holdout 上评测，发现退化立刻停；
5. **人工抽检**：每 100 step 抽 50 条 generation 人眼看一下。

### 5.3 Critic 初始化与设计

经典做法：用 RM 复制一份当 critic，把 reward head 换成 value head：

```
RM backbone  + reward head (Linear(d, 1))     ← 训完冻结
       ↓
Critic backbone + value head (Linear(d, 1))   ← 与 actor 同步训练
```

为什么用 RM 初始化 critic？因为 RM 已经"懂偏好"，能给出比随机初始化稳定得多的 value 估计。

简化版本（省一半显存）：actor 与 critic **共享 backbone**，只是 head 不同：

```
shared transformer
       ↓
     ┌─┴─┐
   actor critic
```

代价：actor 和 critic 的训练目标不同，共享会互相干扰，效果略逊但工程简单。Llama-2 / Tülu-3 走分离 critic。

### 5.4 训练超参（PPO RLHF）

| 超参 | 典型值 | 说明 |
|---|---|---|
| Rollout batch | 512-2048 prompts | 越大越稳 |
| Multi-epoch (PPO) | 2-4 | 多了 policy 漂离 rollout 分布 |
| Mini-batch | 64-128 | grad accumulation |
| Actor LR | 1e-6 ~ 5e-6 | 比 SFT 小一个量级 |
| Critic LR | 5e-6 ~ 1e-5 | 比 actor 略大 |
| Clip $\epsilon$ | 0.1 - 0.2 | 0.2 是经典默认 |
| GAE $\lambda$ | 0.95 | |
| Discount $\gamma$ | 1.0 | sequence reward 一次性给，不折扣 |
| KL $\beta$ | 0.01-0.05 | 配合 adaptive controller |
| Entropy bonus $c_e$ | 0.0-0.01 | 通常 0（LLM 自带高熵） |
| Value loss coef $c_v$ | 0.5-1.0 | |
| Generation temperature | 0.7-1.0 | 太低没探索 |
| Max gen length | 1024-4096 | |
| Grad clip | 1.0 | 严，PPO 比 SFT 更易 NaN |

### 5.5 PPO 的 6 个工程坑

1. **采样和训练数据分布漂移**：rollout 后第 2-4 个 PPO epoch 已经偏离原分布 → multi-epoch 不要 > 4。
2. **Advantage 必须 normalize**：`(A - mean) / (std + 1e-8)`，否则方差炸。
3. **梯度范围异常**：PPO 比 SFT 更易 NaN，grad clip 必须严（1.0）。
4. **Generation 长度不稳定**：先长后短或反之，要监控。
5. **EOS 不出现**：用截断 + EOS reward shaping。
6. **训练发散**：90% 是 KL 没控好 + reward hacking。Loss 看不出来，要看 KL/reward/length 三联监控。

### 5.6 PPO 训练监控面板

健康的 PPO 训练应该看到：

```
reward:    缓慢上升（每 100 step ~ +0.1）
KL:        在 target 附近震荡（不超过 2× target）
length:    在合理范围（不持续上升）
entropy:   缓慢下降（policy 在收敛），但不应趋于 0
value loss: 缓慢下降
clip frac: 10-30%（< 5% clip 没作用；> 50% 步长太大）
```

如果你看到 reward 飙升但 length 也飙升、KL 突然爆炸——那不是模型变好了，那是 hacking。

---

## 六、PPO 之外的 on-policy RL（2024-2025 演进）

PPO 工程太重 → 涌现一系列**轻量化变体**：

| 方法 | 关键改进 | 备注 |
|---|---|---|
| **REINFORCE++** | 去掉 critic，用 batch baseline + 多个稳定技巧 | 工程简单，2024 复兴 |
| **RLOO** (Cohere 2024) | Leave-One-Out baseline | 用同 prompt 其他 sample 的平均当 baseline |
| **GRPO** (DeepSeek 2024) | Group-relative advantage | R1 主力（详见 §11.5） |
| **GSPO** (Qwen 2025) | Sequence-level importance ratio | long CoT 更稳 |
| **DAPO** (ByteDance 2025) | Decoupled clip + dynamic sampling | reasoning 实战 |
| **DPO 家族** | 完全跳过 RL（offline） | 见 §11.4 |
| **Reference-free PPO** | 移除 ref model | 显存省，但需更强 KL 替代 |

**2025 工业现状**：

- 通用 RLHF（chat 偏好对齐）：**DPO / Iterative DPO** 占主流（OpenAI 内部据传仍 PPO，但开源界 DPO 主导）；
- Reasoning RL：**GRPO / DAPO / GSPO** 主流；
- Agentic RL：GRPO + 工具 trajectory；
- PPO 作为"母算法"仍是理解所有后续方法的前提。

---

## 七、关键问答

**Q1**：为什么 PPO 用 clip 而不是 trust region？
- TRPO 用 conjugate gradient + line search 实现 trust region，数学优雅但工程复杂、容易数值不稳。
- PPO clip 是"软 trust region"——梯度被 clip 限制在 $[1-\epsilon, 1+\epsilon]$ 区间，等价于隐式约束。
- 工程简单，效果接近 TRPO，所以 2017 后是 RL 默认。

**Q2**：$\beta$（KL 系数）怎么调？
- 先用 0.01-0.02 起步，跑几百 step 看 per-sequence KL；
- KL > 15 nats → β 加大；KL < 5 → β 减小；
- 用 adaptive KL controller 最省心。

**Q3**：Reward Model 多大合适？
- 至少和 policy 同尺寸；最好更大（policy 7B + RM 13B）。
- 小 RM 容易被 hack（OOD 表现差）；大 RM 推理慢。
- Llama-3：使用与 policy 同等规模的 RM；DeepMind 报告过 "RM 比 policy 大效果更好"。

**Q4**：value head 一定要单独的 critic 吗？
- 经典 PPO：单独 critic（显存贵但稳）；
- 共享 backbone：省一半显存，效果略逊但工程简单；
- GRPO：彻底不要 critic（用 group baseline 替代）。

**Q5**：PPO vs DPO 现在怎么选？
- **PPO**：上限高、能 online iterate、需 RM + 4 模型显存；
- **DPO**：简单稳定、不需 RM、显存省一半、但只能用 offline 数据、上限略低；
- 大厂（OpenAI/Anthropic）据传仍 PPO，开源 / 中小团队 DPO 居多。

**Q6**：reward hacking 怎么早期识别？
- 监控 reward / length / KL / entropy / clip_frac 五个指标的联动；
- 设 holdout RM（不同初始化）打分对比；
- 每 100 step 抽 50 条 generation 人工抽检。

**Q7**：PPO 训完了还要做什么？
- **迭代式**：再 SFT (用 PPO 模型做 rejection sampling) → 再 PPO → 直到收敛。Llama-3 报告做了 6 轮 SFT-DPO 迭代。
- **接 reasoning RL**：用 verifiable reward（GRPO）替代或补充 RM。
- **接安全 alignment**：CAI / RLAIF（§11.7）。

**Q8**：为什么 token-level KL 是有效的近似？
- 严格的 sequence-level KL 需要对整个 $\pi_\theta(y|x)$ 积分，不可解。
- Token-level 拆分用链式法则：$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_y \sum_t [\log \pi_\theta(y_t|y_{<t}) - \log \pi_{\text{ref}}(y_t|y_{<t})]$。
- 把 sample 的 KL 当 reward penalty 加进每个 token，让 RL 框架自动处理。

**Q9**：PPO 训出来的模型为什么有时不如 SFT？
- 通常是 RM 质量太差（accuracy < 65%）；
- 或 KL 没控好，policy 过拟合到 RM 的 OOD 区域；
- 修：升级 RM 数据质量、加大 β、用 RM ensemble。

**Q10**：能不能跳过 RM 直接 PPO？
- 不能。PPO 必须有 reward signal，RM 是 RLHF 路径的 reward 来源。
- 替代：用 verifiable reward（程序化校验，见 §11.5）或 LLM-as-judge reward（见 §11.7），或者切换到 DPO（不需要 RM）。

---

## 八、本节与其他节关系

```
§11.1 SFT ──────────────→ §11.3 PPO (本节) ←── reward model
                                ↓
                          §11.4 DPO 家族（简化的 offline 方案）
                                ↓
                          §11.5 RLVR / GRPO（verifiable reward 替代 RM）
                                ↓
                          §11.6 Agentic RL（长程多轮）
                                ↓
                          §11.7 Constitutional / RLAIF（规模化 reward）
```

PPO 是后训练 RL 的母算法：DPO 是它的 offline 简化；GRPO 是它去掉 critic 的群体 baseline 版本；DAPO/GSPO 是 GRPO 的稳定性改进。理解 PPO 是理解后面所有变体的前提。

---

## 九、参考资料

**开山 / 原典**：
- [InstructGPT (Ouyang 2022)](https://arxiv.org/abs/2203.02155) ⭐⭐（RLHF 开山，三阶段范式）
- [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347) ⭐⭐
- [TRPO (Schulman 2015)](https://arxiv.org/abs/1502.05477)（PPO 的前身）
- [GAE (Schulman 2015)](https://arxiv.org/abs/1506.02438) ⭐
- [Bradley & Terry (1952)](https://www.jstor.org/stable/2334029)（成对比较模型原典）

**工业 RLHF**：
- [Anthropic HH-RLHF (Bai 2022)](https://arxiv.org/abs/2204.05862) ⭐
- [Llama-2 — RLHF §](https://arxiv.org/abs/2307.09288) ⭐⭐（工业最详尽的 PPO 描述）
- [Llama-3 — Iterative SFT-DPO §](https://arxiv.org/abs/2407.21783) ⭐⭐
- [HelpSteer2 — Multi-attribute RM (NVIDIA 2024)](https://arxiv.org/abs/2406.08673)
- [UltraFeedback (Cui 2023)](https://arxiv.org/abs/2310.01377)

**工程实战 / 复现**：
- [Secrets of RLHF in LLMs Part I (Zheng 2023)](https://arxiv.org/abs/2307.04964) ⭐（工程坑详解）
- [N Implementation Details of RLHF with PPO (Huang 2024)](https://arxiv.org/abs/2403.17031) ⭐⭐（37 个实战细节）
- [HuggingFace TRL 库](https://github.com/huggingface/trl) ⭐⭐（PPO/DPO/GRPO 标准实现）
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) ⭐（DeepSpeed + ray，大规模 PPO/GRPO）
- [VeRL (ByteDance)](https://github.com/volcengine/verl) ⭐（生产级 RLHF/RLVR）

**变体 / 演进**：
- [REINFORCE++ (Ahmadian 2024)](https://arxiv.org/abs/2402.14740) ⭐
- [RLOO (Ahmadian 2024)](https://arxiv.org/abs/2402.14740)
- [GRPO — DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) ⭐⭐
- [DPO (Rafailov 2023)](https://arxiv.org/abs/2305.18290) ⭐⭐

**理论 / 综述**：
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) ⭐
- [A Survey of RLHF (Kaufmann 2023)](https://arxiv.org/abs/2312.14925)
- [Lilian Weng — Reward Hacking (2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) ⭐⭐
- [Sebastian Raschka — RLHF deep dive](https://magazine.sebastianraschka.com/) ⭐
