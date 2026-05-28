# 11.3 DPO 家族（DPO / IPO / KTO / SimPO / ORPO）

[← 返回框架](../../README.md) · [📎 materials.md → §11.3](../../materials.md)

---

## 〇、本节回答什么

> 为什么 DPO 能在没有 reward model 和没有 RL 循环的情况下做"偏好对齐"？DPO 的核心数学是什么？IPO/KTO/SimPO/ORPO 各解决什么问题？2024-2025 怎么选？

DPO（Direct Preference Optimization）是 2023 末出现、2024 年成为**事实标准**的偏好对齐方法：**把 RLHF 简化成一个监督学习损失**，不要 RM、不要 PPO 循环、不要 critic。

```
RLHF (§11.2):  data → RM → PPO loop (4 个模型同时驻留)
DPO (本节):    data ───────→ 一个 BCE-like loss（只需 policy + ref）
```

---

## 一、DPO 的核心推导

### 1.1 RLHF 目标的闭式解

RLHF 目标：

$$
\max_\pi \mathbb{E}_{x, y \sim \pi}\big[ r(x, y) \big] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})
$$

这个目标有一个**已知的闭式最优解**（KL-regularized policy）：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left( \frac{1}{\beta} r(x, y) \right)
$$

反过来解出 reward：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

### 1.2 套入 Bradley-Terry 模型

人类偏好概率（§11.2 §3.2）：

$$
P(y_w \succ y_l | x) = \sigma\big( r(x, y_w) - r(x, y_l) \big)
$$

把上面 reward 表达式代入，**$\log Z(x)$ 在 $y_w$ 和 $y_l$ 上是同一个，消掉**：

$$
P(y_w \succ y_l | x) = \sigma\!\Bigg( \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \Bigg)
$$

### 1.3 DPO loss

把 $\pi^*$ 当成要学的 $\pi_\theta$，对偏好数据取负对数似然：

$$
\boxed{
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x, y_w, y_l)} \log \sigma\!\Bigg( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \Bigg)
}
$$

→ 看起来像 RLHF，**实际只是一个 BCE-like 监督 loss**，没有 RL、没有 RM。

### 1.4 直觉

让 $\pi_\theta$ 增大对 $y_w$ 的 log-prob（相对 ref）、减小对 $y_l$ 的 log-prob。
- $\beta$ 越大 → 越不偏离 ref
- 优化梯度直接调到 token-level logits

---

## 二、DPO 的实战工程

### 2.1 训练流程

```
1. 准备偏好数据 (x, y_w, y_l)
2. SFT 完成后 (π_ref = π_SFT)
3. 复制 π_ref 一份 → π_θ
4. 训练：
   loss = -log σ(β · (logp_θ(y_w) - logp_ref(y_w) - logp_θ(y_l) + logp_ref(y_l)))
5. 只需 forward 两个模型：π_θ (训) + π_ref (冻结)
```

### 2.2 显存预算 vs PPO

```
PPO (70B):    actor + ref + critic + RM = 4 × 70B ≈ 560 GB
DPO (70B):    actor + ref               = 2 × 70B ≈ 280 GB
```

→ DPO 显存约**省一半**，工程复杂度大幅下降。

### 2.3 超参

| 超参 | 推荐 | 备注 |
|------|------|------|
| β | 0.1 - 0.5 | 太小→偏离 ref；太大→学得慢 |
| LR | 5e-7 ~ 5e-6 | 比 SFT 小一个量级 |
| Epochs | 1 - 3 | 多了过拟合（reward 漂） |
| Batch size | 64 - 256 | |
| Schedule | Cosine | warmup 5-10% |

### 2.4 数据要求

- **chosen vs rejected 必须明显有差异**（否则 DPO 学不到信号）
- chosen 不一定完美，但要"明确比 rejected 好"
- 同一 prompt 配对，分布越一致越好
- 工业经验：**5-20w 条偏好数据**即有效

### 2.5 常用数据集

- UltraFeedback、UltraInteract
- HelpSteer2 → DPO pairs
- Anthropic HH-RLHF
- Synthetic：用 GPT-4 当判官生成 (chosen, rejected)

---

## 三、DPO 的问题与扩展

### 3.1 DPO 的几个已知缺陷

| 问题 | 表现 | 解 |
|------|------|---|
| **过拟合到 ref** | 输出贴近 SFT，几乎不前进 | β 调小 / 更多 epochs |
| **reduce both log-prob** | π(y_w) 和 π(y_l) 一起下降，只是后者降更多 | DPOP、Cal-DPO 修正 |
| **长度偏差** | 倾向长 response | SimPO 移除 ref 项 |
| **Distribution shift** | 训练完后 π_θ ≠ π_ref，loss 不再准 | iterative DPO、online DPO |
| **没用上同 prompt 多个 sample 的相对信息** | | RLOO / GRPO |

### 3.2 IPO（Identity Preference Optimization）

[Azar 2023](https://arxiv.org/abs/2310.12036) 指出 DPO 在偏好近乎确定时**过度拟合**（log σ 趋于线性）。

IPO 改用 **MSE 形式**：

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E} \Big[ \Big( h_\theta(x, y_w, y_l) - \frac{1}{2\beta} \Big)^2 \Big]
$$

其中 $h_\theta$ 是 DPO 的"reward 差"项。

→ 更不容易过拟合极端样本，但实战收益有限，没成主流。

### 3.3 KTO（Kahneman-Tversky Optimization）

[Ethayarajh 2024](https://arxiv.org/abs/2402.01306)：**不要 pair，只要 binary 标签**（"这条 good / bad"）。

$$
\mathcal{L}_{\text{KTO}}(x, y) = \begin{cases} 1 - \sigma(\beta \cdot r_\theta - z) & \text{if } y \text{ desirable} \\ 1 - \sigma(z - \beta \cdot r_\theta) & \text{if } y \text{ undesirable} \end{cases}
$$

其中 $r_\theta = \log \pi_\theta / \pi_{\text{ref}}$，$z$ 是 batch 内 KL 估计。

→ **数据标注简化**：用户 like/dislike 信号即可训。生产环境（用户反馈）首选。

### 3.4 SimPO（Simple Preference Optimization）

[Meng 2024](https://arxiv.org/abs/2405.14734)：**完全去掉 ref model**。

$$
\mathcal{L}_{\text{SimPO}} = - \log \sigma\!\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right)
$$

- 用 **average log-prob**（按长度归一化，去 length bias）
- 加 reward margin γ（强制 chosen 和 rejected 拉开距离）
- **不需 ref model** → 显存再省一半

**实战效果**：SimPO 在 AlpacaEval、Arena-Hard 上多次刷新榜，Llama-3-SimPO 是 2024 年开源界标杆。

### 3.5 ORPO（Odds Ratio Preference Optimization）

[Hong 2024](https://arxiv.org/abs/2403.07691)：**SFT 和偏好对齐合二为一**。

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}(y_w, y_l)
$$

其中 OR loss 让 $y_w$ 的 odds 高于 $y_l$ 的 odds。

- **省一阶段**：不需要先 SFT 再 DPO
- 数据用量少 → 适合小数据场景
- 缺点：上限略低于 SFT+DPO

### 3.6 cDPO / Cal-DPO / DPOP / RPO

更多衍生：
- **cDPO**：考虑标签噪声（人标 ~10% 翻车）
- **Cal-DPO**：calibrated DPO，修 log-prob 一起下降的问题
- **DPOP**：Distillation-aided DPO
- **RPO**：Reward-aware Preference Optimization（合并 RM signal）

---

## 四、Online / Iterative DPO

DPO 是 **offline**（数据固定），离线训完后 π_θ 已偏离 π_ref，再优化收益快速衰减。

### 4.1 Iterative DPO（Llama-3 范式）

```
loop:
    π_t ← SFT(initial) or π_{t-1}
    用 π_t 采样 K 条 → 用 RM 打分 → 取 best/worst 成 pair
    DPO 一轮 → π_{t+1}
```

每轮重新生成偏好数据 → policy 不会 stale。Llama-3 用了 6 轮。

### 4.2 Online DPO

边采样边训：
- 一个 batch 内：sample → score → DPO update
- 接近 PPO 但仍是 DPO loss
- 复杂度上升，效果接近 PPO

---

## 五、DPO 家族对比表

| 方法 | 需 ref | 需 RM | 数据类型 | 长度校正 | 备注 |
|------|--------|-------|---------|---------|------|
| DPO | ✅ | ❌ | pairs | ❌ | 经典基线 |
| IPO | ✅ | ❌ | pairs | ❌ | 防过拟合 |
| **KTO** | ✅ | ❌ | **binary** | ❌ | 标注成本最低 |
| **SimPO** | **❌** | ❌ | pairs | ✅ | 2024 SOTA |
| **ORPO** | ❌ | ❌ | pairs (+SFT) | ❌ | 与 SFT 合并 |
| cDPO | ✅ | ❌ | pairs (noisy) | ❌ | 抗噪 |
| Iterative DPO | ✅ | ✅ | pairs | ❌ | Llama-3 范式 |
| PPO (§11.2) | ✅ | ✅ | RM scalar | ❌ | 经典 RL |

---

## 六、什么时候选哪个？

```
有强 RM + 算力够   →  PPO（上限最高）/ Iterative DPO
偏好对（chosen/rejected）→  DPO 或 SimPO
仅有 like/dislike →  KTO
SFT + 偏好一起做   →  ORPO
追求 leaderboard  →  SimPO（2024 工业最常上分）
```

**2025 工业常见栈**：
- SFT → DPO（一次性 offline）→ 上线
- SFT → Iterative DPO（3-6 轮）→ 上线
- 复杂场景：SFT → DPO → RLVR/GRPO（§11.4）

---

## 七、关键问答

**Q1**：DPO 数学上等价于 RLHF 吗？
- **不严格等价**——RLHF 闭式解假设 π_θ 能逼近 π_ref · exp(r/β)/Z，对 LLM 表达能力的假设
- 实证上 DPO 与 PPO 在偏好任务上接近，但**PPO 在 hard task 上限更高**

**Q2**：为什么 DPO 训完后 chosen 和 rejected 的 log-prob 都降了？
- 因为模型整体输出分布在压缩（KL 拉向 ref 时被推开）
- DPO loss 只关心**差值**而不是绝对值
- Cal-DPO / DPOP 把绝对 log-prob 当正则项

**Q3**：β 怎么调？
- 0.1 起步，看 KL(π || π_ref)：太大（>50）→ β 加大；太小（<5）→ β 减小
- SimPO 的 β 一般更大（2.0-2.5）因为没有 ref 项

**Q4**：DPO 数据量需要多少？
- 1w 起步看见效果；5-20w 是常见区间
- 数据**质量 > 数量**——dirty pair 会严重伤害

**Q5**：DPO 训完后还需要 PPO 吗？
- 不一定。简单 chat 对齐 DPO 够了
- reasoning / 工具使用 / 多步任务 → 仍需 PPO 或 GRPO

**Q6**：SimPO 真的不需要 ref model 吗？
- 是的，loss 中没有 π_ref 项
- 显存省、训练快，但**对 SFT 初始模型质量更敏感**（没有 anchor）
- 适合 SFT 模型已比较稳的场景

**Q7**：iterative DPO 比 single-shot DPO 强多少？
- AlpacaEval-2 LC win rate：+5-10 个点（Llama-3 报告）
- 工程成本：3-6 倍
- 大厂标配，中小团队按需

---

## 八、本节与其他节关系

```
§11.1 SFT  ──→ §11.2 PPO         (上限高，工程贵)
                ↓
            §11.3 DPO 家族 (本节)
                ↓
            §11.4 RLVR / GRPO    (verifiable reward + DPO 思想结合)
```

DPO 家族吸取 RL 思想做"伪 RL"，GRPO 又借鉴 DPO 的简化 + REINFORCE++ 的 baseline。

---

## 九、参考资料

- [DPO (Rafailov 2023)](https://arxiv.org/abs/2305.18290) ⭐⭐（开山）
- [IPO (Azar 2023)](https://arxiv.org/abs/2310.12036)
- [KTO (Ethayarajh 2024)](https://arxiv.org/abs/2402.01306) ⭐
- [SimPO (Meng 2024)](https://arxiv.org/abs/2405.14734) ⭐⭐（2024 SOTA）
- [ORPO (Hong 2024)](https://arxiv.org/abs/2403.07691) ⭐
- [cDPO (Mitchell 2023)](https://ericmitchell.ai/cdpo.pdf)
- [Iterative DPO 详解 — Llama-3 Post-training §](https://arxiv.org/abs/2407.21783) ⭐⭐
- [Tülu-3 — DPO/PPO 对比](https://arxiv.org/abs/2411.15124) ⭐
- [Self-Rewarding LM (Yuan 2024)](https://arxiv.org/abs/2401.10020)（DPO + 自评 pair）
- [HuggingFace TRL DPO Trainer](https://huggingface.co/docs/trl/main/dpo_trainer) ⭐⭐（事实标准实现）
- [Sebastian Raschka — DPO/IPO/KTO/SimPO 对比](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction) ⭐⭐
- [A Comprehensive Survey of Direct Preference Optimization (Xiao 2024)](https://arxiv.org/abs/2410.15595)
