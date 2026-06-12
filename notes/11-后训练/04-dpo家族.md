# 11.4 DPO 家族（DPO / IPO / KTO / SimPO / ORPO）

[← 返回框架](../../README.md) · [📎 materials.md → §11.4](../../materials.md)

---

## 〇、本节回答什么

> 为什么 DPO 能在**不用 reward model、不用 RL 循环**的情况下完成"偏好对齐"，而且数学上看起来等价于 RLHF？KL-regularized 最优策略的闭式解到底说了什么？IPO / KTO / SimPO / ORPO 各自解决 DPO 的什么缺陷？为什么训练后 chosen 和 rejected 的 log-prob 经常**一起下降**？2024-2025 工业上怎么选？

DPO（Direct Preference Optimization, Rafailov 2023）是 2024 年成为**事实标准**的偏好对齐方法：

> **DPO 的发现**：在 KL-regularized RL 目标下，**最优策略与奖励之间是一对一的解析对应**——所以**根本不需要先学 reward 再 RL**，可以直接用偏好数据写出一个**监督学习** loss，把策略推到 RL 的最优解。

```
RLHF (§11.3):  pref data → RM → PPO loop (4 个模型同时驻留)
DPO (本节):    pref data ────────→ 1 个 BCE-like loss (policy + ref)
```

本节按这个顺序展开：

1. **完整推导**：从 KL-regularized RL 目标的闭式最优解 → 反解 reward → 套 Bradley-Terry → DPO loss。逐步骤说"为什么 $\log Z(x)$ 会消掉"。
2. **DPO 梯度的直觉**：解释为什么 chosen 和 rejected 的 log-prob 经常一起降、为什么 $\beta$ 调小反而能"更激进"。
3. **DPO 的几个已知缺陷**：长度偏差、reward "shrink"、对 SFT 起点敏感、offline 限制。
4. **IPO / KTO / SimPO / ORPO**：各自怎么解决一个特定缺陷，推导各自的 loss。
5. **Iterative / Online DPO**：把 DPO offline 限制突破的工业范式（Llama-3）。
6. **怎么选**：2025 决策树。

---

## 一、DPO 的完整推导

### 1.1 第一步：KL-regularized RL 目标的最优解

RLHF 的 RL 阶段目标（§11.3 §4.2）：

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \big[ r(x, y) \big] - \beta \cdot \mathbb{E}_x\big[ \text{KL}\big( \pi(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x) \big) \big]
$$

这个目标对任意给定 $x$ 都有**闭式解**。证明：在条件 $\sum_y \pi(y|x) = 1$ 下用 Lagrange 乘子：

$$
\mathcal{L}(\pi, \mu) = \mathbb{E}_{y \sim \pi}\big[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \big] + \mu \big( 1 - \sum_y \pi(y|x) \big)
$$

对 $\pi(y|x)$ 求导置零：

$$
r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \beta - \mu = 0
$$

解出 $\pi(y|x)$：

$$
\pi(y|x) = \pi_{\text{ref}}(y|x) \cdot \exp\left( \frac{r(x, y) - \beta - \mu}{\beta} \right)
$$

把 normalization 常数吸到一起：

$$
\boxed{
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left( \frac{1}{\beta} r(x, y) \right), \quad Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)
}
$$

**几何直觉**：最优策略 = ref 分布 × "reward 指数加权"。$\beta$ 越小，加权越激进（policy 倾向 reward 高的 $y$）；$\beta$ 越大，policy 越贴近 ref。当 $\beta \to \infty$ 时 $\pi^* \to \pi_{\text{ref}}$；当 $\beta \to 0$ 时 $\pi^*$ 集中在 $\arg\max_y r(x, y)$ 上。

### 1.2 第二步：反解 reward

从闭式解两边取 log：

$$
\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

整理得：

$$
\boxed{
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
}
$$

**关键观察**：给定 $\pi^*$ 和 $\pi_{\text{ref}}$，**reward 可以被表示成 log-ratio 加一个 $x$-依赖的偏置**。这意味着 reward 函数族在 $\beta \log Z(x)$ 自由度上**不可识别**——任何 $r$ 和 $r + f(x)$ 给出同一个最优策略。

### 1.3 第三步：套 Bradley-Terry，$\log Z(x)$ 神奇消失

Bradley-Terry 偏好概率（§11.3 §3.2）：

$$
P(y_w \succ y_l | x) = \sigma\big( r(x, y_w) - r(x, y_l) \big)
$$

把 §1.2 的 reward 表达式代入：

$$
\begin{aligned}
r(x, y_w) - r(x, y_l) &= \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \beta \log Z(x) \\
&= \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\end{aligned}
$$

**$\log Z(x)$ 神奇地消失了**——因为它只依赖 $x$，在 $y_w$ 和 $y_l$ 上相同。这是 DPO 的**核心数学魔术**：让"不可识别"的 partition function 在差分时刚好抵消，余下的只是 log-prob ratio。

### 1.4 第四步：把 $\pi^*$ 换成可训练的 $\pi_\theta$

DPO 的关键变化：**不再先学 reward，再优化策略——直接把 $\pi^*$ 视为要学的 $\pi_\theta$，把偏好概率对数似然当 loss**：

$$
\boxed{
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
}
$$

这是一个**纯监督损失**——给定 $(x, y_w, y_l)$，计算 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 在 $y_w$、$y_l$ 上的 log-prob，套一个 BCE。

→ 看起来像 RLHF，实际上**只需要 2 个模型（policy + 冻结的 ref）、不需要 RM、不需要 RL 循环、不需要 critic**。

### 1.5 DPO 与 RLHF 的关系：等价还是近似？

严格说**不等价**。三个细节：

1. **表达能力假设**：DPO 要求 $\pi_\theta$ 能精确表示 $\pi^*$（neural net 通常能逼近，但不严格相等）；
2. **偏好数据假设**：Bradley-Terry 假设需要成立，即人类偏好遵循 sigmoid 形式；
3. **数据分布**：DPO 是 offline——训练数据 $(y_w, y_l)$ 来自某个固定分布，而 PPO 是 on-policy——每步用当前策略采样新数据。

实证上 DPO 与 PPO 在**简单偏好任务**上接近（AlpacaEval、MT-Bench）；在**复杂任务**（reasoning、hard math）上 PPO 上限更高（因为 on-policy 能继续 explore）。

---

## 二、DPO 梯度的直觉

### 2.1 梯度形式

对 DPO loss 求梯度（设 $h_\theta = \beta(\log\pi_\theta(y_w|x)/\pi_{\text{ref}}(y_w|x) - \log\pi_\theta(y_l|x)/\pi_{\text{ref}}(y_l|x))$，即"reward 差"）：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = - \mathbb{E}\Big[ \underbrace{\sigma(- h_\theta)}_{\text{加权系数}} \cdot \beta \cdot \big( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \big) \Big]
$$

直觉：

- **加权系数 $\sigma(-h_\theta)$**：当模型已经把 $y_w$ 排在 $y_l$ 前（$h_\theta$ 大），$\sigma(-h_\theta) \to 0$，梯度变小（这条样本"学会了"）；反之梯度大（这条样本急需调整）。**这是 hard mining 的自动权重**。
- **方向**：增大 $\log \pi_\theta(y_w|x)$、减小 $\log \pi_\theta(y_l|x)$——正是我们直观上想要的"推上 chosen、压下 rejected"。

### 2.2 "log-prob 一起下降"现象

实测中常观察到：训练几个 step 后，**$\log \pi_\theta(y_w|x)$ 和 $\log \pi_\theta(y_l|x)$ 都下降了**——只不过 $y_l$ 下降得多一些（差值仍朝对的方向变）。

为什么？

- DPO loss 只关心 log-prob 的**差**，不关心绝对值；
- $\beta \log Z(x)$ 的"消失"意味着模型可以自由地把整个分布"挪到别处"，只要差值对就行；
- 直观地：模型可能把 $\pi_\theta$ 的概率质量挪到 $y_w, y_l$ **以外的某些 $y$ 上**——比如更长更套话的回答（如果训练数据里 $y_w$ 都偏长，模型可能学到"我应该输出更长的东西"，进而把更多 mass 给"未见过的长 response"上）。

这导致一个隐患：DPO 训完的模型可能**对 chosen 和 rejected 都不熟**——在 chosen 上比 ref 概率还低，在新输入上行为难预测。

**修补方案**：

- **DPOP**（[Pal 2024](https://arxiv.org/abs/2402.13228)）：在 chosen 比 ref 概率还低时加额外惩罚项；
- **Cal-DPO**：calibrated DPO，把绝对 log-prob 当正则；
- **β 调大**：β 越大，KL 锚定越强，绝对 log-prob 漂移越小。

### 2.3 $\beta$ 的角色

$\beta$ 不只是"KL 系数"，它还充当**温度**和**梯度缩放**：

- **小 $\beta$**：reward 差被放大（同样的 log-ratio 差 → 更大的 $h_\theta$），sigmoid 饱和快，梯度衰减快；
- **大 $\beta$**：reward 差被压缩，sigmoid 在线性区，梯度持续大。

工程上 $\beta = 0.1$ 是 DPO 默认起点。如果模型学得太慢，**降 $\beta$**（让 reward 差更显著）；如果模型偏离 ref 太远（输出崩），**升 $\beta$**。

注意：DPO 的 $\beta$ 和 PPO 的 $\beta$ 数值不同含义不同。PPO 的 $\beta$ 越大 KL 锚定越强；DPO 的 $\beta$ 越大也是 KL 锚定越强但同时也压缩 reward 差。

---

## 三、DPO 的工程实战

### 3.1 训练流程

```
1. 准备偏好数据 (x, y_w, y_l)            # 5w-20w pair 起步
2. 完成 SFT (π_ref = π_SFT)
3. 复制 π_ref 一份 → π_θ
4. 训练 loop:
   - forward π_θ 和 π_ref（ref 不需要 grad）：
     拿到 logp_θ(y_w), logp_θ(y_l), logp_ref(y_w), logp_ref(y_l)
   - 计算 h_θ 和 loss
   - backward 只更新 π_θ
5. 收尾：用 π_θ 替代原 SFT model
```

实现要点：

- **logp 是 sequence-level**：把整段 response 每个 token 的 log-prob 求和（mask 掉 prompt token）。
- **ref 可以 offload**：ref 不训练，可以 BF16 + CPU/disk offload；或预先把每条样本的 ref logp 算好缓存（**强烈推荐**：训练时 ref 不再前向，省一半显存 + 一半时间）。
- **避免数值问题**：log-ratio 直接相加减，不要先算 ratio 再 log（exp/log 会放大误差）。

### 3.2 显存预算 vs PPO

```
PPO (70B):    actor + critic + ref + RM ≈ 4 × 70B (含 master + grad + Adam state) ≈ 3000+ GB
DPO (70B):    actor + ref               ≈ 2 × 70B 但 ref 只前向，≈ 1500 GB
DPO + ref cache:  只 actor ≈ 1100 GB
```

→ DPO 显存约**省一半**（更好的实现可省 2/3），工程复杂度大幅下降。这是 DPO 在中小团队普及的关键。

### 3.3 超参

| 超参 | 推荐 | 备注 |
|---|---|---|
| **β** | 0.1 - 0.5 | 太小 → policy 漂离 ref；太大 → 学得慢 |
| **LR** | 5e-7 ~ 5e-6 | 比 SFT 小一个量级 |
| **Epochs** | 1 - 3 | 多了过拟合（reward 漂） |
| **Batch size** | 32-128 (effective) | DPO 每个样本 = 一对 (y_w, y_l)，显存压力大 |
| **Max length** | 4K-16K | 含 y_w + y_l 各自的长度 |
| **Schedule** | Cosine | warmup 5-10% |
| **Optimizer** | AdamW (β1=0.9, β2=0.95) | |
| **Grad clip** | 1.0 | |

### 3.4 数据要求

- **chosen vs rejected 必须有明显差异**——否则 DPO 学不到信号，loss 几乎不动；
- **chosen 不一定完美**，但要"明确比 rejected 好"；
- 同一 prompt 配对，**分布越一致越好**（不同 prompt 的对比无意义）；
- 工业经验：**5-20w 条偏好数据**有效；超过 50w 边际收益快速衰减。

### 3.5 常用数据集

- **UltraFeedback**：6w prompt × 4 response × GPT-4 评分，开源标杆；
- **UltraInteract**：reasoning 任务 + step-level 偏好；
- **HelpSteer2 → DPO pairs**：NVIDIA 开源；
- **Anthropic HH-RLHF**：早期标杆；
- **Magpie-DPO**：完全自合成；
- **Synthetic**：用 GPT-4 / Claude 当判官，几小时合成几万条。

---

## 四、DPO 的已知缺陷

### 4.1 缺陷一览

| 问题 | 表现 | 解 |
|---|---|---|
| **过拟合到 ref** | 输出贴近 SFT，前进幅度小 | β 调小 / 更多 epoch（小心） |
| **chosen log-prob 也下降** | π_θ(y_w) < π_ref(y_w)，但 π_θ(y_l) 降更多 | DPOP / Cal-DPO 加正则 |
| **长度偏差** | DPO 后模型输出变长 | SimPO 长度归一化 |
| **Distribution shift** | 训完 π_θ ≠ π_ref，offline data 失效 | iterative / online DPO |
| **没用上同 prompt 多 sample 信息** | 一次只看 1 对 | RLOO / GRPO 用组内 baseline |
| **对 SFT 起点敏感** | SFT 弱则 DPO 难推动 | 先把 SFT 做好 |
| **noisy label 退化** | 标注 ~10% 翻车直接污染 | cDPO 处理噪声 |

下面挑几个深入解释。

### 4.2 长度偏差的机制

DPO 数据里 chosen 平均略长（标注员偏好"看起来认真"的回答）→ DPO 训练让 π_θ 增大 y_w 的 log-prob 相对 ref → 因为 y_w 偏长，相当于在"长 response"方向上加权 → 模型生成时倾向输出更长的东西。

形式化：设 chosen 长度均值为 $\mu_w$，rejected 长度均值为 $\mu_l$，差值 $\Delta\mu = \mu_w - \mu_l > 0$。DPO 的梯度推动 policy 把"长度 $\sim \mu_w$"附近的 mass 放大——模型从训练数据里抽到的"chosen 信号"无法和"长度信号"解开，长度被一起放大。

**SimPO（§5.3）**通过长度归一化解决：用 average log-prob 替代 sum log-prob，把长度从信号中扣除。

### 4.3 chosen log-prob 一起降的机制

如 §2.2 所述，DPO 只约束差值，绝对 log-prob 可以漂走。但更深层的原因：

- DPO loss 没有显式约束 $\pi_\theta$ 不能减小 $y_w$ 的概率——只要 $y_l$ 减小得更多，loss 就降；
- 在 sigmoid 的"已正确"区间（$h_\theta$ 大），梯度的方向是"$y_l$ 减得快、$y_w$ 减得慢"，导致联动下降。

**DPOP** 的修补：

$$
\mathcal{L}_{\text{DPOP}} = \mathcal{L}_{\text{DPO}} + \lambda \cdot \max\big(0, \log \pi_{\text{ref}}(y_w|x) - \log \pi_\theta(y_w|x) \big)
$$

第二项是 hinge 损失：只在 $\pi_\theta(y_w) < \pi_{\text{ref}}(y_w)$ 时激活——惩罚 "chosen 反而比 ref 还差"的情况。

### 4.4 Distribution shift 与 offline 限制

DPO 是离线（offline）算法：训练用的 $(y_w, y_l)$ 来自某个固定数据集分布 $\mathcal{D}$。训练后 $\pi_\theta$ 偏离 $\pi_{\text{ref}}$ 也偏离 $\mathcal{D}$ 的生成分布——这时**再用同一批 $\mathcal{D}$ 训练**，loss 已经在错的分布上算，继续训没收益。

→ **DPO 的根本局限**：单次 offline 训练后 policy 卡在 $\mathcal{D}$ 附近，无法 explore。突破方案是 iterative DPO（§六）。

---

## 五、DPO 家族变体

### 5.1 IPO（Identity Preference Optimization, Azar 2023）

**[Azar 2023](https://arxiv.org/abs/2310.12036)** 的核心发现：当 Bradley-Terry 的偏好概率接近 0 或 1（即 chosen 比 rejected 好得很"确定"），DPO 的 $\log \sigma(\cdot)$ 趋于线性，导致**对极端偏好样本的过拟合**——模型把这些样本的 reward 差推到无穷大，把分布严重扭曲。

IPO 改用 MSE 形式：

$$
\boxed{
\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}\Big[ \big( h_\theta(x, y_w, y_l) - \tfrac{1}{2\beta} \big)^2 \Big]
}
$$

其中 $h_\theta$ 是 DPO 的"reward 差"项。

- MSE 在 $h_\theta$ 很大时仍有梯度（不饱和），但目标值是 $1/(2\beta)$ 有界——不会把差值推到无穷；
- 等价于"IPO 想要的 reward 差恰好是 $1/(2\beta)$，不多不少"；
- 实证：IPO 更稳但学得慢，AlpacaEval 上略逊于 DPO。

→ IPO 在理论上漂亮但**没成主流**——因为实际数据里"确定偏好"占比不高，DPO 的过拟合问题没那么严重。

### 5.2 KTO（Kahneman-Tversky Optimization, Ethayarajh 2024）

**[Ethayarajh 2024](https://arxiv.org/abs/2402.01306)** 的动机：**生产环境的反馈通常是 binary 的**——用户点赞/点踩，而不是给两条对比。KTO 让我们能用这种 binary 标签训练。

KTO 借鉴 Kahneman-Tversky 的前景理论（人类对收益和损失的非对称响应），设计：

$$
\mathcal{L}_{\text{KTO}}(x, y) = \begin{cases} 1 - \sigma\big( \beta \cdot r_\theta(x, y) - z_0 \big) & \text{if } y \text{ desirable} \\ 1 - \sigma\big( z_0 - \beta \cdot r_\theta(x, y) \big) & \text{if } y \text{ undesirable} \end{cases}
$$

其中：

- $r_\theta(x, y) = \log \pi_\theta(y|x) / \pi_{\text{ref}}(y|x)$，是单条样本的"reward proxy"；
- $z_0 = \mathbb{E}_x[\text{KL}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))]$，是 batch 内 KL 估计；起"参考点"作用（Kahneman-Tversky 的 reference point）。

直觉：

- desirable 样本：希望 $r_\theta > z_0$（这条样本的 reward 比 batch 平均 KL 高）；
- undesirable 样本：希望 $r_\theta < z_0$。

→ **数据标注成本大幅降低**——用户 like/dislike 信号即可训。生产环境（用户反馈、A/B 测试日志）首选。缺点：弱于 DPO 一点（信号密度低于成对比较）。

### 5.3 SimPO（Simple Preference Optimization, Meng 2024）

**[Meng 2024](https://arxiv.org/abs/2405.14734)** 的核心创新：**完全去掉 ref model**——既简化工程、又通过长度归一化消除长度偏差。

$$
\boxed{
\mathcal{L}_{\text{SimPO}}(\theta) = - \mathbb{E} \log \sigma\!\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma \right)
}
$$

两个关键改动：

1. **Average log-prob**：用 $\frac{1}{|y|} \log \pi_\theta(y|x)$ 替代 sum log-prob——天然长度归一化，扣除长度偏差。
2. **Target margin $\gamma$**：强制 chosen 和 rejected 拉开 $\gamma$ 的距离，避免它们靠太近。

并且**没有 $\pi_{\text{ref}}$ 项**——为什么可以去掉？因为：

- ref 在 DPO 里的作用是 "KL anchor 不让 policy 偏离过远"；
- 平均 log-prob 已经隐式控制了 policy 的输出分布（不能让所有 token 的 log-prob 都很高/很低）；
- $\gamma$ margin 也起到防止过激进的效果。

但代价：**SimPO 对 SFT 起点更敏感**——没有 ref anchor，如果 SFT 不稳，SimPO 容易把 policy 推向奇怪的方向。Llama-3-Instruct + SimPO 的组合在 2024 年屡屡刷新榜（Arena-Hard、AlpacaEval-2），但前提是 Llama-3-Instruct 本身已经很强。

### 5.4 ORPO（Odds Ratio Preference Optimization, Hong 2024）

**[Hong 2024](https://arxiv.org/abs/2403.07691)** 的动机：**把 SFT 和偏好对齐合并成一个 stage**——省一个训练阶段。

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}(y_w, y_l)
$$

其中 SFT loss 是经典的 CE，OR (odds ratio) loss 是：

$$
\mathcal{L}_{\text{OR}} = - \log \sigma\!\left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right), \quad \text{odds}_\theta(y|x) = \frac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}
$$

直觉：

- SFT 部分让模型学 $y_w$ 的内容；
- OR 部分让 $y_w$ 的 odds 显著高于 $y_l$ 的 odds（"odds" 而不是 prob，对 likely 样本的差异更敏感）；
- 两个 loss 加权——一个 pass 同时学示范 + 偏好。

优势：

- **省一阶段**：base + 偏好数据 → ORPO 直接出 chat 模型；
- 数据用量少 → 适合小数据场景；
- 实证：在小数据上有时优于 SFT + DPO 两阶段（因为不需要先 SFT 再 DPO 双重过拟合）。

缺点：**上限略低于 SFT + DPO**——两阶段分工可以更精细地优化各自目标。

### 5.5 其他衍生：cDPO / Cal-DPO / DPOP / RPO

- **cDPO** (Mitchell 2023)：考虑标签噪声——假设 ~10% 标注被翻转，loss 形式变成：

  $$
  \mathcal{L}_{\text{cDPO}} = (1 - \epsilon) \cdot \mathcal{L}_{\text{DPO}}(y_w, y_l) + \epsilon \cdot \mathcal{L}_{\text{DPO}}(y_l, y_w)
  $$
  
  $\epsilon$ 是估计的噪声率；本质是 label smoothing。

- **Cal-DPO**：calibrated DPO，添加正则项防 log-prob 联动下降；
- **DPOP**：见 §4.3，hinge 项保证 chosen 不退化；
- **RPO** (Reward-aware Preference Optimization)：当**同时有** RM 分数和 pair 时，把 RM 信号融进 loss。

---

## 六、Online / Iterative DPO（突破 offline 限制）

### 6.1 Iterative DPO（Llama-3 范式）

```
for t in 1..T:
    π_t (上一轮的 policy 或初始 SFT)
    ↓
    用 π_t 采样 K 条 response per prompt
    ↓
    用 RM (或 LLM judge) 排序，取 best/worst → pair (y_w, y_l)
    ↓
    DPO 训练一轮 → π_{t+1}
```

每轮重新采样偏好数据 → policy 不会 stale（与最新策略分布一致）。

Llama-3 用了 **6 轮** SFT-DPO 迭代，AlpacaEval-2 win rate 累计提升 ~10 点。每轮花费约 1 周（含数据合成 + DPO 训练）。

### 6.2 Online DPO

更激进：一个 batch 内同步采样 → DPO update：

```
for batch in stream:
    prompts → π_θ 采 K 条 → judge 选 best/worst → DPO loss → update
```

接近 PPO 的 online 设定，但仍是 DPO loss。复杂度上升（需要在训练 loop 里跑 generation），效果接近 PPO。

代表实现：OpenRLHF 的 online DPO mode、TRL 的 OnlineDPO。

### 6.3 Iterative DPO 与 PPO 的比较

| 维度 | Iterative DPO | PPO |
|---|---|---|
| 模型数 | 2-3（policy + ref + 可选 RM） | 4 |
| 显存 | 中 | 大 |
| 工程复杂度 | 低 | 高 |
| Reward hacking 风险 | 较低（offline 数据已固定） | 高 |
| 上限 | 接近 PPO | 略高 |
| 训练时间 | 6 轮 ~ 几周 | 1 次 ~ 几周 |

→ 工业上 Iterative DPO **逐渐取代了 PPO** 作为主流偏好对齐方法。OpenAI 内部据传仍 PPO（资源充足，求极致），开源界（Llama / Qwen / Tülu）DPO 主导。

---

## 七、DPO 家族对比表

| 方法 | 需 ref | 需 RM | 数据类型 | 长度校正 | 备注 |
|---|---|---|---|---|---|
| **DPO** | ✅ | ❌ | pairs | ❌ | 经典基线 |
| IPO | ✅ | ❌ | pairs | ❌ | 防过拟合（理论好，实战收益小） |
| **KTO** | ✅ | ❌ | **binary** | ❌ | 标注成本最低 |
| **SimPO** | **❌** | ❌ | pairs | **✅** | 2024 SOTA leaderboard |
| **ORPO** | ❌ | ❌ | pairs (+SFT) | ❌ | 与 SFT 合并 |
| cDPO | ✅ | ❌ | pairs (noisy) | ❌ | label smoothing 处理噪声 |
| DPOP | ✅ | ❌ | pairs | ❌ | 防 chosen log-prob 退化 |
| **Iterative DPO** | ✅ | ✅ | pairs | ❌ | Llama-3 范式 |
| PPO (§11.3) | ✅ | ✅ | RM scalar | ❌ | 经典 RL，上限高 |

---

## 八、什么时候选哪个？（2025 决策树）

```
有强 RM + 算力够（大厂）       →  PPO（上限最高）或 Iterative DPO
有偏好对（chosen/rejected）    →  DPO 或 SimPO
  SFT 起点够稳                  →  SimPO（榜单一般更高）
  SFT 起点不稳                  →  DPO（ref anchor 更安全）
只有 like/dislike 信号         →  KTO
SFT + 偏好一起做（数据少）     →  ORPO
追求 leaderboard               →  SimPO + DPO 双跑取最好
复杂场景（reasoning）          →  SFT → DPO → RLVR/GRPO（§11.5）
```

**2025 工业标准栈**：

1. **快速产品**：SFT → 一次性 offline DPO → 上线；
2. **质量优先**：SFT → Iterative DPO（3-6 轮）→ 上线；
3. **极致质量**：SFT → DPO → GRPO/PPO(RLVR) → 上线（Llama-3 / Tülu-3 / DeepSeek 路径）。

---

## 九、关键问答

**Q1**：DPO 数学上等价于 RLHF 吗？
- **不严格等价**——DPO 假设 $\pi_\theta$ 能精确表示 KL-regularized 最优解，对 LLM 表达能力的隐含假设；
- 实证：DPO 在简单偏好任务上接近 PPO，复杂任务上 PPO 上限更高；
- DPO 的优势在工程复杂度和稳定性，不在理论最优性。

**Q2**：为什么 DPO 训完后 chosen 和 rejected 的 log-prob 都降了？
- DPO loss 只关心差值，绝对 log-prob 可以漂移；
- KL anchor 不足以约束绝对 mass，模型可能把 mass 放到 $(y_w, y_l)$ 以外；
- 用 DPOP / Cal-DPO 加正则，或加大 $\beta$。

**Q3**：$\beta$ 怎么调？
- 起步 0.1。看 sequence-level KL(π || π_ref)：
  - KL > 50（policy 漂远）→ β 加大；
  - KL < 5（policy 几乎不动）→ β 减小；
- SimPO 的 $\beta$ 通常更大（2.0-2.5），因为没有 ref 项。

**Q4**：DPO 数据量需要多少？
- 1w 起步看到效果；5-20w 是常见区间；
- 数据**质量 > 数量**——dirty pair 严重伤害（cDPO 也只能救 ~10% 噪声）。

**Q5**：DPO 训完后还需要 PPO 吗？
- 简单 chat 对齐：DPO 够了；
- Reasoning、工具使用、多步任务：仍需 PPO / GRPO；
- 工业上 DPO → GRPO 是常见接力（DPO 做偏好、GRPO 做 reasoning）。

**Q6**：SimPO 真的不需要 ref model 吗？
- 是的，loss 中没有 $\pi_{\text{ref}}$ 项；
- 显存省 50%，训练快 ~30%；
- 但**对 SFT 初始模型质量更敏感**——没有 anchor，SFT 弱则 SimPO 容易崩；
- 实战：先看 SFT 后的 MT-Bench 是否 > 7.5，是再用 SimPO，否则 DPO 更稳。

**Q7**：Iterative DPO 比 single-shot DPO 强多少？
- AlpacaEval-2 LC win rate：**+5-10 个点**（Llama-3 报告）；
- 工程成本：3-6 倍；
- 大厂标配，中小团队按需。

**Q8**：DPO 之外，什么时候用 KTO？
- 数据形态是 like/dislike binary 而非 pair（如生产环境用户反馈）；
- 数据稀缺（KTO 信号密度低但能用大量样本）；
- 工业上 chat 平台后期持续 KTO 微调是常见做法。

**Q9**：ORPO 真的能省一个阶段吗？
- 是的，但**前提是数据足够好**——ORPO 把 SFT 和偏好挤在一起训，对数据噪声更敏感；
- 小数据（<5w pairs）场景 ORPO 优势明显；
- 大数据场景 SFT → DPO 两阶段仍优。

**Q10**：DPO 训练时 loss 不下降怎么办？
- 检查数据：chosen 和 rejected 是不是真的有差异？随机交换一下 loss 是否变化？
- 检查 mask：是不是只对 response 部分算 logp？prompt 部分应该 mask；
- 检查 $\beta$：太大会让 loss 几乎不动；
- 检查 LR：太小（< 1e-7）几乎学不动。

---

## 十、本节与其他节关系

```
§11.1 SFT  ──→ §11.3 PPO         (上限高，工程贵)
                ↓
            §11.4 DPO 家族 (本节)
                ↓
            §11.5 RLVR / GRPO    (verifiable reward + 群体 baseline)
```

DPO 家族借鉴 RL 闭式解思想做"伪 RL"；GRPO 又借鉴 DPO 的简化 + REINFORCE++ 的 baseline 思想。理解路径：**PPO（母算法）→ DPO（offline 简化）→ GRPO（group baseline + verifiable reward）**。

---

## 十一、参考资料

**开山**：
- [DPO (Rafailov 2023)](https://arxiv.org/abs/2305.18290) ⭐⭐（理论清晰、推导优雅）

**变体**：
- [IPO (Azar 2023)](https://arxiv.org/abs/2310.12036)
- [KTO (Ethayarajh 2024)](https://arxiv.org/abs/2402.01306) ⭐
- [SimPO (Meng 2024)](https://arxiv.org/abs/2405.14734) ⭐⭐
- [ORPO (Hong 2024)](https://arxiv.org/abs/2403.07691) ⭐
- [cDPO (Mitchell 2023)](https://ericmitchell.ai/cdpo.pdf)
- [DPOP / Smaug (Pal 2024)](https://arxiv.org/abs/2402.13228)
- [RPO — Reward-aware PO (Adler 2024)](https://arxiv.org/abs/2406.11704)

**Iterative / Online**：
- [Iterative DPO — Llama-3 Post-training §](https://arxiv.org/abs/2407.21783) ⭐⭐
- [Self-Rewarding Language Models (Yuan 2024)](https://arxiv.org/abs/2401.10020)（DPO + 自评 pair）
- [Online DPO (Guo 2024)](https://arxiv.org/abs/2402.04792)
- [Online RLHF (Dong 2024)](https://arxiv.org/abs/2405.07863)

**对比 / 分析**：
- [Tülu-3 — DPO vs PPO 对比](https://arxiv.org/abs/2411.15124) ⭐⭐
- [A Comprehensive Survey of DPO (Xiao 2024)](https://arxiv.org/abs/2410.15595)
- [Insights into Alignment (Saeidi 2024)](https://arxiv.org/abs/2404.14723)

**工程实现**：
- [HuggingFace TRL DPO Trainer](https://huggingface.co/docs/trl/main/dpo_trainer) ⭐⭐
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Alignment Handbook (HuggingFace)](https://github.com/huggingface/alignment-handbook) ⭐

**深度博客**：
- [Sebastian Raschka — DPO/IPO/KTO/SimPO 对比](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction) ⭐⭐
- [Nathan Lambert — Interconnects: DPO 系列](https://www.interconnects.ai/) ⭐
- [Lilian Weng — Reward Hacking (含 DPO 段)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) ⭐
