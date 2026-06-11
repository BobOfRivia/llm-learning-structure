# 11.4 Reasoning 训练（RLVR / GRPO / DAPO / GSPO / 推理蒸馏）

[← 返回框架](../../README.md) · [📎 materials.md → §11.4](../../materials.md)

---

## 〇、本节回答什么

> o1 / R1 这条 "reasoning model" 路线到底训了什么？为什么 verifiable reward 比 learned reward 更稳？GRPO 是怎么把 PPO 的 critic 干掉的、为什么 group baseline 是 unbiased 的？DAPO / GSPO 在 GRPO 上各改了什么、解决什么具体痛点？PRM 与 ORM 的取舍是什么？为什么大模型 RL 出来的 reasoning 能用 SFT 蒸到小模型？

2024 年 9 月 OpenAI o1 发布、2025 年 1 月 DeepSeek-R1 复现并开源——**Reasoning RL** 取代经典 RLHF 成为后训练新主线。核心范式：

```
SFT base
  ↓
RLVR (Reinforcement Learning with Verifiable Rewards)
  ↓
长 CoT、能"想"几千 token 再答的 reasoning 模型
```

它和经典 RLHF（§11.2-3）的根本区别在 **reward 来源**：

- **RLHF**：reward = 学到的 RM（neural net 估计的人类偏好）→ 易被 hack；
- **RLVR**：reward = 程序化校验（数学答案对错、代码 unit test）→ 0/1 信号，无法 hack。

这个变化看似简单，影响是结构性的：
- **不再需要 RM**——省一个模型 + 一个数据集；
- **reward 信号 sparse 但严格**——必须设计能处理稀疏 reward 的 RL 算法（→ GRPO 的诞生）；
- **reasoning 是 RL 涌现出来的**（R1-Zero 现象）——长 CoT、反思、自我修正不是 SFT 教的，而是 RL 自己找到的。

本节按这个顺序展开：

1. **RLVR 的理论根基**：为什么 verifiable reward 解决了 RLHF 的 hacking 问题。
2. **R1-Zero 现象**：reasoning 涌现的实证证据与机制猜想。
3. **GRPO**：从 PPO 推到 GRPO 的完整变换、group baseline 为什么是 unbiased、为什么省 critic 仍 work。
4. **DAPO / GSPO**：GRPO 在 long CoT 训练中的不稳定根因、四个改进的针对性。
5. **PRM vs ORM**：过程奖励 vs 终值奖励的取舍、R1 为什么放弃 PRM。
6. **推理蒸馏**：为什么小模型上"大模型 RL → 小模型 SFT"反而比"小模型直接 RL"好。

---

## 一、RLVR：为什么 verifiable reward 是革命

### 1.1 RLHF 的根本困难

回顾 §11.2：RLHF 的 reward 是 **learned RM**。这带来三个根本问题：

1. **OOD 不可靠**：RM 在训练分布内打分准确，policy 一旦走到 OOD（如生成新格式、新策略），RM 就乱给分；
2. **Reward hacking 不可避免**（Goodhart's Law）：RM 是 quality 的代理，policy 会找到代理的漏洞；
3. **学习 + RL 两次误差累积**：RM 训出来准确率 70-72% 就到顶（人类一致性的上限），policy 再在这个 noisy reward 上爬坡。

→ RLHF 的"reward 噪声"决定了 RLHF 的天花板。

### 1.2 Reasoning 任务的特殊性：有 ground-truth

数学、代码、形式逻辑等任务有**程序化的 ground-truth**：

```
数学题:    答案是不是 42 ？        verifier(answer) → 1 or 0
代码题:    unit test 通过否 ？     pytest → pass/fail
逻辑题:    形式化验证               z3 / lean → ok/fail
事实题:    检索 + 校验               retrieval verifier
```

Reward 是**布尔 / 稀疏标量**，由**确定性程序**给出——不需要学习、不会被 hack（除非 reward 设计有漏洞，比如做题作弊）。

这一类 reward 设定叫 **RLVR (RL with Verifiable Rewards)**——Tülu-3 (AllenAI 2024) 正式命名。

### 1.3 RLVR 解决了什么

| RLHF 难题 | RLVR 的解 |
|---|---|
| RM OOD 不可靠 | verifier 是确定性程序，OOD 同样判 0/1 |
| Reward hacking | policy 想 hack 必须 hack 程序，门槛极高 |
| RM 训练误差 | 无 RM，无此误差 |
| 数据成本 | 只需 (question, ground-truth)，比偏好对便宜 100× |

但 RLVR 有自己的**局限**：

- **只适用于有 ground-truth 的任务**——数学、代码、逻辑；
- **通用 chat（"哪个回答更友好"）仍需 RM 或 LLM-judge**；
- Reward 是 0/1 → sparse → 训练初期信号弱（要靠 group baseline 才能挖出信号）。

→ RLVR 不取代 RLHF，而是**补充**：reasoning 用 RLVR，chat 偏好仍用 DPO/PPO。工业最佳实践是**两者混训**（Tülu-3、R1 都是混合 reward 类型）。

### 1.4 Verifier 设计的核心问题

Verifier 的**误判率**直接是 RL 的 noise floor——verifier 错一次，policy 就被引导到错的方向。所以 verifier 设计必须严谨：

| 任务 | Verifier 形式 | 误判风险 |
|---|---|---|
| 数学 | sympy 等价校验（不只字符串匹配） | 等价表达式漏判（"1/2" vs "0.5"） |
| 代码 | sandbox 执行 + unit test 全过 | 测试不全则误判（漏掉 edge case） |
| 多选题 | 字符串匹配 + 答案抽取 | 输出格式不一致（"A" vs "The answer is A"） |
| 通用 QA | LLM-as-judge | judge 的偏见 |
| 形式逻辑 | Z3 / Lean | 形式化转录的偏差 |

**经验**：

- 数学题用 sympy 化简后比较，能处理 80% 的等价表达式；
- 代码题用 hidden test cases（不让 model 看到测试，避免作弊）；
- 通用 QA 必须有 fallback（LLM judge 不确定时给中性 reward 0）。

---

## 二、R1-Zero 现象：reasoning 是 RL 涌现出来的

### 2.1 现象

[DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) 报告了一个**反直觉**的实验：

```
DeepSeek-V3-Base （未做任何 instruction-tuning SFT）
   ↓  纯 RL (GRPO + verifiable reward on math/code)
DeepSeek-R1-Zero
```

结果：

- **不需要 SFT 教 "think 格式"**——R1-Zero 自发学会用 `<think>...</think>` 标签包裹推理过程；
- **CoT 长度从 < 1k 自发增长到 ~ 10k token**——模型发现"多想几步，正确率更高"；
- **出现反思（"wait, let me reconsider"）、自我修正、尝试多路径**——这些都是 RL 涌现的，不是数据里教的；
- DeepSeek 称之为 **"Aha moment"**——某个 RL step 后突然看到模型行为质变。

### 2.2 涌现的机制猜想

R1-Zero 的现象需要解释三个问题：

**问题 A**：为什么 base model 没 SFT 就能听话？

答：DeepSeek-V3 的 pretrain 数据里**已经有大量代码和数学语料**——包含解题步骤、"let me think"、"first, ...", "second, ..."这类模式。Base model 本身已知道"在数学题前面写推理过程"是高概率序列。

**问题 B**：为什么 CoT 自发变长？

答：RLVR reward 是 binary 的 0/1，长 CoT vs 短 CoT 的差别在**正确率**：

- 短 CoT 在简单题上能对，但在难题上错；
- 长 CoT（多步推导、回溯、验证）在难题上能对；
- 在题目难度分布混合时，长 CoT 的期望 reward 更高 → policy 自然偏向长 CoT。

形式化：设 $\pi_s$ 是短 CoT 策略，$\pi_l$ 是长 CoT 策略，在难度 $d$ 的题上正确率分别 $p_s(d), p_l(d)$，$p_l(d) > p_s(d)$ 在难题上差距更大。期望 reward $\mathbb{E}_d[p(d)]$ 长 CoT 更高 → 梯度推 policy 去 $\pi_l$。

**问题 C**：为什么会自发出现"反思"？

答：在某些题上**只有"先尝试一条路径、发现错了、回头改"**才能解出。RL 在大量 sample 中找到这条路径就能持续强化。base model 在 pretrain 里见过这种 "wait, this isn't right"的语料模式 → RL 把它捡起来用。

→ R1-Zero 没有"教模型推理"，而是**让模型在大量尝试里挑出 reasoning 这条 high-reward trajectory，并强化它**。这是经典的 RL 套路，只是因为 reward sparse、CoT 长，对算法的耐心和稳定性要求极高。

### 2.3 完整 R1 pipeline

R1-Zero 虽然能推理，但**可读性差**（推理过程混合语言、跳步、不友好）。所以生产版本要再做几步：

```
DeepSeek-V3-Base
   ↓  纯 RL (GRPO + verifiable)
R1-Zero (可推理但可读性差)
   ↓  cold-start SFT (用人挑的、可读的少量长 CoT 数据微调)
R1-SFT (可读、可推理)
   ↓  RL (reasoning + helpfulness 混合 reward)
R1 (生产形态：reasoning 强 + chat 友好)
   ↓  rejection sampling + SFT distill
R1-Distill-Qwen-{1.5B / 7B / 14B / 32B / 70B}
```

这条 pipeline 体现了 R1 的核心方法论：

- **RL 先于 SFT**：先让模型自由 explore 找到 reasoning trajectory；
- **SFT 改样式不改能力**：cold-start SFT 只调输出格式（可读性），不灌新的推理能力；
- **混合 reward**：除 verifiable reward 外，加 helpfulness reward（避免模型只会数学不会聊天）；
- **蒸馏到小模型**：大模型 RL 出的 reasoning trajectory 直接 SFT 给小模型，小模型继承推理能力。

---

## 三、GRPO（Group Relative Policy Optimization）

DeepSeek 在 [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) 提出，R1 训练主力算法。本节完整推导。

### 3.1 动机：怎么去掉 critic

PPO 的目标：

$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t\big[ \min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t) \big]
$$

$A_t$ 用 GAE 算，依赖 $V(s_t)$——需要 **critic**（一个和 policy 同尺寸的 value 模型）。70B PPO 训 critic 显存压力巨大（详见 §11.2 §4.5）。

**核心问题**：能不能不要 critic？答案是可以——只要找到一个**无偏的 baseline 替代 $V(s_t)$**。

### 3.2 Group baseline 的设计

对每个 prompt $x$，**采样 $G$ 条 response** $\{y_1, \ldots, y_G\}$（典型 G=8-64）：

1. 计算每条的 reward $\{r_1, \ldots, r_G\}$（verifier 给）；
2. 用**组内均值**作 baseline：

$$
\bar r(x) = \frac{1}{G} \sum_{i=1}^G r_i, \quad A_i = r_i - \bar r(x)
$$

3. 归一化（可选但实战必加）：

$$
A_i = \frac{r_i - \bar r(x)}{\text{std}(r_1, \ldots, r_G)}
$$

4. 把整条 $y_i$ 的所有 token 都赋予同一个 $A_i$。

### 3.3 为什么 group baseline 是 unbiased

REINFORCE 的标准梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\big[ \nabla_\theta \log \pi_\theta(y|x) \cdot r(x, y) \big]
$$

可以减去任何**只依赖 $x$ 的 baseline** $b(x)$，期望不变：

$$
\nabla_\theta J(\theta) = \mathbb{E}\big[ \nabla_\theta \log \pi_\theta(y|x) \cdot (r(x, y) - b(x)) \big]
$$

证明：$\mathbb{E}_y[\nabla_\theta \log \pi_\theta(y|x) \cdot b(x)] = b(x) \cdot \nabla_\theta \mathbb{E}_y[1] = 0$。

group baseline $\bar r(x) = \frac{1}{G} \sum r_i$ 是 batch 内的 $r(x, y)$ 估计——它**依赖采样**，理论上不是严格的 $b(x)$（因为它包含当前样本本身）。

**leave-one-out baseline** 才是严格无偏的：$b_i(x) = \frac{1}{G-1} \sum_{j \ne i} r_j$。但 GRPO 直接用包含自身的 mean——实证上偏差小可忽略（当 G 不太小时）。

→ **GRPO 的 group baseline 在 G 足够大（> 4）时方差降低显著、偏差小到不重要**。这是用 baseline 替代 critic 的核心论据。

### 3.4 GRPO loss

$$
\boxed{
\mathcal{L}_{\text{GRPO}}(\theta) = - \mathbb{E}\!\left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\big( \rho_t^i A_i, \text{clip}(\rho_t^i, 1\!-\!\epsilon, 1\!+\!\epsilon) A_i \big) \right] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
}
$$

其中 $\rho_t^i = \pi_\theta(y_t^i | x, y_{<t}^i) / \pi_{\theta_{\text{old}}}(y_t^i | x, y_{<t}^i)$。

关键设计：

- **token-level $\rho$、sequence-level $A$**：每个 token 算自己的 importance ratio，但 advantage 整条共享（来自 group baseline 的组内归一化）。
- **per-response 长度归一化** $\frac{1}{|y_i|}$：避免长 response 在 loss 中权重过大（每条 response 平等贡献）。
- **KL 显式作为 loss 一部分**（而不是 reward shaping）：PPO 把 KL 加在 token reward 上，GRPO 直接加在 loss 上，数学等价但工程简洁。

### 3.5 GRPO 比 PPO 的优势

| 维度 | PPO | GRPO |
|---|---|---|
| 模型数 | 4（actor + critic + ref + RM） | 2-3（actor + ref [+ RM]） |
| 显存（70B） | ~3000 GB | ~1500 GB |
| Critic 训练问题 | 长 trajectory 上 critic 难收敛 | 无 critic，无此问题 |
| Reward 类型 | sparse 也可，但 critic 难学 | sparse reward 友好（0/1 组内差异即信号） |
| Reasoning 任务 | 受 critic 限制 | 实测 ≥ PPO |
| 通用 chat | 略优（critic 有用） | 略逊（baseline 噪声大） |

### 3.6 GRPO 实战超参（R1 配置）

| 超参 | 值 | 说明 |
|---|---|---|
| Group size G | 16-64 | 大 G 方差小但慢 |
| Rollout batch | 256-1024 prompts | |
| Multi-epoch | 1-2 | 比 PPO 少，因为 group baseline 噪声小 |
| Clip $\epsilon$ | 0.2 | 同 PPO |
| KL $\beta$ | 0.001 - 0.01 | R1 用很小 KL（让 policy 自由 explore） |
| LR | 1e-6 ~ 5e-6 | |
| Max gen length | 8K-32K | reasoning 输出可能很长 |
| Generation temperature | 1.0 | 高熵保探索 |
| Reward type | 0/1 verifiable | |

注意 R1 的 KL $\beta$ **很小**（0.001 量级）——比 PPO 小一两个数量级。原因：reasoning task 需要充分 explore，KL 锁太紧反而压制 reasoning 涌现。

### 3.7 GRPO 的实战坑

**坑 1**：Reward 不 normalize → 不同 prompt 难度不可比（简单题 reward 全 1，难题全 0，没差异）。
→ 组内归一化（除以 std）是关键。

**坑 2**：组内全对或全错 → advantage = 0，rollout 浪费。
→ **难度分桶采样**：从不同难度题目里均匀采，保证组内有 reward 差异。或用 **DAPO 的 dynamic sampling**（§四）。

**坑 3**：Generation 长度爆炸（> 32K）→ KV cache 爆。
→ 必须 cap max gen length，且对超长 response 做 reward 衰减（DAPO §四）。

**坑 4**：Reward signal 太稀疏（5% 题目能对）→ 几乎所有 group 都全错。
→ 用难度合适的 dataset（不要纯 IMO 难题），或先 cold-start SFT 让 base 上来一个能力起点。

**坑 5**：KL 算错 → 训练发散。
→ 用 unbiased KL estimator（k3 形式：$\text{kl} \approx (r - 1) - \log r$，$r = \pi_{\text{ref}}/\pi_\theta$）。

---

## 四、DAPO（Decoupled Clip and Dynamic Sampling, ByteDance 2025）

[ByteDance 2025](https://arxiv.org/abs/2503.14476)：GRPO 的工程改进，针对长 CoT 训练不稳定的四个针对性补丁。

### 4.1 改进 1：Clip-Higher（非对称 clip）

**问题**：长 CoT 中存在大量"低概率但关键"的 token（如反思词 "wait"、回溯词 "actually"）。PPO 对称 clip $\text{clip}(\rho, 1-\epsilon, 1+\epsilon)$ 对这些 token 的探索过于压制——一旦 $\rho > 1+\epsilon$ 就被 clip，policy 难以学到这些新模式。

**解**：**上下 clip 解耦**：

$$
\text{clip}(\rho, 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}), \quad \epsilon_{\text{high}} > \epsilon_{\text{low}}
$$

典型 $\epsilon_{\text{high}} = 0.28, \epsilon_{\text{low}} = 0.2$——允许 policy 更激进地"放大新模式"（往上 clip 更松），但"压制旧模式"仍严格（往下 clip 同 PPO）。

为什么非对称是对的？因为我们希望：

- 对**已经高概率的好 token**：策略适度增强即可，不要过激（保 stable）；
- 对**目前低概率的好 token**：策略要敢放大（鼓励 explore）；

而 importance ratio $\rho > 1$ 主要对应"目前低概率的 token 被强化"——这正是需要鼓励的方向。

### 4.2 改进 2：Dynamic Sampling

**问题**：GRPO 组内全对或全错时，$A_i = 0$，这 group 的 rollout 完全浪费。在难题 batch 上可能 30% 的 rollout 都是浪费。

**解**：**动态过滤**——只保留 group 内有 reward 多样性的 prompt：

```python
for batch:
    rollouts = sample(prompts, G)
    valid = [p for p in batch if 0 < accuracy(rollouts[p]) < 1]
    if len(valid) < batch_size:
        resample_more_prompts()
    train_on(valid)
```

即"continue sampling until enough valid prompts"。代价：rollout 时间增加 1.5-2×，但每条数据都有效，整体效率反而高。

### 4.3 改进 3：Token-Level Loss

**问题**：GRPO 对每条 response 取 mean log-prob（除以 $|y_i|$）→ 长 response 中每个 token 的权重被稀释。

但在长 CoT 中，**所有 token 都重要**（每一步推理都参与最终答案）——稀释让"长 CoT 中段 token 的梯度信号"过弱。

**解**：**直接对所有 token 求和**（"flat token-level loss"），不做长度归一化：

$$
\mathcal{L}_{\text{DAPO}} = -\frac{1}{\sum_i |y_i|} \sum_{i=1}^G \sum_t \min(\rho_t^i A_i, \ldots)
$$

分母是所有 token 数之和（在整个 batch 上归一化），而不是每条 response 独立归一化。

效果：长 response 中每个 token 拿到与短 response 中 token **同等的梯度权重**——长 CoT 收敛更稳。

### 4.4 改进 4：Overlong Reward Shaping

**问题**：response 截断到 max_len 时，简单给 reward=0 不太合理——被截断 ≠ 错（可能只是没说完）。

**解**：**软惩罚** + **逐步衰减**：

$$
r_{\text{shaped}} = \begin{cases} r_{\text{verifier}} & \text{if } |y| \le L_{\text{soft}} \\ r_{\text{verifier}} \cdot \frac{L_{\text{max}} - |y|}{L_{\text{max}} - L_{\text{soft}}} & \text{if } L_{\text{soft}} < |y| \le L_{\text{max}} \\ -1 & \text{if } |y| > L_{\text{max}} \end{cases}
$$

- $L_{\text{soft}}$（如 16K）：以下给原始 reward；
- $[L_{\text{soft}}, L_{\text{max}}]$（如 16K-32K）：逐步线性衰减（鼓励适度长度）；
- $> L_{\text{max}}$：直接负 reward（截断惩罚）。

效果：policy 学到"在合理长度内做完推理"，避免长度爆炸。

### 4.5 DAPO 工业实战

ByteDance 用 DAPO 训出 Qwen-32B-DAPO，在 AIME 2024 上超过 R1-Zero。完整开源（代码 + 数据 + checkpoint）→ DAPO 成为 2025 后 reasoning RL 的开源主力。

---

## 五、GSPO（Generative Sequence Policy Optimization, Qwen 2025）

[Qwen 2025](https://arxiv.org/abs/2507.18071)：GRPO 的另一支改进，重写 importance ratio。

### 5.1 问题：token-level ratio 与 sequence-level reward 不匹配

GRPO 在每个 token 上算 $\rho_t = \pi_\theta(y_t|\cdot) / \pi_{\theta_{\text{old}}}(y_t|\cdot)$，但 reward 是 **sequence-level**（整条 response 一个 reward）。

这种粒度错配在长 CoT 上会爆发：

- 一条 1000 token 的 response：$\rho_t$ 在某些 token 上偶尔 > 1+$\epsilon$、< 1-$\epsilon$；
- 因为 clip 是 per-token 的，部分 token 被 clip、部分没被 clip；
- 累乘起来（隐式）的总 importance ratio **方差极大**——梯度方向飘忽。

形式化：sequence-level 真实 ratio 是 $\prod_t \rho_t$，方差随 $|y|$ 指数增长。token-level 处理"假装"每 token 独立，但实际上它们高度相关，方差累积反映出来。

### 5.2 GSPO 的改动

用**sequence-level importance ratio**（等价于 average log-prob 差的 exp）：

$$
\rho_{\text{seq}} = \exp\!\left( \frac{1}{|y|} \sum_t \log \frac{\pi_\theta(y_t | \cdot)}{\pi_{\theta_{\text{old}}}(y_t | \cdot)} \right)
$$

然后整条 sequence 共享一个 clip 与 advantage：

$$
\mathcal{L}_{\text{GSPO}} = - \mathbb{E}\big[ \min(\rho_{\text{seq}} A, \text{clip}(\rho_{\text{seq}}, 1-\epsilon, 1+\epsilon) A) \big]
$$

### 5.3 GSPO 的偏差-方差分析

- **方差**：sequence-level ratio 是 mean log-ratio 的 exp，方差不随 $|y|$ 指数爆炸（mean 是个稳健统计量）；
- **偏差**：用 sequence-level ratio 当 importance weight 不严格 unbiased（严格的 importance sampling 需要 $\prod_t \rho_t$），但实证偏差小；
- 对**长 CoT** 极友好（变长 KV 缓存 + token-level ratio 易 NaN，GSPO 避免）。

### 5.4 GSPO 与 GRPO 怎么选

| 场景 | 推荐 |
|---|---|
| 短 response（< 2K token） | GRPO（差不多） |
| 长 CoT（> 8K token） | GSPO 更稳 |
| Qwen 系生态 | GSPO（官方默认） |
| 复现 R1 | GRPO（R1 用的是 GRPO） |
| Agentic（数十轮工具调用） | GSPO 或 trajectory-level 变体 |

**2025 趋势**：GSPO 在长 CoT、agentic 场景逐渐取代 GRPO。

---

## 六、其他 reasoning RL 算法

| 算法 | 来源 | 关键改进 |
|---|---|---|
| **RLOO** | Cohere 2024 | Leave-One-Out baseline（严格无偏的 group baseline） |
| **REINFORCE++** | 2024 | 无 critic 的 REINFORCE + 多个稳定技巧（normalize、clip、KL） |
| **GRPO++** | 社区 | GRPO + entropy bonus + advantage shaping |
| **DAPO** | ByteDance 2025 | 见 §四 |
| **GSPO** | Qwen 2025 | 见 §五 |
| **VAPO** | 2025 | Variance-aware PO，自适应 clip ε |
| **VC-PPO** | 2024 | Value-Calibrated PPO，long CoT 上 critic 的修补 |
| **PRM + RL** | OpenAI 2023+ | Process Reward Model：每步打分 |

→ 2024-2025 是 reasoning RL 算法的**寒武纪大爆发**，每个月都有新变体。但底层范式不变：**on-policy + group/baseline + clipped surrogate + verifiable reward**。

---

## 七、Process Reward Model（PRM）

### 7.1 PRM vs ORM

- **Outcome Reward Model (ORM)**：只看最终答案对错。Reward 在 sequence 末尾给。
- **Process Reward Model (PRM)**：给 CoT 每一步打分。Reward 在每步给。

```
Q: 求 2x + 3 = 7 的 x
Step 1: 移项 → 2x = 4    [PRM: 0.95]
Step 2: 除 2 → x = 2     [PRM: 0.98]
答案: x = 2              [ORM: 1.0]
```

### 7.2 PRM 数据获取

- **人工标**（昂贵）：[PRM800K (OpenAI 2023)](https://arxiv.org/abs/2305.20050)——80 万 step-level 标注，~$1M 成本；
- **自动标**：Math-Shepherd（rollout 多条，从 step 出发统计完成率回填 step 标签）；
- **MCTS 标**：树搜索探索每步价值，每步的 V 估计作为 PRM target。

### 7.3 PRM 的两种用法

**(1) 训练时 reward signal**：每 step 给 reward + 末尾 outcome reward：

$$
R_t = r_{\text{PRM}}(\text{step}_t) + \mathbb{1}[t = T] \cdot r_{\text{ORM}}
$$

**(2) 推理时 Best-of-N reranking**：sample N 条 CoT，用 PRM 选最优一条。或当 value head 引导 MCTS。

### 7.4 R1 vs o1 在 PRM 上的分歧

- **o1（OpenAI）**：据称用 PRM + MCTS（PRM800K 之后的延续）；
- **DeepSeek-R1**：**明确否定 PRM 路线**——发现：
  1. PRM 训练数据成本高；
  2. PRM 易被 hack（policy 学会"装出合理 step"骗 PRM）；
  3. R1 用纯 outcome reward + GRPO 反而更稳。

→ 2025 工业共识**逐渐偏向 outcome-only**（GRPO 简化 + verifiable）。但 PRM 在 **推理时 search**（best-of-N、MCTS）里仍是 SOTA。

### 7.5 PRM 的根本困难

PRM 给"每一步"打分听上去合理，但实际上：

- "什么算一步"没有客观定义（粒度问题）；
- "这一步对不对"在数学上需要看后续——孤立判断单步对错不可靠；
- 自动标注（Math-Shepherd 类）回填出来的 PRM 标签噪声大；
- PRM 的训练目标和 ORM 不一致——可能与最终目标偏离。

R1 团队的判断是："让模型自由 explore CoT，最后用 ORM 校验"是更鲁棒的设计。这条经验**改变了 2025 业界对 PRM 的态度**——从"未来主线"降级为"推理时 reranking 工具"。

---

## 八、推理蒸馏（R1-Distill 范式）

### 8.1 R1 的现象级发现

R1 论文报告：用 R1 (671B MoE) 生成的长 CoT trajectory 直接 SFT 给小模型，**小模型继承了推理能力**：

```
1. 用 DeepSeek-R1 (671B MoE) 生成 80w 条 (问题, 长 CoT 解答)
2. 在 Qwen-7B/14B/32B base 上做 SFT（不需 RL）
3. 得到 R1-Distill-Qwen-{7B/14B/32B}
```

结果：

- **R1-Distill-32B 在 AIME 上接近 o1-mini**（小模型史上第一次）；
- 不需要任何 RL，**纯 SFT**；
- 训练时间 / 算力比"从头 RL"少 **1-2 个数量级**。

### 8.2 为什么蒸馏比 RL 还有效（在小模型上）

直觉上，"小模型 SFT 大模型推理 trajectory"等价于：

- **大模型的 RL 探索**：在 trillions 个可能的 CoT 路径中找出"通往正确答案的"几条；
- **小模型直接学这几条路径**：相当于把 RL 探索过的好 trajectory 直接灌进去，跳过 explore 阶段。

为什么小模型直接 RL 反而不如蒸馏？

1. **小模型的 base 能力弱**——RL 初始时几乎所有 rollout 都错（reward 全 0），没信号训；
2. **小模型的 explore 能力有限**——找不到长 CoT 的好路径（容量不够）；
3. **大模型已经找到的路径对小模型仍是"可学的"**——小模型至少能模仿，即使无法自发产生。

形式化：把 reasoning 看成 trajectory $\tau$ 的分布问题。

- 大模型的 reasoning 分布 $\pi^*(\tau)$ 集中在好 trajectory；
- 小模型的 base 分布 $\pi_s(\tau)$ 分散；
- 直接 RL 小模型：从 $\pi_s$ explore，期望 reward 极低，梯度信号弱；
- SFT 蒸馏小模型：直接拟合 $\pi^*$，相当于 imitation learning，效率高 N 倍。

→ **大模型 RL → 小模型 distill** 成为 2025 标准 pipeline。

### 8.3 蒸馏的甜点：7B-32B

| 小模型规模 | 蒸馏效果 |
|---|---|
| 1B | 显著掉点（容量不足以承载长 CoT 模式） |
| 3B | 部分能力（GSM8K OK，AIME 弱） |
| **7B** | 甜点——AIME 30%+ |
| **14B** | AIME 50%+ |
| **32B** | AIME 70%+，接近 o1-mini |
| 70B | 接近 R1 本体 |

→ 7B 以下蒸馏效果显著下降，1.5B 以下几乎不 work（这也是为什么 R1-Distill-Qwen-1.5B 性能远低于 7B）。

### 8.4 与传统 CoT 蒸馏的对比

- **经典 CoT distill** (Orca 2023, Wizard 2023)：让小模型学短 CoT（几百 token），效果有限；
- **R1-Distill**：长 CoT（reasoning trace 几千-上万 token） + outcome 同时正确 → 质变。

差别的核心：**长度**。短 CoT 不够表达多步推理 + 反思 + 自我修正这套行为模式；长 CoT 是"reasoning 的容器"。

详见 §10.5 知识蒸馏的 R1-Distill 部分。

---

## 九、Reasoning RL 工程实战

### 9.1 数据集（公开）

- **MATH-500 / AIME / OlympiadBench / Math-OlympiadBench**：竞赛级数学；
- **GSM8K**：小学数学（已饱和，仅作 sanity check）；
- **CodeContests / LiveCodeBench / TACO**：代码；
- **OpenR1-Math** (HuggingFace 2025)：完整开源 R1 复现数据；
- **AceCoder / OpenCodeReasoning** (NVIDIA 2025)：开源 reasoning RL 代码数据；
- **Skywork-OR1**：完整 R1 复现链；
- **RUC-OpenR1 / Open-Reasoner-Zero**：学界 R1 复现。

### 9.2 训练成本（参考）

- **R1 完整训练**：估计 **数百万美元** GPU（DeepSeek 未公开精确数）；
- **R1-Distill-Qwen-7B**：单 8×H100 几天即可；
- **DAPO-Qwen-32B**：8×H100 几周；
- 中小团队普遍走 **distill 路线**（拿 R1 trajectory + SFT 训 7B/14B）。

### 9.3 训练监控指标

```
reward:           缓慢上升（每 100 step ~ +0.02）
group accuracy:   每 group 内 0-1 之间（接近 1 → 过简单；接近 0 → 过难）
gen length:       缓慢增长（reasoning 模型应该自发长 CoT）
KL:               很小（R1 用 0.001 量级）
clip frac:        10-30%（高了说明 ratio 漂移大，可能不稳）
entropy:          缓慢下降（policy 收敛），但保持 explore
```

异常信号：

- gen length 突然爆炸（> 32K）→ overlong reward shaping 失效；
- reward 飙升但 group accuracy 不变 → reward hacking（policy 在刷格式 reward）；
- entropy 骤降到 0 → policy collapse（KL 太小或 sample temperature 太低）。

---

## 十、关键问答

**Q1**：为什么 RLVR 比 RLHF 稳？
- Reward 是程序化的 0/1，无 RM 被 hack 风险；
- Reward signal 严格（verifier 判定确定），policy 找不到"刷分捷径"；
- 缺点：只能用于有 ground-truth 的任务（数学/代码/形式逻辑），通用 chat 仍需 RM 或 LLM-judge。

**Q2**：GRPO 与 PPO 的差距？
- GRPO 无 critic、用 group baseline；
- 显存省 ~40%、训练快 1.5-2×；
- 推理任务上效果 ≥ PPO（R1 实测）；
- 通用 chat 上略逊（baseline noise 大）。

**Q3**：R1-Zero 真的没 SFT 吗？
- 论文报告"直接 base model + RL"——没用 instruction-tuning SFT 步；
- 但 base model 已有大量代码/数学 pretrain，所以严格说是"无后训练 SFT"，不是"完全 from scratch"；
- 这点 R1 论文写得清楚。

**Q4**：长 CoT 输出会拖慢推理吗？
- 是。reasoning 模型输出 1k-32k thinking tokens；
- 推理成本：thinking token 数 × token cost，可能比 chat 模型贵 10×；
- 部署需考虑：是否让用户看 think 内容、latency 预算、是否需要"快慢思考切换"。

**Q5**：PRM 还有用吗？
- 在 search-based 推理（MCTS、Best-of-N）里仍重要；
- 在 RL 训练阶段，2025 工业主流是 **outcome-only**（GRPO 简化 + verifiable）；
- PRM 的训练成本与 hack 风险让大家暂时回避。

**Q6**：DAPO、GSPO、GRPO 我应该选哪个？
- 复现 R1 / 短 CoT 数学 RL：**GRPO** 是默认起点；
- Long CoT 不稳 / 想刷 leaderboard：**DAPO**；
- Qwen 系生态 / 极长 CoT / agentic：**GSPO**；
- 三者都是工业可用，差异在工程稳定性，不在天花板。

**Q7**：reasoning 能力可以蒸馏到 7B 吗？
- 可以，R1-Distill-Qwen-7B 在 MATH 上接近 o1-mini；
- 但 1-3B 蒸馏效果显著下降（容量不足以承载长 CoT 模式）；
- 7B-32B 是当前 reasoning distill 的甜点。

**Q8**：为什么 group baseline 是 unbiased？
- REINFORCE 允许减去任何只依赖 $x$ 的 baseline，期望梯度不变；
- group mean 是 $r(x, y)$ 在 batch 内的估计；leave-one-out 严格无偏，group mean 包含自身有微小偏差但实战无影响；
- 关键好处：方差降低（同 prompt 的 reward 都聚集在 baseline 附近，差值方差小）。

**Q9**：RL 训出来的 reasoning 是真的"思考"吗？
- 神经科学意义上不是——模型在 sample 子分布；
- 但行为上模型确实表现出"多步推理、回溯、验证"——这些行为在 RL 强化下是真实的、可重复的；
- "是否真思考"是哲学问题，实证上 reasoning RL 模型在 MATH/AIME 上正确率显著高于 base + CoT，这是工程意义上的"真"。

**Q10**：reward sparse（5% 题目能对）怎么办？
- **降难度**：用更简单的题目起步，rollout 有 30-70% 对率（reward 有差异）；
- **课程学习**：从简单到难逐步过渡；
- **DAPO dynamic sampling**：过滤掉全错的 prompt；
- **Cold-start SFT**：先用人挑数据 SFT base，给个能力起点。

**Q11**：能不能完全跳过 SFT、跳过 RM、直接 base + RLVR？
- 可以（R1-Zero 证明）；
- 但可读性差，不能直接生产用；
- 工业上 R1-Zero 之后还要 cold-start SFT + 二次 RL 才出货。

**Q12**：GRPO 的 group size G 怎么调？
- G 越大方差越小，但 rollout 时间线性增加；
- 典型 G = 16；硬件够的话 G = 64 效果略好；
- G < 4 baseline 噪声大，不建议。

---

## 十一、本节与其他节关系

```
§11.1 SFT
   ↓
§11.2 PPO  ─┐
§11.3 DPO  ─┴── 经典对齐（learned reward）
   ↓
§11.4 RLVR / GRPO (本节) ── reasoning 新主线（verifiable reward）
   ↓
§11.5 Agentic RL          ── 多轮工具调用 + 长 horizon
   ↓
§10.5 蒸馏 (R1-Distill)    ── 把大模型推理能力压到小模型
```

GRPO 是 PPO 的减法（去 critic）+ DPO 的补法（保留 RL on-policy 优势）的混合。理解路径：

- PPO 是母算法（§11.2）；
- DPO 是 offline 简化（§11.3）；
- GRPO 是 critic 简化 + verifiable reward（§11.4）；
- DAPO / GSPO 是 GRPO 在长 CoT 上的稳定性补丁。

---

## 十二、参考资料

**开山 / 现象级**：
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) ⭐⭐（reasoning RL 现象级）
- [DeepSeekMath — GRPO 原始论文 (2024)](https://arxiv.org/abs/2402.03300) ⭐⭐
- [OpenAI o1 system card](https://openai.com/index/learning-to-reason-with-llms/) ⭐
- [Let's Verify Step by Step — PRM (OpenAI 2023)](https://arxiv.org/abs/2305.20050) ⭐⭐

**算法变体**：
- [DAPO (ByteDance 2025)](https://arxiv.org/abs/2503.14476) ⭐⭐
- [GSPO (Qwen 2025)](https://arxiv.org/abs/2507.18071) ⭐
- [REINFORCE++ (2024)](https://arxiv.org/abs/2501.03262)
- [VinePPO / VC-PPO long CoT (2024)](https://arxiv.org/abs/2410.01679)
- [Tülu-3 — RLVR 命名 (AllenAI 2024)](https://arxiv.org/abs/2411.15124) ⭐⭐

**PRM / 自动数据**：
- [Math-Shepherd — 自动 PRM 数据 (2024)](https://arxiv.org/abs/2312.08935)
- [Process Reward Model 综述 (2024)](https://arxiv.org/abs/2410.08146)

**开源复现 / 工具**：
- [HuggingFace open-r1 项目](https://github.com/huggingface/open-r1) ⭐⭐
- [Skywork-OR1 (开源 R1 复现)](https://github.com/SkyworkAI/Skywork-OR1)
- [Open-Reasoner-Zero (RUC 2025)](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)
- [OpenRLHF (含 GRPO/DAPO)](https://github.com/OpenRLHF/OpenRLHF) ⭐
- [VeRL (ByteDance, DAPO 母框架)](https://github.com/volcengine/verl) ⭐
- [TRL (含 GRPO trainer)](https://github.com/huggingface/trl)

**评测**：
- [AIME / MATH / OlympiadBench 系列 benchmarks](https://github.com/QwenLM/Qwen2.5-Math)
- [LiveCodeBench](https://livecodebench.github.io/)

**深度博客**：
- [Lilian Weng — Reward Hacking in RL (2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) ⭐⭐
- [Nathan Lambert — RLVR / R1 解读](https://www.interconnects.ai/) ⭐
- [The Bitter Lesson Strikes Reasoning (Sutton 风格 blog 2024)](https://lilianweng.github.io/) 
- [Sebastian Raschka — GRPO/R1 解读](https://magazine.sebastianraschka.com/) ⭐
