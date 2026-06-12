# 14.5 后训练 & Reasoning 高频题（§11）

[← 返回框架](../../README.md) · [📎 materials.md → §14.5](../../materials.md)

---

## 〇、本节回答什么

> 2024-2026 最火、面试必问深水区：SFT、RLHF/PPO、DPO 家族、GRPO/RLVR、R1 范式、Agentic RL。
>
> 覆盖 §11 后训练全栈。能把 DPO 损失推一遍、能讲清 PPO 为什么 unstable、能说出 GRPO 和 PPO 的区别——是 reasoning model 团队的硬门槛。

---

## 一、SFT（→ §11.1）

### Q1.1 SFT 数据该怎么准备？ ⭐

**【答 30s】**
- 来源：人工写（如 Open-Assistant、Anthropic HH-RLHF SFT 子集）、teacher 蒸馏（如 R1 → Qwen 800K）、自动合成（Self-Instruct）
- 质量 > 数量：LIMA (Zhou 2023, NeurIPS) **1K 高质量样本**就能调出好 chatbot
- 形式：`{instruction, input?, output}` 三元组；多轮加 `[USER]/[ASSISTANT]` 标签
- chat template：每家不同（ChatML / Llama-3 / DeepSeek format）

**【加分项】**
- LIMA "Less is More for Alignment" 论点：SFT 主要是教格式不是教能力
- 数据多样性 > 数据量；类型覆盖（reasoning / coding / safety / refusal）比堆数据重要
- Loss 只算 assistant token，user prompt 部分 mask 掉

**【追问】**
1. 多轮怎么算 loss？→ 只算 assistant 段；user 段 ignore_index=-100
2. SFT 会不会损害 base 能力？→ 会，叫"alignment tax"；Llama-2、Qwen 都讨论过
3. Packing 是什么？→ 把多条短样本拼到一起填满 seq_len，提高训练效率

---

### Q1.2 SFT vs continued pretrain 怎么选？ ⭐⭐

**【答 30s】**
- **continued pretrain**：raw text、不分 user/assistant、目标是注入领域知识
- **SFT**：QA 对、教格式 + 表达方式
- 顺序：通常 base → continued pretrain → SFT → RLHF/DPO → reasoning RL
- 注入大量领域知识应在 continued pretrain；少量行为偏好用 SFT

**【追问】**
1. continued pretrain 多大数据量？→ 10B-100B token 量级
2. 学习率怎么设？→ 比 pretrain 低（如 1e-5），避免冲刷已有能力
3. CPT 和 SFT 的 catastrophic forgetting 怎么办？→ replay 部分原 pretrain 数据

---

## 二、RLHF + PPO（→ §11.3）

### Q2.1 RLHF 三阶段是什么？ ⭐

**【答 30s】**
1. **SFT**：监督微调 base 模型
2. **Reward Model**：用 pairwise preference 数据训 RM，loss = `-log σ(r(x,y_w) - r(x,y_l))`
3. **PPO**：用 RM 当 reward，在 prompt 上做 RL，让模型 maximize reward 同时被 KL 约束在 SFT policy 附近

**【加分项】**
- RM 通常用 SFT model + 一个 linear head（scalar 输出）
- PPO 在 LLM 里的目标：`max E[r(x,y)] - β·KL(π||π_sft)`
- InstructGPT (Ouyang 2022) 是 RLHF 经典框架

**【追问】**
1. 为什么用 pairwise 而非 absolute score？→ 人类一致性差，相对偏好更可靠（Bradley-Terry 模型）
2. RM 训不好的症状？→ Reward hacking，policy 学会奇怪话术骗 RM
3. KL 系数 β 怎么调？→ 自适应（KL controller），目标 KL 通常 6-15 nats

---

### Q2.2 PPO 在 LLM 上的目标函数怎么写？ ⭐⭐⭐

**【答 30s】**

$$
L_\text{PPO}(\theta) = \mathbb{E}_t\!\left[\min\big(\rho_t(\theta) A_t,\ \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\right]
$$

- $\rho_t = \pi_\theta(a_t|s_t) / \pi_\text{old}(a_t|s_t)$ 是重要性采样比
- $A_t$ 是 advantage（GAE 估计）
- clip 防止 policy 更新过大

LLM 里再加 KL penalty：

$$
\text{Reward}_t = r(x,y) - \beta \cdot \log \frac{\pi_\theta}{\pi_\text{SFT}}
$$

**【加分项】**
- Critic（value head）也要训，loss = MSE
- LLM PPO 用 token-level reward：r 在末尾给，KL 每 token 给
- 是不是 on-policy？理论 on-policy，工程上用 mini-batch 多次更新所以是近 on-policy

**【追问】**
1. 为什么 PPO 比 vanilla policy gradient 稳？→ Clip + KL，避免 catastrophic update
2. critic 必须吗？→ 经典 PPO 要；GRPO/RLOO 把 critic 去了，用 group baseline
3. PPO 工程坑？→ 4 个网络（actor / critic / RM / ref）显存爆炸；reward hacking；KL controller 不稳

---

### Q2.3 RLHF 为什么 unstable？工程上的关键技巧？ ⭐⭐⭐

**【答 30s】**
- **不稳来源**：
  1. **reward 稀疏**（只在末尾）+ token-level credit assignment 难
  2. **policy 容易 mode collapse**（学会重复几句话骗 RM）
  3. **RM 易被 hack**（OOD prompt 让 RM 给莫名高分）
  4. **超参敏感**（KL β、lr、clip ε、batch 全敏感）
- **技巧**：
  - 严格 KL 控制 + adaptive KL controller
  - 多个 RM ensemble、用 mean - std 当 reward
  - 限制 generation length 防长度 hack
  - PPO ratio clip + value clip

**【加分项】**
- "Secrets of RLHF" (Zheng 2023) 列了几十个坑
- OpenAI 内部用 reward hacking 检测器
- Anthropic 用 Constitutional AI + RLAIF 缓解部分人工标注瓶颈

**【追问】**
1. Reward hacking 怎么发现？→ 人评 vs RM 评分背离、生成文本变奇怪
2. PPO 训练 1B 模型要多少卡？→ 至少 4-8 张 A100/H100（actor+critic+RM+ref）
3. 现代为什么很多公司转 DPO？→ 复杂度低、不需要 critic 和 reward model

---

## 三、DPO 家族（→ §11.4）

### Q3.1 DPO 的目标函数和直觉？ ⭐⭐⭐

**【答 30s】**

DPO (Rafailov 2023 NeurIPS Best Paper) 推导：

$$
L_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]
$$

- **直觉**：把 RLHF 中"训 RM + PPO 优化"两步合并成一步监督学习
- 用 closed-form 解析消去 RM：reward 隐式定义为 `r(x,y) = β log(π_θ/π_ref) + Z(x)`
- 训练时只需 preference 对，不需要 RM、不需要 sampling

**【加分项】**
- 推导关键：从 PPO 最优解 `π* ∝ π_ref · exp(r/β)` 反推 r 用 π 表达
- β 控制偏离 π_ref 的程度，typical 0.1-0.5
- 不需要 critic、不需要 rollout、显存友好（只 2 倍 ref policy）

**【追问】**
1. 为什么 DPO 是 implicit reward？→ 不显式训 RM，π 自身扮演 RM
2. DPO 在 reasoning 上 vs PPO？→ 简单任务 DPO 不输 PPO；复杂 reasoning（如 AIME）PPO/GRPO 更强
3. DPO 失败模式？→ 容易降低 chosen 概率（同时降 rejected 但 chosen 降更慢），见 "DPO Likelihood Decreasing" 现象

---

### Q3.2 DPO 变种：IPO / KTO / SimPO / ORPO 区别？ ⭐⭐⭐

**【答 30s】**

| 变种 | 改进点 |
| --- | --- |
| **IPO** (Azar 2023) | 修正 DPO 对完美偏好假设的过拟合，用 squared loss |
| **KTO** (Ethayarajh 2024) | 不需要 pair，只需要 "good/bad" 标签（更易标注） |
| **SimPO** (Meng 2024) | 不需要 ref policy，用 length-normalized log-prob |
| **ORPO** (Hong 2024) | 直接对 SFT loss 加 odds ratio penalty，one-stage 训练 |

- KTO 在工业 RLHF 数据收集上很实用（人不需要标 pair）
- SimPO 省一个 ref policy 显存
- ORPO 把 SFT + DPO 合一，最省资源

**【追问】**
1. IPO 解决了什么？→ DPO 有完美偏好假设 σ(...) → 1 时梯度爆炸，IPO 改 squared 不会
2. SimPO 不用 ref policy 有没有问题？→ 容易偏离原 policy 太多，建议加 SFT loss 正则
3. ORPO 怎么 one-stage？→ loss 同时包含 NLL（SFT）+ odds ratio（preference）

---

### Q3.3 DPO vs PPO 实践怎么选？ ⭐⭐

**【答 30s】**

| 维度 | DPO | PPO |
| --- | --- | --- |
| 工程复杂度 | **简单** | 复杂（4 net、rollout） |
| 显存 | 低（2× ref） | 高（actor+critic+RM+ref） |
| 数据要求 | offline preference | online sampling 多 |
| reasoning 上限 | 中 | **高**（结合 RLVR） |
| 调参 | 容易 | 难 |

**经验**：偏好对齐（safety / helpfulness）DPO 足够；reasoning（math / code）必须 RL（PPO / GRPO + verifiable reward）。

**【追问】**
1. 能不能 DPO + RL 混合？→ 可以，TÜLU 3 用 DPO 做 alignment，再 RLVR 提 reasoning
2. DPO 数据从哪来？→ RM 打分自动配 pair / 人工标 / GPT-4 当 judge
3. 在线 DPO？→ Online DPO / Online IPO，每轮 sample 新 pair 比 offline 强

---

## 四、Reasoning 训练 / RLVR / GRPO（→ §11.5）

### Q4.1 RLVR 是什么？为什么 reasoning 必须用它？ ⭐⭐⭐

**【答 30s】**
- **RLVR** = Reinforcement Learning with **Verifiable Reward**
- 在数学 / 代码任务上，**reward 可以自动验证**（数学答案正确 = 1，代码通过测试 = 1），不需要 RM
- 好处：
  - 没有 reward hacking（reward 是 ground truth）
  - 可以无限 scale online sampling
  - 跨任务一致

**【加分项】**
- DeepSeek-R1-Zero 直接在 base 上 RLVR（**不需要 SFT**），证明 reasoning 可以从 RL 涌现
- AIME / MATH / SWE-bench 都可以做 verifier
- TÜLU 3 (Allen AI) 把 RLVR 用在 IFEval 这种规则可验证任务

**【追问】**
1. 没有 verifier 怎么办？→ 用 LLM judge（reward model），但容易 hack
2. RLVR 和 self-play 关系？→ RLVR 是 self-improvement 的一种特殊形式
3. RLVR 极限？→ 受限于可验证任务，开放任务难做

---

### Q4.2 GRPO 和 PPO 区别？ ⭐⭐⭐

**【答 30s】**

GRPO (Group Relative Policy Optimization, DeepSeek-Math 2024):

$$
A_i = \frac{r_i - \mathrm{mean}(r_1, ..., r_G)}{\mathrm{std}(r_1, ..., r_G)}
$$

- 同 prompt 采 G 个 sample，用 **group mean 做 baseline**（替代 critic）
- 去掉 PPO 的 critic network，显存砍 ~50%
- 其余目标函数和 PPO clip + KL 一样

**【加分项】**
- GRPO 是 DeepSeek-R1 的核心算法
- 优势：省显存、训练稳定、不需要 value head 收敛
- 劣势：高 variance（小 G 时），需 G≥8

**【追问】**
1. 为什么 group baseline 行？→ 同 prompt 下 reward 差异主要来自 policy 选择，group mean 是无偏 baseline
2. G 怎么选？→ 8-64，G 越大 variance 越低但成本线性涨
3. KL 怎么算？→ 对 ref policy 做 token-level KL（reverse 或 forward）

---

### Q4.3 R1 训练流程是什么？ ⭐⭐⭐

**【答 30s】**

DeepSeek-R1 (2025-01) 多阶段：
1. **R1-Zero**：从 V3 base 直接 GRPO + RLVR（数学/代码 verifiable reward），出现长 CoT 涌现 + "aha moment"
2. **Cold Start SFT**：用 R1-Zero 生成 + 人工筛选的高质量 CoT 做 SFT
3. **Reasoning RL**：再做一轮 GRPO + RLVR
4. **Rejection Sampling SFT**：从 RL 模型筛优质 trajectory 做 SFT
5. **Final RL**：综合任务（reasoning + general + safety）

**【加分项】**
- R1 在 AIME 2024 达到 79.8% pass@1，接近 o1
- "Aha moment"：R1-Zero 自主学会"等等，让我重新检查"这种 reflection token
- R1 → Qwen-7B/14B/32B 蒸馏出来的 reasoning 小模型也很强

**【追问】**
1. 为什么不直接一阶段 RL？→ R1-Zero 可读性差（中英混、格式乱），SFT 是为了可读
2. R1 比 o1 弱在哪？→ 部分 agent / long-context reasoning；优势在开源、可复现
3. 推理蒸馏到 Qwen-32B 能保留多少？→ AIME ~70%、MATH ~94%，接近 R1 本身

---

### Q4.4 DAPO / GSPO / RLOO 等新算法演进？ ⭐⭐⭐

**【答 30s】**
- **DAPO** (ByteDance 2025)：动态 sampling + clip-higher（不对称 clip）+ overlong penalty，AIME 2024 50% w/ Qwen-32B
- **GSPO** (Alibaba Qwen3 2025)：sequence-level 替代 token-level clipping，更稳
- **RLOO** (Ahmadian 2024 ACL Best Paper)：Reinforce-style baseline（leave-one-out），简单且能 match PPO
- 共同趋势：**去 critic + 改 advantage 算法 + 改 clip 策略**

**【加分项】**
- GSPO 解决长序列下 token clipping 不一致问题（Qwen3 用）
- DAPO clip-higher：clip 上限 0.28、下限 0.2，鼓励 exploration
- 后训练这条线 2025-2026 演进非常快，每月新算法

**【追问】**
1. 这些算法间有什么共性？→ 都是 PPO 变体，简化或更高效
2. 工业上一般选哪个？→ GRPO 仍是主流，DAPO/GSPO 是改进
3. 怎么知道自己 RL 训没训稳？→ 看 KL 曲线、reward 上升、entropy 不崩

---

## 五、Agentic RL（→ §11.6）

### Q5.1 Agentic RL 和 RLHF 区别？ ⭐⭐

**【答 30s】**
- **RLHF**：单轮（prompt → answer），reward 在末尾
- **Agentic RL**：多轮（tool use / web browse / code exec），每步要决策 + 工具调用 + 环境反馈
- 关键挑战：
  1. **长 horizon credit assignment**
  2. **tool calling 解析 + 错误恢复**
  3. **环境状态管理**（partial observability）

**【加分项】**
- Search-R1、ToolRL、Agent-R1 是 2025 代表
- SWE-bench Verified 上做 RL 是 hottest topic
- ReAct / CodeAct / Reflexion 是 scaffolding，不一定要 RL

**【追问】**
1. agent RL 训练数据从哪来？→ rollout + verifier（如 SWE-bench unit test）
2. 单轮 vs 多轮 PPO 差别？→ 多轮 advantage 估计难度大，需 trajectory-level reward shaping
3. 工业落地？→ Anthropic Claude Code、OpenAI Operator 都用 agent RL

---

## 六、Constitutional AI / RLAIF（→ §11.7）

### Q6.1 CAI 是什么？ ⭐

**【答 30s】**
- Anthropic Constitutional AI (Bai 2022)：用 LLM 替代人类做 RLHF 标注
- 两阶段：
  1. **SL-CAI**：让模型按 constitution 自我批评 + 改写，得到 self-improved 数据 SFT
  2. **RL-CAI**：用 LLM 当 judge 生成偏好 → 训 RM → PPO
- 优势：减少人类标注、可控 safety 行为；缺点：constitution 写法敏感

**【追问】**
1. RLHF vs RLAIF 性能差距？→ Lee 2023 显示 RLAIF 接近 RLHF，部分任务超过
2. constitution 怎么写？→ Anthropic 公开了 ~70 条原则，覆盖 helpfulness / harm / honesty
3. Claude 3/4 还用 CAI 吗？→ 大方向继承，细节迭代

---

## 七、关键问答（综合）

### Q7.1 一句话说"为什么 reasoning 模型一定要 RL"

> SFT 教格式不教探索，CoT 长度和深度受限于训练数据；RL 让模型自己试错产生超越数据分布的 reasoning trajectory（"aha moment"），且 RLVR 的可验证 reward 天然防 hacking——这是 SFT 永远做不到的。

### Q7.2 PPO 和 GRPO / DPO 显存对比？

> 同 7B 模型 actor：PPO 4 net（actor+critic+RM+ref）~100GB；GRPO 3 net（去 critic）~70GB；DPO 2 net（actor+ref）~30GB。所以小团队优先 DPO；做 reasoning 必须上 GRPO/PPO。

### Q7.3 R1-Zero 的"aha moment"是什么？

> R1-Zero 训练中后期，policy 自发产生"Wait, let me reconsider this"这种 reflection token，说明 RL 让模型涌现出**自我检查能力**——这是 SFT 蒸馏不可能学到的，因为 SFT 数据里没有。这被认为是 reasoning RL 的 Sutton "bitter lesson" 时刻。

### Q7.4 RLHF 里 KL 为什么必须？

> 没 KL，policy 会狂奔到 RM 的高 reward 区域（且 RM 在 OOD 通常给奇怪高分），输出退化（重复、空话、骗 RM）。KL 拉住 policy 在 SFT 分布附近，相当于"信任域"约束。

### Q7.5 GRPO 没 critic 怎么做 advantage？

> 同 prompt 采 G 个 sample，用 group mean / std 归一化每个 sample 的 reward 得到 advantage（z-score）。等价于把"critic 估计的 baseline"换成"empirical group baseline"，前提是 G 足够大（≥8）。

### Q7.6 给你 8 张 H100，怎么训一个 reasoning 32B 模型？

> base 选 Qwen2.5-32B-base → continued pretrain (数学 / 代码 textbook ~50B token) → SFT (R1 蒸馏 + 自筛 100K reasoning) → **GRPO + RLVR**（数学/代码 verifier, G=16, batch ~64）→ rejection sampling SFT → 最终 RL 综合任务。预算约 2 周。

### Q7.7 如何判断你的 RL 训练崩了？

> 4 个信号：(1) reward 飙升但人评看不到提升 → reward hacking；(2) KL 爆炸（>50 nats）→ policy 跑飞；(3) entropy 跌到 0 → mode collapse；(4) gradient norm 爆 → 学习率/clip 调太大。监控这 4 条 + 每 N 步采样人看输出。

---

## 八、参考资料

### SFT
- ⭐⭐ [LIMA (Zhou 2023 NeurIPS)](https://arxiv.org/abs/2305.11206)
- ⭐ [Open-Assistant / Dolly / Alpaca / WizardLM 数据集](https://huggingface.co/datasets)

### RLHF + PPO
- ⭐⭐ [InstructGPT (Ouyang 2022)](https://arxiv.org/abs/2203.02155)
- ⭐⭐ [Anthropic HH-RLHF (Bai 2022)](https://arxiv.org/abs/2204.05862)
- ⭐⭐ [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347)
- ⭐⭐ [Secrets of RLHF (Zheng 2023)](https://arxiv.org/abs/2307.04964)
- ⭐ [The 37 implementation details of PPO (HuggingFace 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### DPO 家族
- ⭐⭐ [DPO (Rafailov 2023 NeurIPS Best Paper)](https://arxiv.org/abs/2305.18290)
- ⭐⭐ [IPO (Azar 2023)](https://arxiv.org/abs/2310.12036)
- ⭐⭐ [KTO (Ethayarajh 2024)](https://arxiv.org/abs/2402.01306)
- ⭐⭐ [SimPO (Meng 2024)](https://arxiv.org/abs/2405.14734)
- ⭐ [ORPO (Hong 2024)](https://arxiv.org/abs/2403.07691)

### RLVR / GRPO / R1
- ⭐⭐ [DeepSeekMath / GRPO (Shao 2024)](https://arxiv.org/abs/2402.03300)
- ⭐⭐ [DeepSeek-R1 Tech Report (2025-01)](https://arxiv.org/abs/2501.12948)
- ⭐⭐ [TÜLU 3 (Allen AI 2024)](https://arxiv.org/abs/2411.15124)
- ⭐⭐ [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)
- ⭐ [DAPO (ByteDance 2025)](https://arxiv.org/abs/2503.14476)
- ⭐ [GSPO (Qwen3 2025)](https://arxiv.org/abs/2503.20783)
- ⭐ [RLOO (Ahmadian 2024 ACL Best Paper)](https://arxiv.org/abs/2402.14740)

### Agentic RL
- ⭐⭐ [Search-R1 (2025)](https://arxiv.org/abs/2503.09516)
- ⭐ [ToolRL / Agent-R1 (2024-2025)](https://github.com/)
- ⭐ [SWE-RL (FAIR 2025)](https://arxiv.org/abs/2502.18449)

### CAI
- ⭐⭐ [Constitutional AI (Bai et al., Anthropic 2022)](https://arxiv.org/abs/2212.08073)
- ⭐ [RLAIF (Lee 2023)](https://arxiv.org/abs/2309.00267)

### 框架
- ⭐⭐ [TRL (HuggingFace)](https://github.com/huggingface/trl)
- ⭐⭐ [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- ⭐⭐ [verl (字节)](https://github.com/volcengine/verl)

### 解读
- ⭐⭐ [Yi Tay — Post-training survey](https://www.yitay.net/)
- ⭐⭐ [Sebastian Raschka — DPO 系列](https://magazine.sebastianraschka.com/)
- ⭐⭐ [Nathan Lambert / Interconnects.ai — RLHF 时事评](https://www.interconnects.ai/)

---

→ 下一节 [14.6 推理优化题](./06-推理优化题.md)
