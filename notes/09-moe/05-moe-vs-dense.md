# 9.5 MoE vs Dense 选型与实例

[← 返回框架](../../README.md) · [📎 materials.md → §9.5](../../materials.md)

---

## 〇、本节回答什么

§9.1-9.4 已经讲完 MoE 怎么做、怎么训、怎么部署。本节是 **决策**：

> 给定一个具体目标（质量、成本、context 长度、推理硬件），到底**用不用 MoE**？用什么规模？

本节先用 **Scaling Law** 给出选型的理论基础（不只是经验法则），然后落到具体场景，最后讨论几个常被忽略的"边角"问题（PEFT、Upcycling、long context）。

---

## 一、MoE vs Dense 的本质差异

### 1.1 基本对照

```
                  Dense           MoE
                  ─────           ───
总参数量          P               P_total
单 token 算力     ≈ P             ≈ P_active (= P_total × K/N + shared)
单 token 显存读   P (KV+W)        P_total (W) + KV
                                   ↑ 推理时全 expert weight 都要驻留显存
通信开销          全梯度同步      额外 All-to-All（4 次/层）
质量上限          高              更高（同算力，按 scaling law）
推理延迟（decode） 低              高（除非 expert replicate）
```

### 1.2 用一个数字总结

> **MoE 的核心交易**：用 N× 显存 + 额外通信，换 ~3-7× 等效参数容量。

- 显存充裕 + 推理 batch 大 → MoE 划算
- 显存紧张 + 推理 batch 小 → dense 划算

### 1.3 显存账（关键）

|  | Dense 70B (BF16) | MoE 671B / 37B active (FP8) |
|---|---|---|
| 模型权重 | 140 GB | 670 GB |
| KV cache (per request, 4k tokens) | ~4 GB | ~2 GB（MLA 压缩） |
| 单 token 算力（FLOPs） | 140 GFLOPs | 74 GFLOPs |
| 单卡可装 | 单 H100 80GB 装不下，需 2 卡 | 单 H100 装不下，需 8+ 卡 |
| 总显存（serving） | ~150 GB | ~720 GB |

→ MoE 的显存代价巨大。即使激活参数小，**所有 expert 都得在显存里**（因为不知道下一个 token 会路由到哪）。

---

## 二、MoE Scaling Law（理论基础）

### 2.1 Clark et al. 2022 — Unified Scaling Law

[Clark et al. 2022](https://arxiv.org/abs/2202.01169) 把 dense 和 MoE 的 scaling law 统一在同一个公式里：

$$L(N, N_a, D) = c + \frac{a}{N^{\alpha}} + \frac{b}{N_a^{\beta}} + \frac{e}{D^{\delta}}$$

- $L$：验证 loss
- $N$：总参数（MoE 时就是 $N_{\text{total}}$）
- $N_a$：激活参数（dense 时 $N_a = N$）
- $D$：训练 token 数
- $a, b, c, e, \alpha, \beta, \delta$：拟合常数

**Dense 退化**：当 $N = N_a$（每参数都激活），公式合并为传统 Chinchilla loss：

$$L(N, D) = c + \frac{a + b}{N^{\alpha}} + \frac{e}{D^{\delta}}$$

**MoE 不同**：$N$ 和 $N_a$ 独立 scaling，$N$ 提供"参数容量"，$N_a$ 提供"算力"。

### 2.2 关键洞察：两个独立的 scaling 斜率

Clark 2022 拟合给出（在他们的设置下）：

- $\alpha \approx 0.34$（总参数 scaling 指数）
- $\beta \approx 0.28$（激活参数 scaling 指数）
- $\delta \approx 0.28$（数据 scaling 指数）

含义：

- 单独涨总参（保持激活）→ loss 按 $1/N^{0.34}$ 降
- 单独涨激活（保持总参）→ loss 按 $1/N_a^{0.28}$ 降
- 总参数 scaling 比激活参数 scaling 更高效（$\alpha > \beta$）

→ "白嫖参数"在 scaling law 里有理论支撑：**总参数比激活参数 scaling 更高效**。

### 2.3 Krajewski et al. 2024 — 加入 Granularity

[Krajewski et al. 2024](https://arxiv.org/abs/2402.07871) 进一步加入 granularity $G$（见 §9.3）：

$$L(N, N_a, D, G) = c + \frac{a}{N^{\alpha}} + \frac{b}{N_a^{\beta}} + \frac{e}{D^{\delta}} + \frac{h}{G^{\gamma}}$$

其中 $\gamma \approx 0.15$。

→ 这给 DSv3 的"细粒度 + 大 N"提供了**联合优化**的理论指导。

### 2.4 Compute-Optimal MoE

Chinchilla 给出 dense compute-optimal $D \approx 20 N$（每参数 20 token）。

MoE compute-optimal（Clark 2022 拟合）：

- 对总参数 $N$：$D \propto N^{0.5}$（与 dense 类似）
- 对激活参数 $N_a$：$D \propto N_a^{0.3}$（数据需求增长更慢）

含义：**MoE 在固定算力下，应该选择更大的总参数 + 适度训练 token**，而不是 dense 风格的 1:20。

DSv3 实际：671B 总参 × 14.8T tokens ≈ 22× tokens per total param——略低于 Chinchilla 比例，但远高于 MoE compute-optimal（说明 DSv3 用了"过量训练"换质量）。

### 2.5 经验放大率（重新理解）

> **Switch Transformer 经验法则**：同 FLOPs 的 MoE 模型，质量 ≈ **3-7×** 参数量的 dense 模型。

现在我们可以用 scaling law 推出这个数字：

在 Clark 2022 的拟合下，"等效 dense 大小" $N_{\text{eq}}$ 满足：

$$\frac{a + b}{N_{\text{eq}}^{\alpha}} = \frac{a}{N^{\alpha}} + \frac{b}{N_a^{\beta}}$$

代入 DSv3 的 $N = 671B, N_a = 37B$，解得 $N_{\text{eq}} \approx 120-250B$。这个范围**和经验法则一致**——MoE 不是免费午餐，但 scaling law 解释了它为什么有效。

---

## 三、按场景选

### 3.1 训练规模决定

| 总训练算力 | 推荐 |
|----------|------|
| < $10^{22}$ FLOPs（~7B dense） | **Dense** 优先（MoE 收益小，工程复杂度大） |
| $10^{22} \sim 10^{23}$ FLOPs（~70B） | **MoE 起步**（DeepSeek 路线），dense 仍可竞争 |
| > $10^{23}$ FLOPs | **MoE 优先**，几乎所有 SOTA 都 MoE |

**为什么 < 1B 几乎没人用 MoE**：

- 小模型的瓶颈是参数容量绝对量，不是参数效率
- MoE 的 N 倍参数容量在小规模下绝对值仍然太小
- 工程复杂度 / training instability 不划算

### 3.2 推理硬件决定

| 推理硬件 | 推荐 |
|--------|------|
| 单卡消费级（4090, 24GB） | **Dense ≤ 14B**，MoE 装不下 |
| 单机 8 卡 H100 | Dense 70B 或 **MoE ≤ 200B**（INT4 量化） |
| 多机 H100 集群 | MoE 任意，**DeepSeek-V3 671B 标准配置** |
| CPU 推理 | Dense 优先（MoE 通信开销在 CPU 上更显著） |
| 端侧 / 移动 | **Dense 小模型**（< 3B） |

### 3.3 任务类型决定

| 任务 | Dense 表现 | MoE 表现 |
|------|----------|---------|
| 通用对话 | 良 | 优（容量大，知识广） |
| 数学 / 代码 | 良 | 优（细粒度专家可专门化） |
| 长 context retrieval | 良 | 良（与 MoE 正交，看 attention 设计） |
| 多语言 | 良 | **优**（不同 expert 可专攻语种） |
| 强 reasoning（CoT） | 优 | 良（routing 离散是否破坏 reasoning 连贯性？争议中） |
| 极低延迟 serving | **优** | 略弱（All-to-All） |

---

## 四、2024-2026 主流模型横评

| 模型 | 总参 | 激活参 | Expert 数 | top-K | Attention | 关键路由策略 |
|------|------|-------|----------|-------|-----------|------------|
| **Mixtral 8x7B** (2023-12) | 47B | 13B | 8 | 2 | GQA | aux loss |
| **Mixtral 8x22B** (2024-04) | 141B | 39B | 8 | 2 | GQA | aux loss |
| **DBRX** (Databricks 2024) | 132B | 36B | 16 | 4 | GQA | aux loss + z-loss |
| **DeepSeek-MoE 16B** (2024-01) | 16B | 2.8B | 64 + 2 shared | 6 | MHA | aux loss + 细粒度+共享 |
| **DeepSeek-V2** (2024-05) | 236B | 21B | 160 + 2 shared | 6 | MLA | aux loss + 细粒度+共享 |
| **DeepSeek-V3** (2024-12) | **671B** | **37B** | 256 + 1 shared | 8 | MLA | **Aux-Loss-Free + grouped TopK** ⭐ |
| **Qwen3-MoE** (Alibaba 2025) | 235B | 22B | 128 | 8 | GQA | aux loss + 细粒度 |
| **Llama-4 Scout** (Meta 2025-04) | 109B | 17B | 16 | 1 | GQA | top-1 + aux loss |
| **Llama-4 Maverick** (Meta 2025-04) | 400B | 17B | 128 + 1 shared | 1 | GQA | top-1 + shared expert |
| **Kimi K2** (Moonshot 2025) | ~1T | 32B | 细粒度 + shared | top-K | MLA | MoBA + aux-loss-free 风格 |

### 4.1 趋势观察

1. **细粒度 + 共享** 路线（DeepSeek 风格）正在被广泛复制
2. **Aux-loss-free** 是 2024-2025 最大算法创新点
3. **MLA + MoE** 组合（KV + FFN 都稀疏）成为长 context MoE 的标准
4. 总参 vs 激活参的比例普遍是 **10-30×**（更高的比例意味着更"参数效率"）
5. Llama-4 用 top-1 是 Switch Transformer 复辟（简化通信）

### 4.2 一个微妙的事实

> 同一年代里，**dense 70B（Llama-3-70B）** 和 **MoE 200B/30B（DeepSeek-V2）** 在多数 benchmark 上互有胜负。
> → MoE 不是"必赢"，它是"另一条 scaling 曲线"。
> → 选 MoE 还是 dense 取决于 **你优化的是什么**（训练成本 vs 推理成本 vs 部署复杂度）。

### 4.3 不同路线的"灵魂"

| 路线 | 代表 | 灵魂 | 适合谁 |
|---|---|---|---|
| **Switch 极简** | Switch / Llama-4 | top-1 + 大 N，最低通信 | 通信受限场景 |
| **GShard 经典** | GShard / Mixtral | top-2 + aux loss，平衡可控 | 工程稳健团队 |
| **DeepSeek 细粒度** | DSv3 / Qwen3 / Kimi K2 | 256 expert + sigmoid + 共享 + aux-loss-free | 追求质量上限 |
| **Layer-wise 混合** | 部分 Llama-4 | 部分 layer dense / 部分 MoE | 调和稳定性与稀疏性 |

---

## 五、MoE 的隐藏成本

不在论文里、但实战里要付的代价：

### 5.1 训练

- **超参对 MoE 敏感**：lr、aux loss weight、capacity 都要调
- **训练 collapse 风险**：dead expert / capacity overflow，前几千 step 高发
- **检查点大**：671B 模型 checkpoint ~ 1.3TB，存储、传输都贵
- **训练监控复杂**：要看 expert utilization、router entropy、token drop ratio 等

### 5.2 部署

- **推理工具链相对新**：vLLM/SGLang 2024 才完善 MoE 支持
- **多机部署门槛高**：单机装不下 → 必须跨机 → InfiniBand / RoCE 必须
- **量化策略复杂**：expert 间数值分布差异大，朴素 INT8 容易掉点
- **batch 调度复杂**：MoE 的 latency 对 batch 大小敏感

### 5.3 评测

- **MMLU / 通用 benchmark 表现不显**：MoE 的优势在长尾任务（多语种、长 context、专业领域）
- **NIAH / RULER** 等长 context 评测受 attention 而不是 MoE 影响

→ **盲目用 MoE 可能"质量没涨但开发成本翻倍"**。

---

## 六、MoE 特有的微调问题

### 6.1 LoRA on MoE 的难点

经典 LoRA：给 $W \in \mathbb{R}^{d \times d'}$ 加上 $\Delta W = BA$，$B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d'}$。

MoE 上的 LoRA 有几个难题：

**(1) 加在哪？**

- 加在所有 expert 上：每 expert 一组 LoRA → N 组 LoRA，参数量 N×
- 只加在 router 上：影响路由但不改 expert，能力有限
- 只加在 shared expert 上：只影响通用能力
- 加在共享层（attention）上：和 dense LoRA 一样

**(2) 路由是否需要冻结？**

- 冻结 router：保留预训练 routing 决策，只调 expert 内部
- 训练 router：允许任务相关的 routing 调整，但有 collapse 风险
- 折中：router 用极小学习率（如主 lr 的 0.1×）

**(3) 训练数据稀疏触发 expert**

- 微调数据小（如 1k samples），可能不会激活所有 expert
- 未激活的 expert 的 LoRA 参数不更新 → 浪费
- 解决：只对被激活的 expert 加 LoRA（动态）

### 6.2 现状

- [Mixtral LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/moe)：默认对所有 expert 加 LoRA + 冻结 router
- DSv3 LoRA：尚无成熟方案（社区在探索）
- 实际生产经验：MoE 微调比 dense 微调难度大 2-3 倍

### 6.3 全参数微调 vs LoRA

- MoE 全参数微调：671B 的 V3 ≈ 1.3TB 优化器状态，要求极大算力
- DSv3 R1 系列：用 GRPO 全参数 RL，需要 ~2000 H100
- LoRA on MoE：可行但收益不确定

---

## 七、Sparse Upcycling（dense → MoE）

### 7.1 思路

[Komatsuzaki et al. 2022 — Sparse Upcycling](https://arxiv.org/abs/2212.05055)：

> 从训练好的 dense 模型出发，把单个 FFN **复制 N 份**作为初始 N 个 expert，加上 router 继续训练。

数学：

$$\text{Expert}_i^{(0)} = \text{FFN}^{\text{dense}}, \quad \forall i = 1, \dots, N$$

router 随机初始化，然后继续训练。

### 7.2 收益

- 不用从头训 MoE，省 70%+ 训练算力
- 起点质量 = dense FFN，比 random init 高
- 在小算力 fine-tuning 后能达到接近"从头训 MoE"的质量

### 7.3 限制

- 收益主要在 ~30% dense 训练 token 量级（充分微调后）
- 长期效果不如从头训 MoE（差距 ~1-3% PPL）
- 只能用于已训好 dense 的情况

### 7.4 实际案例

- Mistral 早期实验，证明可行性
- Google 部分 PaLM-MoE 系列用此方法
- DSv3 没用 upcycling（从头训）

### 7.5 反方向：MoE → Dense 蒸馏

- Mixtral → Mixtral-Mini：用蒸馏把 MoE 模型压成小 dense
- DeepSeek-V3 → DeepSeek-V3-Distill：蒸到 dense 7B / 14B / 32B / 70B
- 蒸馏后的 dense 模型保留了大部分能力，部署更简单

→ 大模型 MoE 训练 + 小模型 dense 蒸馏，是 2024-2025 的标准 pipeline。

---

## 八、长 context 与 MoE 的协同

### 8.1 正交性

- 长 context = attention 优化（KV cache, YaRN, Sparse Attention，§6 §8）
- MoE = FFN 优化（§9）
- 两者**正交**，可以叠加：DSv3 = MLA (压 KV) + MoE (稀疏 FFN) + 128k context

### 8.2 MoE 在长 context 上的优势

- Routing 可以**位置感知**：长 sequence 内不同位置的 token 可能去不同 expert
- Sequence-level aux loss（§9.2）让长 sequence 的 expert 调用分散，便于推理 batching

### 8.3 MoE 在长 context 上的挑战

- 长 sequence → 单 token 算力小但 routing 决策多
- 推理时 KV cache + expert weights 都要装显存 → 显存压力大
- DSv3 用 MLA 压 KV + INT8 expert + node-limited routing 共同应对

---

## 九、当下（2026 视角）的选型决策树

```
你要训一个新模型？
  │
  ├─ 训练算力 < 10²² FLOPs？        ─→ Dense (3B-13B, GQA + RoPE)
  │
  ├─ 推理目标是单卡消费级？          ─→ Dense (≤14B) + INT8
  │
  ├─ 总参数预算 > 100B + 多机集群？  ─→ MoE
  │     ├─ 长 context 优先 ─→ MLA + 细粒度 MoE + Aux-Loss-Free
  │     │                          (DeepSeek-V3 路线)
  │     ├─ 推理 latency 优先 ─→ top-1 MoE (Switch / Llama-4 路线)
  │     ├─ 多语种 / 多任务 ─→ 细粒度 + shared (DeepSeek-MoE 路线)
  │     └─ 长 reasoning（CoT）─→ 看团队偏好，dense 70B 仍是稳健选择
  │
  └─ 中间地带（10-50B 训练）？      ─→ 看团队工程能力
        ├─ 团队熟：MoE 可尝试（Mixtral 风格起步）
        └─ 团队陌：稳妥 Dense
```

---

## 十、关键问答

**Q1**：MoE 是不是"参数量虚标"？

- 不是。MoE 的总参数都被训练过、都有梯度更新
- 但**单 token 不会用到所有参数** → "等效 dense 大小"约为总参的 1/3 ~ 1/5
- 媒体说 "671B 模型" 可能误导，更准确是 "671B 总 / 37B 激活"
- Scaling law（§2.5）给出更精确的等效计算

**Q2**：MoE 在 reasoning（CoT）上是否劣于 dense？

- 学术上有争议
- 经验：DeepSeek-V3 / R1 推理任务（math、code）非常强，似乎 MoE 不影响 reasoning
- 但 OpenAI 的 o1 / o3 是否 MoE 不公开
- → "MoE 不能 reasoning" 没有强证据，但 dense 在 small scale 上 reasoning 更稳

**Q3**：能不能"训 dense + 部署 MoE"？

- 反过来可以：训 MoE + 蒸到 dense（DeepSeek-V3 → V3-Distill 7B/14B/32B/70B）
- dense → MoE 的 Sparse Upcycling（§7）也有研究，但收益有限
- 实际 pipeline：训大 MoE + 蒸小 dense

**Q4**：MoE 显存为啥比同激活 dense 大？

- 推理时所有 expert weight 都要在 GPU 内存中可访问（router 决定才能选）
- 即使 token 不去某 expert，它的参数也得在 GPU
- → MoE 节省**计算**，不节省**显存**
- 这是 MoE 部署的最大物理约束

**Q5**：MoE 适合小模型吗？

- 一般不适合 < 7B
- 总参太少 → 每个 expert 太小，路由分辨率不够
- 工程复杂度不划算
- 例外：DeepSeek-MoE 16B 证明 MoE 在 16B 也能有收益（但需要细粒度 + shared 这套）

**Q6**：DeepSeek-V3 为什么没出 dense 版？

- 同算力下 dense 等效 ~ 100-200B，已经被 Llama-3-70B / Qwen2-72B 占着
- MoE 给了 DeepSeek 一个"差异化技术路线" → 671B 总参在 inference 上仍可控
- 训练成本 ~ 6M USD（DeepSeek 报告），dense 70B 训练相当
- 但 V3 → V3-Distill 系列出了 dense 蒸馏版

**Q7**：未来 2 年（2026-2028）MoE 会取代 dense 吗？

- **大模型（70B+）**：MoE 已基本取代 dense
- **中模型（7-30B）**：dense + MoE 并存（蒸馏 MoE 也是 dense 形态）
- **小模型（< 7B）**：dense 主导
- 长期看 MoE 是"参数 scaling 的捷径"，但需要更多通信优化

**Q8**：MoE 的 LoRA 微调好不好做？

- 难。要决定 LoRA 加在哪（所有 expert / shared expert / router）
- Router 是否参与训练有 trade-off：参与可能 collapse，不参与限制适应性
- 实际经验：MoE 微调难度比 dense 高 2-3 倍

**Q9**：Sparse Upcycling 现在还用吗？

- 不主流。主要原因：质量比从头训 MoE 略低
- 适合场景：已经有 dense 模型，想低成本探索 MoE 收益
- DSv3 / Qwen3-MoE 都从头训

**Q10**：MoE 推理 latency 比 dense 高多少？

- 同激活参数下：MoE decode latency ≈ 1.5-3× dense
- 主要来自 All-to-All 通信
- 用 replicated expert / node-limited routing / DualPipe 可减少差距
- 单机 + 全 expert 复制时可以接近 dense

---

## 十一、本章总结（§9 全章）

```
§9.1 路由         ── 谁去激活谁
                     - 从 dense FFN 到 sparse MoE 数学推导
                     - top-K 离散选择的梯度通路
                     - softmax vs sigmoid 评分
                     - 四种路由（Switch/GShard/Mixtral/DSv3）完整公式

§9.2 负载均衡     ── 让 router 不偏
                     - 三种不均衡量化指标
                     - aux loss 公式 + 梯度推导
                     - z-loss 数值稳定
                     - aux-loss-free 完整算法 + sequence-level loss

§9.3 架构变体     ── 改 expert 结构
                     - Granularity Scaling Law (Krajewski 2024)
                     - shared expert 数学结构
                     - Dropless block-sparse GEMM 算法

§9.4 EP / 通信    ── 把 expert 切到多卡
                     - 4 次 All-to-All 通信量推导
                     - Node-limited / Device-limited Routing
                     - DualPipe 双流水线
                     - FP8 通信
                     - 5D 并行嵌套

§9.5 选型（本节） ── 用不用 MoE，用什么 MoE
                     - MoE Scaling Law (Clark 2022, Krajewski 2024)
                     - 等效 dense 大小估算
                     - LoRA on MoE 难点
                     - Sparse Upcycling
                     - 决策树
```

**一句话总结整章**：

> MoE = "FFN 这一侧的稀疏化"，是 LLM scaling 的现代手段。
> 不是免费午餐，是用通信复杂度换参数容量。
> 工程上由 router + load balance + EP + 推理调度构成；
> 2024-2026 主流路径 = **DeepSeek 风格**（细粒度 + 共享 + Aux-Loss-Free + MLA + node-limited routing）。

---

## 十二、与其他章关系

```
§5 attention 变体（GQA/MLA） ── MoE 与 attention 优化正交，可叠加
§6 sparse attention          ── 这是"attention 侧的稀疏"，MoE 是"FFN 侧的稀疏"
§7 SSM / 混合架构            ── MoE 与 SSM 都是降"等效算力"的手段
§8 长上下文                  ── MoE 与长 context 正交，可叠加（DSv3 + 128k）
§10 并行化                   ── EP 是 5D 并行的一维
§11 后训练                   ── MoE 模型 SFT / RLHF 有额外稳定性问题（router 微调风险）
§12 推理优化                 ── MoE serving 仍是 active 研究领域
§13 评测                     ── MoE 的优势在长尾，benchmark 选择影响结论
```

---

## 参考资料

**Scaling Law（理论基础）**：
- [Hoffmann et al. 2022 — Chinchilla](https://arxiv.org/abs/2203.15556) ⭐（dense scaling law）
- [Clark et al. 2022 — Unified scaling laws for routed LM](https://arxiv.org/abs/2202.01169) ⭐⭐（MoE scaling law）
- [Krajewski et al. 2024 — Fine-grained MoE scaling laws](https://arxiv.org/abs/2402.07871) ⭐⭐（granularity scaling）
- [Du et al. 2022 — GLaM](https://arxiv.org/abs/2112.06905)

**现代 MoE 大模型**：
- [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) ⭐
- [DeepSeek-MoE](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) ⭐⭐
- [DBRX 技术博客](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- [Llama-4 (Meta 2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Qwen3-MoE (Alibaba 2025)](https://qwenlm.github.io/blog/qwen3/)
- [Kimi K2 (Moonshot 2025)](https://arxiv.org/abs/2501.12599)

**Upcycling 与 PEFT**：
- [Komatsuzaki et al. 2022 — Sparse Upcycling](https://arxiv.org/abs/2212.05055)（dense → MoE）
- [PEFT MoE docs](https://huggingface.co/docs/peft/main/en/conceptual_guides/moe)
- [MoE-LoRA 综述](https://arxiv.org/abs/2405.00732)

**综述**：
- [Sebastian Raschka — Inside DeepSeek-V3](https://magazine.sebastianraschka.com/p/the-state-of-llms-in-2024) ⭐
- [HuggingFace blog — MoE Explained](https://huggingface.co/blog/moe) ⭐
- [Hugging Face — How to fine-tune MoE](https://huggingface.co/blog/moe)
