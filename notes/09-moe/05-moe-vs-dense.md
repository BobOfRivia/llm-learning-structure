# 9.5 MoE vs Dense 选型与实例

[← 返回框架](../../README.md) · [📎 materials.md → §9.5](../../materials.md)

---

## 〇、本节回答什么

§9.1-9.4 已经讲完 MoE 怎么做、怎么训、怎么部署。本节是 **决策**：

> 给定一个具体目标（质量、成本、context 长度、推理硬件），到底**用不用 MoE**？用什么规模？

---

## 一、MoE vs Dense 的本质差异

```
                  Dense           MoE
                  ─────           ───
总参数量          P               P_total
单 token 算力     ≈ P             ≈ P_active (= P_total × K/N)
单 token 显存读   P (KV+W)        P_total (W) + KV  ← 推理时全 expert weight 都要驻留
通信开销          全梯度同步      额外 All-to-All
质量上限          高              更高（同算力）
推理延迟（decode） 低              高（除非 expert replicate）
```

### 1.1 用一个数字总结

> **MoE 的核心交易**：用 N× 显存 + 额外通信，换 ~3-7× 等效参数容量。

→ 显存充裕 + 推理 batch 大 → MoE 划算
→ 显存紧张 + 推理 batch 小 → dense 划算

### 1.2 Scaling Law 视角

[Hoffmann et al. 2022 — Chinchilla](https://arxiv.org/abs/2203.15556) 显示：dense 模型质量随 FLOPs 增长有明确斜率。

[Clark et al. 2022 / Du et al. 2022 — MoE scaling](https://arxiv.org/abs/2202.01169) 显示：MoE 也有 scaling law，**斜率更陡**（同 FLOPs 涨更多）但需要更多数据 token 才能"激活"参数。

经验：
- 1B-10B 规模：MoE 优势小（dense 简单胜）
- 70B+ 规模：MoE 收益显著（DeepSeek-V3 671B 总 / 37B 激活 ≈ 250B dense）
- < 1B：MoE 几乎没用，不如直接小 dense

---

## 二、按场景选

### 2.1 训练规模决定

| 总训练算力 | 推荐 |
|----------|------|
| < $10^{22}$ FLOPs（~7B dense） | **Dense** 优先（MoE 收益小，复杂度大） |
| $10^{22} \sim 10^{23}$ FLOPs（~70B） | **MoE 起步**（DeepSeek 路线），dense 仍可竞争 |
| > $10^{23}$ FLOPs | **MoE 优先**，几乎所有 SOTA 都 MoE |

### 2.2 推理硬件决定

| 推理硬件 | 推荐 |
|--------|------|
| 单卡消费级（4090, 24GB） | **Dense ≤ 14B**，MoE 装不下 |
| 单机 8 卡 H100 | Dense 70B 或 **MoE ≤ 200B**（INT4 量化） |
| 多机 H100 集群 | MoE 任意，**DeepSeek-V3 671B 标准配置** |
| CPU 推理 | Dense 优先（MoE 通信开销在 CPU 上更显著） |
| 端侧 / 移动 | **Dense 小模型**（< 3B） |

### 2.3 任务类型决定

| 任务 | Dense 表现 | MoE 表现 |
|------|----------|---------|
| 通用对话 | 良 | 优（容量大，知识广） |
| 数学 / 代码 | 良 | 优（细粒度专家可专门化） |
| 长 context retrieval | 良 | 良（取决于架构，与 MoE 正交） |
| 多语言 | 良 | **优**（不同 expert 可专攻语种） |
| 强 reasoning（CoT） | 优 | 良（routing 离散会破坏 reasoning 连贯性？争议中） |
| 极低延迟 serving | **优** | 略弱（All-to-All） |

---

## 三、2024-2026 主流模型横评

| 模型 | 总参 | 激活参 | Expert 数 | top-K | Attention | 关键路由策略 |
|------|------|-------|----------|-------|-----------|------------|
| **Mixtral 8x7B** (2023-12) | 47B | 13B | 8 | 2 | GQA | aux loss |
| **Mixtral 8x22B** (2024-04) | 141B | 39B | 8 | 2 | GQA | aux loss |
| **DBRX** (Databricks 2024) | 132B | 36B | 16 | 4 | GQA | aux loss + z-loss |
| **DeepSeek-MoE 16B** (2024-01) | 16B | 2.8B | 64 + 2 shared | 6 | MHA | aux loss + 细粒度+共享 |
| **DeepSeek-V2** (2024-05) | 236B | 21B | 160 + 2 shared | 6 | MLA | aux loss + 细粒度+共享 |
| **DeepSeek-V3** (2024-12) | **671B** | **37B** | 256 + 1 shared | 8 | MLA | **Aux-Loss-Free** ⭐ |
| **Qwen3-MoE** (Alibaba 2025) | 235B | 22B | 128 | 8 | GQA | aux loss + 细粒度 |
| **Llama-4 Scout** (Meta 2025-04) | 109B | 17B | 16 | 1 | GQA | top-1 + aux loss |
| **Llama-4 Maverick** (Meta 2025-04) | 400B | 17B | 128 + 1 shared | 1 | GQA | top-1 + shared expert |
| **Kimi K2** (Moonshot 2025) | ~1T | 32B | 细粒度 + shared | top-K | MLA | MoBA + aux-loss-free 风格 |

### 3.1 趋势观察

1. **细粒度 + 共享** 路线（DeepSeek 风格）正在被广泛复制
2. **Aux-loss-free** 是 2024-2025 最大算法创新点
3. **MLA + MoE** 组合（KV + FFN 都稀疏）成为长 context MoE 的标准
4. 总参 vs 激活参的比例普遍是 **10-30×**（更高的比例意味着更"参数效率"）
5. Llama-4 用 top-1 是 Switch Transformer 复辟（简化通信）

### 3.2 一个微妙的事实

> 同一年代里，**dense 70B（Llama-3-70B）** 和 **MoE 200B/30B（DeepSeek-V2）** 在多数 benchmark 上互有胜负。
> → MoE 不是"必赢"，它是"另一条 scaling 曲线"。
> → 选 MoE 还是 dense 取决于 **你优化的是什么**（训练成本 vs 推理成本 vs 部署复杂度）。

---

## 四、MoE 的隐藏成本

不在论文里、但实战里要付的代价：

### 4.1 训练

- **超参对 MoE 敏感**：lr、aux loss weight、capacity 都要调
- **训练 collapse 风险**：dead expert / capacity overflow，前几千 step 高发
- **检查点大**：671B 模型 checkpoint ~ 1.3TB，存储、传输都贵

### 4.2 部署

- **推理工具链相对新**：vLLM/SGLang 2024 才完善 MoE 支持
- **多机部署门槛高**：单机装不下 → 必须跨机 → InfiniBand / RoCE 必须
- **量化策略复杂**：expert 间数值分布差异大，朴素 INT8 容易掉点

### 4.3 评测

- **MMLU / 通用 benchmark 表现不显**：MoE 的优势在长尾任务（多语种、长 context、专业领域）
- **NIAH / RULER** 等长 context 评测受 attention 而不是 MoE 影响

→ **盲目用 MoE 可能"质量没涨但开发成本翻倍"**。

---

## 五、当下（2026 视角）的选型决策树

```
你要训一个新模型？
  ├─ 训练算力 < 10²² FLOPs？        ─→ Dense (3B-13B, GQA + RoPE)
  │
  ├─ 推理目标是单卡消费级？          ─→ Dense (≤14B) + INT8
  │
  ├─ 总参数预算 > 100B + 多机集群？  ─→ MoE
  │     ├─ 长 context 优先 ─→ MLA + 细粒度 MoE + Aux-Loss-Free
  │     │                          (DeepSeek-V3 路线)
  │     ├─ 推理 latency 优先 ─→ top-1 MoE (Switch / Llama-4 路线)
  │     └─ 多语种 / 多任务 ─→ 细粒度 + shared (DeepSeek-MoE 路线)
  │
  └─ 中间地带（10-50B 训练）？      ─→ 看团队工程能力
        ├─ 团队熟：MoE 可尝试
        └─ 团队陌：稳妥 Dense
```

---

## 六、关键问答

**Q1**：MoE 是不是"参数量虚标"？
- 不是。MoE 的总参数都被训练过、都有梯度更新
- 但**单 token 不会用到所有参数** → "等效 dense 大小"约为总参的 1/3 ~ 1/5
- 媒体说 "671B 模型" 可能误导，更准确是 "671B 总 / 37B 激活"

**Q2**：MoE 在 reasoning（CoT）上是否劣于 dense？
- 学术上有争议
- 经验：DeepSeek-V3 / R1 推理任务（math、code）非常强，似乎 MoE 不影响 reasoning
- 但 OpenAI 的 o1 / o3 是否 MoE 不公开
- → "MoE 不能 reasoning" 没有强证据，但 dense 在 small scale 上 reasoning 更稳

**Q3**：能不能"训 dense + 部署 MoE"？
- 反过来可以：训 MoE + 蒸到 dense（Mixtral → Mixtral-Mini）
- dense → MoE 的"sparsification" 也有研究（[Sparse Upcycling](https://arxiv.org/abs/2212.05055)），但收益有限

**Q4**：MoE 显存为啥比同激活 dense 大？
- 推理时所有 expert weight 都要在 GPU 内存中可访问（router 决定才能选）
- 即使 token 不去某 expert，它的参数也得在 GPU
- → MoE 节省**计算**，不节省**显存**

**Q5**：MoE 适合小模型吗？
- 一般不适合 < 7B
- 总参太少 → 每个 expert 太小，路由分辨率不够
- 工程复杂度不划算
- 例外：DeepSeek-MoE 16B 证明 MoE 在 16B 也能有收益（但需要细粒度 + shared 这套）

**Q6**：DeepSeek-V3 为什么没出 dense 版？
- 同算力下 dense 等效 ~ 100-200B，已经被 Llama-3-70B / Qwen2-72B 占着
- MoE 给了 DeepSeek 一个"差异化技术路线" → 671B 总参在 inference 上仍可控
- 训练成本 ~ 6M USD（DeepSeek 报告），dense 70B 训练相当

**Q7**：未来 2 年（2026-2028）MoE 会取代 dense 吗？
- **大模型（70B+）**：MoE 已基本取代 dense
- **中模型（7-30B）**：dense + MoE 并存
- **小模型（< 7B）**：dense 主导
- 长期看 MoE 是"参数 scaling 的捷径"，但需要更多通信优化

---

## 七、本章总结（§9 全章）

```
§9.1 路由         ── 谁去激活谁
§9.2 负载均衡     ── 让 router 不偏
§9.3 架构变体     ── 细粒度 + 共享 + dropless
§9.4 EP / 通信    ── 把 expert 切到多卡
§9.5 选型（本节） ── 用不用 MoE，用什么 MoE
```

**一句话总结整章**：

> MoE = "FFN 这一侧的稀疏化"，是 LLM scaling 的现代手段。
> 不是免费午餐，是用通信复杂度换参数容量。
> 工程上由 router + load balance + EP + 推理调度构成；
> 2024-2026 主流路径 = **DeepSeek 风格**（细粒度 + 共享 + Aux-Loss-Free + MLA）。

---

## 八、与其他章关系

```
§5 attention 变体（GQA/MLA） ── MoE 与 attention 优化正交，可叠加
§6 sparse attention          ── 这是"attention 侧的稀疏"，MoE 是"FFN 侧的稀疏"
§7 SSM / 混合架构            ── MoE 与 SSM 都是降"等效算力"的手段
§8 长上下文                  ── MoE 与长 context 正交，可叠加（DeepSeek-V3 + 128k）
§10 并行化                   ── EP 是 5D 并行的一维
§11 后训练                   ── MoE 模型 SFT / RLHF 有额外稳定性问题（router 微调风险）
§12 推理优化                 ── MoE serving 仍是 active 研究领域
§13 评测                     ── MoE 的优势在长尾，benchmark 选择影响结论
```

---

## 参考资料

- [Hoffmann et al. 2022 — Chinchilla](https://arxiv.org/abs/2203.15556) ⭐（scaling law）
- [Clark et al. 2022 — Unified scaling laws for routed LM](https://arxiv.org/abs/2202.01169) ⭐
- [Du et al. 2022 — GLaM](https://arxiv.org/abs/2112.06905)
- [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) ⭐
- [DeepSeek-MoE](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) ⭐⭐
- [DBRX 技术博客](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- [Llama-4 (Meta 2025)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Qwen3-MoE (Alibaba 2025)](https://qwenlm.github.io/blog/qwen3/)
- [Kimi K2 (Moonshot 2025)](https://arxiv.org/abs/2501.12599)
- [Komatsuzaki et al. 2022 — Sparse Upcycling](https://arxiv.org/abs/2212.05055)（dense → MoE 蒸馏）
- [Sebastian Raschka — Inside DeepSeek-V3](https://magazine.sebastianraschka.com/p/the-state-of-llms-in-2024) ⭐
- [HuggingFace blog — MoE Explained](https://huggingface.co/blog/moe) ⭐
