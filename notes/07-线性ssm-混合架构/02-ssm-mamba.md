# 7.2 SSM（State Space Models / Mamba / Mamba-2）

[← 返回框架](../../README.md) · [📎 materials.md → §7.2](../../materials.md)

---

## 〇、SSM 是什么、和 Attention 什么关系

State Space Model：来自**控制论 / 信号处理**的经典模型，把序列建模成一个连续时间动力系统：

$$h'(t) = A h(t) + B x(t), \qquad y(t) = C h(t)$$

- $h(t)$：隐藏状态（state，维度 $N$）
- $x(t)$：输入信号（标量或向量）
- $y(t)$：输出
- $A, B, C$：状态转移 / 输入 / 输出矩阵

离散化（用步长 $\Delta$）后变成 RNN：
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t$$

→ **形式上是 RNN**，但通过精心设计 $A$ 可以**长程稳定 + 训练并行**。

```
和 §7.1 线性 attention 的关系：
  线性 attention: state_t = state_{t-1} + K_t^T V_t              ← state = d×d 矩阵
  SSM:            h_t     = Ā h_{t-1}  + B̄ x_t                    ← state = N 维向量
  → 都是 "RNN 形式 + 训练时可 parallel"，只是 state 形态不同
```

---

## 一、为什么 SSM 突然成为主角

2021-2022 期间一系列工作把 SSM 从"信号处理玩具"做成了"能跟 Transformer 比"的 NLP 骨架：

| 工作 | 关键贡献 |
|------|---------|
| **HiPPO** (Gu 2020) | 用正交多项式初始化 $A$，能记住整段历史 |
| **S4** (Gu 2021) | 结构化 $A$ 矩阵 → 训练 parallel 可行（FFT-based） |
| **S5 / Liquid-S4 / DSS** (2022) | 进一步简化 + diagonal $A$ |
| **H3 / Hyena** (2023) | 把 SSM 嵌入 transformer block，开始挑战语言任务 |
| **Mamba** (Gu & Dao 2023) | 引入 "**selective state**"，质量首次匹配同规模 Transformer |
| **Mamba-2** (Dao & Gu 2024) | SSM ↔ attention 等价 ("SSD"), 加速训练 |

→ Mamba 是分水岭：第一次有人能说"SSM 在语言任务上不输 Transformer"。

---

## 二、Mamba 核心：Selective SSM

### 2.1 传统 SSM 的弱点

经典 S4 的 $A, B, C, \Delta$ **不依赖输入** → 等价于一个 LTI（线性时不变）系统 → 能记得"什么时候"发生过事情，但不能根据内容决定"要不要记"。

→ NLP 中需要 "**信息相关的过滤**"（"the" 不重要、人名重要），LTI 做不到。

### 2.2 Mamba 的 Selectivity

把 $B, C, \Delta$ 改成**输入相关**：
$$B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))$$

$A$ 仍保持时不变（diagonal、负实部，稳定）。

```
selective SSM:
   h_t = exp(-Δ_t · |A|) · h_{t-1} + Δ_t · B_t · x_t       ← Δ_t, B_t 依赖输入
   y_t = C_t · h_t                                          ← C_t 也依赖输入
```

→ 当 $\Delta_t$ 大：state 大幅更新（强写入）；$\Delta_t$ 小：state 几乎冻结（忽略当前输入）。
→ 像 LSTM/GRU 的 gate，但放在 SSM 框架里。

### 2.3 训练效率：parallel scan

selective SSM 的 $\bar{A}_t$ 每步不同 → 不能 FFT。Mamba 用 **parallel prefix scan** （Blelloch-style）把 $O(L)$ recurrent 改成 $O(\log L)$ parallel depth。

实现层面：[Gu & Dao 2023 — Mamba paper](https://arxiv.org/abs/2312.00752) 提供了"硬件感知"的 selective scan kernel：
- state 留在 SRAM
- 不把 $\bar{A}_t, \bar{B}_t$ 物化到 HBM
- 类似 FlashAttention 的 tiling 思路

→ 训练吞吐与 Transformer 接近（同规模约 0.8-1.2×）。

### 2.4 Mamba block 结构

```
input ─→ Linear → SiLU ─┐
                        ↘
                     Selective SSM (h is the state) ─→ output
                        ↗
input ─→ Linear ────────┘
                  (gating branch)
```

- Mamba 块没有 attention、也没有 FFN，整个块就一个 selective SSM
- 一个典型 Mamba 模型 = 一堆 Mamba block 堆叠

---

## 三、Mamba-2 与 SSD

[Mamba-2 (Dao & Gu 2024)](https://arxiv.org/abs/2405.21060) 提出 **State Space Duality (SSD)**：

> selective SSM ≡ 一种 **结构化 mask + linear attention**

把 SSM 的 state 转移矩阵 $A$ 展开成一个"半可分（semi-separable）矩阵 $M$"，attention 就是 $Y = M (Q K^\top) V$（其中 $M$ 编码了 SSM 的衰减）。

收益：
- 训练时直接用 attention/matmul 硬件路径（不用 selective scan）
- state 维度可以更大（Mamba-1 受 SRAM 限制，state ≤ 16；Mamba-2 ≥ 64）
- 训练吞吐进一步提升

→ Mamba-2 = "SSM 写法的训练效率" + "Linear attention 的硬件友好"。

---

## 四、Mamba 的实证质量

[Mamba paper, 2023-12](https://arxiv.org/abs/2312.00752) 在 The Pile、ARC 等数据集上的结论：

| 规模 | Mamba vs Transformer (同算力训练) |
|------|----------------------------------|
| 130M | ≈ |
| 1.4B | ≈ |
| 2.8B | ≈ 或略胜 |

但**关键弱点**（后续 paper 反复确认）：

1. **In-context retrieval / NIAH 弱**：state 是固定大小压缩 → "针在草堆里"任务掉点
2. **复杂 reasoning 偏弱**：缺乏 attention 的 sharp routing 能力
3. **scale 到 70B+ 时未充分验证**：截至 2025-05，公开的纯 Mamba 模型最大约 7B

→ 业界共识：纯 Mamba 在小-中规模上能打，但**长 context retrieval 是它的天花板**。这直接催生了 §7.3 的"混合架构"。

---

## 五、Mamba 推理时的成本

```
Mamba 的 state per token = 一个固定大小的张量 (channels × state_dim)
  Mamba-1: state_dim N=16 → state ≈ d × 16
  Mamba-2: state_dim N=64-128 → state ≈ d × 128
```

- decode 每步只读这个小 state，**不读历史 token**
- 1M context 推理：state 大小与 1k context 完全相同
- vs Transformer：MHA 在 1M context 下 KV 几百 GB
- → 这是 Mamba 在长 context 推理上的杀手锏

但参数量上：Mamba 没有 attention 但需要更宽的 channel → 同质量下参数量可能略多。

---

## 六、关键问答

**Q1**：Mamba 是 RNN 吗？
- 数学上是 RNN（state 递推）
- 但训练时用 parallel scan → 不像 LSTM 必须 sequential
- 推理时确实是 sequential（每步更新 state）
- 称之为"**parallel-trainable RNN**"

**Q2**：Mamba state 那么小，是不是会丢信息？
- 是。state 是对历史的"有损压缩"
- 但 Mamba 用 selectivity 学到"什么该留、什么该丢"
- 短序列任务影响小，长 context retrieval 影响大

**Q3**：Mamba 相比 RWKV/RetNet 强在哪？
- **selectivity**：data-dependent gate（RWKV-4 的衰减是 fixed，RetNet 的 $\gamma$ 也固定）
- **硬件感知 kernel**：selective scan + SRAM tile，训练吞吐高
- Mamba-2 进一步把 state 维度拉大，质量逼近 Transformer

**Q4**：Mamba 能做 retrieval（NIAH）吗？
- 短到中长度（≤ 4k）：可以
- 长（≥ 32k）：纯 Mamba 显著掉点
- 修复方案：加少量 attention 层（见 §7.3 Jamba、Nemotron-H、Zamba）

**Q5**：Mamba 和 linear attention 的本质区别？
- 两者都是 "RNN 形式 + parallel 训练 + 固定 state"
- 区别在于 **state 的结构**：
  - Linear attention：state = $\phi(K) V^\top$（矩阵）
  - Mamba：state = $h$（一组 channel-wise 状态向量，配 diagonal $A$）
- Mamba-2 已经证明两者**等价于一类半可分矩阵 attention**

**Q6**：Mamba 训练时 state 怎么保存梯度？
- 不显式保存（吃不下内存）
- 用 selective scan kernel 在 backward 时重算（类似 FA 的 recompute）

**Q7**：为什么不是 OpenAI/Anthropic 在推 Mamba？
- 大厂数据上 Transformer + MoE 已经能拉到 trillion 级别，retrieval 等弱项不暴露
- Mamba 的优势在"长 context 推理省成本"，对学术/中小模型吸引力更大
- 大厂的态度：观望 + 在混合架构上小规模实验（如 Nemotron-H、IBM Bamba）

---

## 七、SSM/Mamba 家族图谱

```
HiPPO (Gu 2020)
  ↓
S4 (Gu 2021)  ───── DSS / S5 (diagonal A)
  ↓
H3 / Hyena (2023)  ──→ 把 SSM 装进 Transformer
  ↓
Mamba (Gu & Dao 2023, Selective SSM)
  ↓
Mamba-2 (Dao & Gu 2024, SSD)  ──→ 等价于一种 linear attention
  ↓
Jamba / Zamba / Nemotron-H / Bamba ── 混合架构（见 §7.3）
```

---

## 参考资料

- [Gu et al. 2020 — HiPPO](https://arxiv.org/abs/2008.07669)
- [Gu et al. 2021 — Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) ⭐
- [Gu & Dao 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) ⭐⭐
- [Dao & Gu 2024 — Transformers are SSMs (Mamba-2 / SSD)](https://arxiv.org/abs/2405.21060) ⭐
- [Poli et al. 2023 — Hyena Hierarchy](https://arxiv.org/abs/2302.10866)
- [Fu et al. 2023 — H3: Hungry Hungry Hippos](https://arxiv.org/abs/2212.14052)
- [Sasha Rush — Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html) ⭐（推导极清楚）
- [Albert Gu — The Annotated S4 (blog)](https://srush.github.io/annotated-s4/)
- [Mamba GitHub](https://github.com/state-spaces/mamba)
- [Tri Dao — SSD blog](https://tridao.me/blog/2024/mamba2-part1-model/)
