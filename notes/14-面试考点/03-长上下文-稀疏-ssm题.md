# 14.3 长上下文 / 稀疏 / SSM 高频题（§6-§8）

[← 返回框架](../../README.md) · [📎 materials.md → §14.3](../../materials.md)

---

## 〇、本节回答什么

> 2025-2026 增量重点：长上下文外推、稀疏注意力（NSA/DSA/MoBA）、Mamba/SSM/线性注意力、混合架构（Jamba/MiniMax/Nemotron-H）。
>
> 覆盖 §6 稀疏 / 压缩注意力、§7 线性注意力 / SSM / 混合架构、§8 长上下文训练 & 外推。

---

## 一、长上下文外推（→ §8）

### Q1.1 PI / NTK-aware / YaRN / LongRoPE 分别是什么思路？ ⭐⭐

**【答 30s】**

| 方法 | 思路 | 训练成本 |
| --- | --- | --- |
| **PI** (Position Interpolation, Chen 2023) | 把位置 m 缩放到 m·(L_train/L_new)，等价"压缩"位置 | 微调 1K step |
| **NTK-aware**（Reddit 用户 bloc97 2023） | 不缩位置，缩 **RoPE base θ**，高频不动低频拉长 | 部分情况免训练 |
| **YaRN** (Peng 2023) | NTK-aware + **attention temperature** 校正 + 分段插值 | 微调更少 |
| **LongRoPE** (Microsoft 2024) | 进化搜索找最优 per-dim 缩放因子 | 一次搜索 + 短微调 |

- PI 是最早最朴素的"等比拉伸"
- NTK 见解：RoPE 高频维度是局部信息（不能改），低频维度是全局位置（可以拉长）
- YaRN 是 Llama-3.1 / Qwen-2.5 长上下文的实际选择
- LongRoPE 把 2M 上下文做到几乎不掉点

**【加分项】**
- 直接外推（不调位置）会让 attention 进入未见过的角度，质量崩塌
- "Lost in the Middle"（Liu 2023）：模型在长上下文中间召回率最差，呈 U 型曲线
- 长上下文不只要 RoPE 改，还要训练数据有真长样本

**【追问】**
1. 为什么 base θ 调高（10000 → 500000）等价 NTK？→ base 大 ⇒ 周期长 ⇒ 长距离不绕圈
2. 1M context 实际能用吗？→ claimed 1M vs effective（RULER@85%）通常打 50% 折扣
3. Lost in the Middle 怎么缓解？→ 长上下文 SFT、加 retrieval 重排、Anthropic 给 prompt 加结构

---

### Q1.2 YaRN 的"attention temperature"为什么有用？ ⭐⭐⭐

**【答 30s】**
- 现象：插值后位置变密，attention 分布会变锐（不希望）
- YaRN 校正：在 attention 上乘一个温度 `t = 1 / sqrt(ln(L_new/L_train) + 1)` 让分布钝化、接近训练时的熵
- 工程上等价于把 softmax 缩放系数从 `1/√d` 改成 `t/√d`

**【追问】**
1. 这个 temperature 怎么推导的？→ 拟合让 attention entropy 在 train/new length 下相等
2. 为什么 ALiBi 不需要这个？→ ALiBi 用线性衰减，外推时不会出现频率混叠问题

---

### Q1.3 KV 量化、KV 驱逐、chunked prefill 是什么？ ⭐

**【答 30s】**
- **KV 量化**：把 KV cache 用 int8/int4 存（GPU 上 fp16 → int8 显存砍半，质量损失 < 1 PPL）
- **KV 驱逐**：超长上下文时只保留高 attention score 的 KV，丢掉其余
  - **H2O** (Zhang 2023)：keep heavy-hitter
  - **SnapKV** (Li 2024)：用 last window 的 attention 选 top-k
  - **Scissorhands**：固定预算 + 持续淘汰
- **Chunked prefill**：把长 prompt 切成块依次 prefill，避免一次性算 OOM + 不阻塞 decode

**【加分项】**
- KV 量化在 vLLM/SGLang/TRT-LLM 都支持，是 1M context 的关键
- KV 驱逐有损，对召回敏感任务（needle-in-haystack）会掉点
- chunked prefill 是 SARATHI / vLLM v0.5+ / SGLang 的标准做法

**【追问】**
1. KV 4-bit 怎么选 group size？→ 通常 64 或 128，越小质量越好但 dequant 开销大
2. H2O 在 chat 场景能用吗？→ 多轮对话里 system prompt 容易被驱逐，要做特殊保护
3. Chunked prefill 和 PD 分离冲突吗？→ 不冲突，PD 分离是把 prefill 给独立机器，chunked 是单机内调度

---

### Q1.4 NIAH 测试和 RULER 区别？为什么 NIAH 现在"已经被打穿"？ ⭐⭐

**【答 30s】**
- **NIAH** (Needle in a Haystack, Greg Kamradt 2023)：在长 context 里塞一句"魔法数字是 X"再问。**所有 frontier 模型在 128K 内基本满分**——但这只测召回单点，不测推理
- **RULER** (NVIDIA 2024)：13 个子任务，包括 multi-needle、variable tracking、CWE、QA，**用 85% 通过率定义 effective length**
- 结论：claimed 128K 的模型 RULER effective 通常 ~64K

**【加分项】**
- LongBench v2 加了真实长文档 QA
- ∞Bench 测到 2M token
- BABILong 是用 bAbI 任务包在长 distractor 里
- NoCha 是未发布小说（防污染）

**【追问】**
1. effective length 是怎么定义的？→ RULER 平均得分 ≥ 85% 的最大长度
2. NIAH 在 multi-needle 下表现差，为什么？→ 多个 needle 互相干扰，模型不擅长同时定位多目标

---

## 二、稀疏注意力（→ §6）

### Q2.1 DeepSeek 稀疏注意力演化：NSA → DSA → CSA+HCA？ ⭐⭐⭐

**【答 30s】**

| 版本 | 方法 | 关键 |
| --- | --- | --- |
| **NSA** (V2.5, 2025-02) | Native Sparse Attention，**原生可训练** | 分支 compress / select / sliding 三种 path 加权 |
| **DSA** (V3.2, 2025-09) | **Lightning Indexer + 细粒度 top-k** | indexer 选 top-k token，sparse attention 只算这些 |
| **CSA + HCA** (V4, 2026-04) | 分层压缩 | CSA 细粒度 KV / HCA 128 token 压成 1 个 entry |

- NSA 解决"训练侧也要稀疏"的关键问题——原来都是 inference-only 稀疏
- DSA 的 Lightning Indexer 是小网络快速打分选 top-k
- CSA+HCA 是 1M+ 上下文的关键

**【加分项】**
- NSA 三分支：compress（块平均）、select（top-k 块）、sliding（局部窗口），attention 输出加权融合
- DSA 比 NSA 又省 ~70% KV 访问，质量持平
- DeepSeek 这条线和 MoBA 路线（Moonshot）并行存在

**【追问】**
1. NSA 怎么端到端训稀疏？→ 三分支可微 + 软 gating，避免 hard top-k 的不可微问题
2. Lightning Indexer 多大？→ 极小（几个 head 的 attention），开销 < 5%
3. 为什么不用 Longformer 那种固定窗口？→ 固定窗口表达力弱，DSA 是"学出来的稀疏"

---

### Q2.2 MoBA 和 DeepSeek NSA 区别？ ⭐⭐⭐

**【答 30s】**
- **MoBA** (Moonshot Kimi 2025-02)：**把 MoE 思想用到 attention block** —— 每个 query 通过 top-k routing 选 K 个 KV block 算 attention
- **NSA** (DeepSeek)：固定三分支结构（compress/select/slide）+ 训练时全开
- 共同点：训练侧就稀疏、不是 inference-only patch
- 区别：MoBA 是动态 router、NSA 是固定分支；MoBA 在 chunk 粒度 routing、NSA 在 block 粒度选

**【追问】**
1. MoBA 的 router 怎么训？→ 用辅助 loss 鼓励负载均衡，类似 MoE
2. 哪个工程化更早？→ MoBA 在 Kimi K1.5 / K2 用，NSA 在 DeepSeek-V2.5/V3 用
3. 二者能合并吗？→ 理论可以但实际没人做

---

### Q2.3 经典稀疏（Longformer / BigBird / Streaming-LLM / H2O）思路？ ⭐

**【答 30s】**

| 方法 | 思路 |
| --- | --- |
| **Longformer** (Beltagy 2020) | 滑动窗口 + 部分全局 token |
| **BigBird** (Zaheer 2020) | 滑动 + 随机 + 全局，三类组合理论保证完整性 |
| **Streaming-LLM** (Xiao 2023) | 保留 **sink token**（前 4 个）+ 最近窗口；解决长流式 chat 崩塌 |
| **H2O** (Zhang 2023) | KV 驱逐：保留 high attention score "heavy hitter" |

**【加分项】**
- Streaming-LLM 的 sink token 现象：**前几个 token 吸收了大量注意力，丢了就崩**——这个观察后来被解释为 attention 必须找地方"倾倒"概率质量
- H2O 是 inference-only KV 驱逐，不是结构改造
- Longformer 那一代主要做 encoder-only 长文档，autoregressive 用得不多

**【追问】**
1. sink token 为什么存在？→ softmax 强制和为 1，没强相关时 attention 会均匀流向某些固定位置
2. 这些经典稀疏现在还用吗？→ Streaming-LLM 概念仍活；其他被 NSA/DSA/MoBA 替代

---

## 三、SSM / 线性注意力 / 混合架构（→ §7）

### Q3.1 Mamba 和 Mamba-2 核心是什么？ ⭐⭐

**【答 30s】**
- **SSM (State Space Model)**：用状态空间方程 `h_t = A·h_{t-1} + B·x_t, y_t = C·h_t` 建模序列
- **Mamba** (Gu & Dao 2023)：**selective SSM**——让 A, B, C 依赖输入（content-aware），从 LTI 变成 LTV；用 hardware-aware parallel scan 高效训练
- **Mamba-2** (Dao 2024)：State-Space Duality (SSD)——证明 SSM 和线性注意力在某条件下数学等价；引入 matrix-form 用 GPU GEMM 加速

**【加分项】**
- Mamba 的"selective"是其超越 S4 的关键：input-dependent 让模型可以学到"过滤"
- Mamba-2 用 head 结构（multi-head SSM）让训练更稳定
- 推理时 SSM **常数 O(1) 内存 per step**（相比 Transformer O(seq_len)）

**【追问】**
1. Mamba 训练复杂度？→ O(L log L) via parallel scan
2. Mamba 推理复杂度？→ O(1) per step，无 KV cache
3. Mamba 缺什么？→ 召回能力弱（无法精确召回历史 token），不擅长 in-context learning 重计算

---

### Q3.2 线性注意力为什么"线性"？怎么从二次降到线性？ ⭐⭐⭐

**【答 30s】**
- 标准 attention：`softmax(QK^T) V`，二次复杂度
- 线性 attention：把 softmax 换成核函数分解 `φ(Q) φ(K)^T V`，由结合律：`φ(Q) (φ(K)^T V)`
- 后者复杂度 O(n·d²)，对长序列线性
- 代价：softmax 的强非线性丢了，召回能力变差

**【加分项】**
- φ 可以是 `elu(·)+1`、ReLU²、Performer 的随机特征等
- RWKV、RetNet、Lightning Attention 都是线性 attention 变种
- DeltaNet (Yang 2024) 把 delta rule 引入 linear attention，召回大幅改善

**【追问】**
1. RWKV 和 RetNet 谁先？→ RWKV (Peng 2023) 早；RetNet (Sun 2023) 引入 multi-scale decay
2. 线性 attention 适合什么任务？→ 长序列、低召回要求；不适合精确检索
3. MiniMax-01 用什么？→ Lightning Attention（线性 attention 工程化版）+ MoE

---

### Q3.3 混合架构：Jamba / Nemotron-H / MiniMax-01 / Hunyuan-TurboS 怎么混？ ⭐⭐

**【答 30s】**

| 模型 | 比例 | 备注 |
| --- | --- | --- |
| **Jamba** (AI21 2024) | 1 Attention : 7 Mamba + MoE | 256K context |
| **Zamba** | Mamba 为主 + 共享 attention block | 小模型 |
| **Nemotron-H** (NVIDIA 2024) | ~8% Attention + ~92% Mamba | 8K → 256K |
| **Hunyuan-TurboS** (腾讯 2025) | hybrid Transformer-Mamba + MoE | 长上下文 |
| **MiniMax-01** | 7 Lightning Attention : 1 Transformer + MoE | 4M context |

**【加分项】**
- 混合思想：**Mamba/线性管"主流"，Transformer 管"召回 + ICL"**
- 比例上 Attention 5-15% 是 sweet spot
- MiniMax-01 是工业级 4M context 落地最激进的混合架构

**【追问】**
1. 为什么不能纯 Mamba？→ 召回弱、ICL 弱、Needle-in-Haystack 大幅掉
2. attention block 放在哪？→ 通常每隔 N 层放一个；不放在前几层（前层主要学局部）
3. 混合架构训推有什么坑？→ 不同 layer 类型对并行化 / cache 处理逻辑不同，框架要单独支持

---

## 四、关键问答（综合）

### Q4.1 用一句话说"为什么长上下文模型实际用起来效果都打折扣"

> 训练 vs 评测的 mismatch：训练时长样本少 + RoPE 外推近似 + KV 驱逐有损 + Lost in the Middle。所以 RULER effective length 通常是 claimed 的 50%，1M 模型 effective ~500K。

### Q4.2 给我说三种主流长上下文技术线

1. **RoPE 外推**：PI → NTK → YaRN → LongRoPE
2. **稀疏注意力**：NSA / DSA / MoBA
3. **混合架构**：Jamba / MiniMax-01（Mamba/线性 attention + Transformer）

### Q4.3 Mamba 为什么没"取代" Transformer？

> 召回弱：精确取回历史 token、in-context learning 都比 Transformer 差。所以纯 Mamba 在 ICL 重的任务（few-shot QA、RAG）打不过同规模 Transformer。但**混合架构**已经在长上下文上正胜 Transformer。

### Q4.4 NSA / DSA / MoBA 都是稀疏，差别在哪？

> NSA：固定三分支（compress/select/slide）训练侧就稀疏；DSA：Lightning Indexer 动态选 top-k；MoBA：MoE 风格 router 选 block。NSA → DSA → CSA+HCA 是 DeepSeek 内部演化路线，MoBA 是 Moonshot 平行路线。

### Q4.5 你怎么估算 70B 模型在 1M context 下 KV cache 显存？

> Llama-3 70B GQA 8 KV head × 128 dim × 80 layer × 2 byte × 2 (KV) × 1M = 320KB/token × 1M ≈ **320 GB**。所以 1M context **必须做 KV 量化 + 驱逐 + PD 分离**，单机几乎不可能。MLA 模型同样 1M context KV 只要 ~20GB。

### Q4.6 H2O 类 KV 驱逐策略对什么任务损害最大？

> 召回敏感任务：needle-in-haystack、多轮 chat 早期 system prompt、长文档 QA 引用早期段。chat 场景一般要给 system prompt 做"piano（保护）"标记。

### Q4.7 Streaming-LLM 的 sink token 现象给了 LLM 解释性什么启发？

> attention 必须找地方倾倒概率质量。"Quiet attention is fictional"——softmax 强制和=1，如果没真信号要 attend，就找前几个固定位置。这是 attention 机制的固有结构性质。Anthropic 的 "attention sinks" 和 OpenAI 的 "off-by-one softmax" 都从这里来。

---

## 五、参考资料

### 长上下文外推
- ⭐⭐ [Position Interpolation (Chen 2023)](https://arxiv.org/abs/2306.15595)
- ⭐⭐ [YaRN (Peng 2023)](https://arxiv.org/abs/2309.00071)
- ⭐⭐ [LongRoPE (Ding 2024)](https://arxiv.org/abs/2402.13753)
- ⭐ [Lost in the Middle (Liu 2023)](https://arxiv.org/abs/2307.03172)
- ⭐⭐ [Streaming-LLM / Attention Sinks (Xiao 2023)](https://arxiv.org/abs/2309.17453)

### KV 优化
- ⭐⭐ [H2O (Zhang 2023 NeurIPS)](https://arxiv.org/abs/2306.14048)
- ⭐ [SnapKV (Li 2024)](https://arxiv.org/abs/2404.14469)
- ⭐ [SARATHI (Chunked Prefill, Agrawal 2023)](https://arxiv.org/abs/2308.16369)

### 稀疏注意力
- ⭐⭐ [Longformer (Beltagy 2020)](https://arxiv.org/abs/2004.05150)
- ⭐⭐ [BigBird (Zaheer 2020)](https://arxiv.org/abs/2007.14062)
- ⭐⭐ [DeepSeek NSA (2025-02)](https://arxiv.org/abs/2502.11089)
- ⭐⭐ [DeepSeek-V3.2 DSA Tech Report](https://github.com/deepseek-ai/DeepSeek-V3.2)
- ⭐⭐ [MoBA (Moonshot 2025-02)](https://arxiv.org/abs/2502.13189)

### SSM / 线性注意力 / 混合
- ⭐⭐ [Mamba (Gu & Dao 2023)](https://arxiv.org/abs/2312.00752)
- ⭐⭐ [Mamba-2 / SSD (Dao 2024)](https://arxiv.org/abs/2405.21060)
- ⭐ [RWKV-v7 (Peng 2024)](https://arxiv.org/abs/2404.05892)
- ⭐ [RetNet (Sun 2023)](https://arxiv.org/abs/2307.08621)
- ⭐⭐ [Jamba (AI21 2024)](https://arxiv.org/abs/2403.19887)
- ⭐⭐ [MiniMax-01 Tech Report (2025)](https://arxiv.org/abs/2501.08313)
- ⭐ [Nemotron-H (NVIDIA 2024)](https://research.nvidia.com/)

### 评测
- ⭐⭐ [RULER (NVIDIA 2024)](https://arxiv.org/abs/2404.06654)
- ⭐ [LongBench v2 (THUDM 2024)](https://arxiv.org/abs/2412.15204)

### 解读
- ⭐⭐ [Sasha Rush — Mamba blog](https://srush.github.io/annotated-mamba/)
- ⭐ [Hazy Research blog — SSM / FlashAttention 系列](https://hazyresearch.stanford.edu/blog)

---

→ 下一节 [14.4 MoE & 训练并行题](./04-moe-训练并行题.md)
