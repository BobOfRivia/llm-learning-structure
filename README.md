# 大模型复习准备

> 目标：一个月内完成 LLM 算法主流知识的系统梳理。
> 本文是**知识框架目录**（index），具体每章的笔记单独建立 `notes/<章节>.md` 后逐步填充。

---

## 补充知识（跨章节复用的工具笔记）

- [熵与交叉熵（从"信息量"讲起）](notes/00-补充知识/05-熵与交叉熵.md) —— 所有分类 / LM / 蒸馏 loss 的地基，KL 的前置
- [KL 散度作为损失函数](notes/00-补充知识/01-kl散度作为损失函数.md) —— DSA 蒸馏 / RLHF / DPO / VAE 都要用
- [反向传播与链式法则](notes/00-补充知识/02-反向传播与链式法则.md) —— MoE 路由 / softmax 反传 / stop-gradient 都要用
- [logits 与归一化函数](notes/00-补充知识/03-logits-与归一化函数.md) —— 分类 / LM head / attention / MoE router / 采样 都要用
- [GPU 运算基础 (GEMM / Roofline / Tensor Core)](notes/00-补充知识/04-gpu运算基础-gemm-roofline.md) —— MFU / FlashAttention / batched GEMM / CUDA Graph / 量化 都要用
- [采样与解码策略 (greedy / beam / temperature / top-k / top-p / min-p)](notes/00-补充知识/06-采样与解码策略.md) —— §11 rollout / §12.3 推测解码 / §12.6 self-consistency / §13 评测 都要用

---

## 框架总览（7 大模块 / 14 章）

```
Part I    基础架构        §1 §2 §3
Part II   注意力效率体系   §4 §5 §6 §7 §8
Part III  模型容量扩展     §9
Part IV   训练体系         §10 §11
Part V    推理体系         §12
Part VI   评测            §13
Part VII  面试考点         §14
```

---

## Part I  基础架构

### §1  Tokenization & 数据 & Scaling Laws
> 训练任何模型前必须先理解的"前置条件"。

- 1.1 Tokenization：BPE / BBPE / SentencePiece / tiktoken；中英文/代码混合的实现差异
- 1.2 预训练数据工程：去重、质量过滤（FineWeb-Edu 思路）、合成数据、数据配比
- 1.3 Scaling Laws：Kaplan 2020 → **Chinchilla 2022（计算最优 ~20 tokens/param）** → 后 Chinchilla 修正
- 1.4 训练目标：Next-Token Prediction / Cross-Entropy / Perplexity

### §2  经典 Transformer + 现代化改进
> 把 2017 原版和 2025 主流 LLM 的差异讲清楚。

- 2.1 Self-Attention：Q/K/V、Scaled Dot-Product、Multi-Head、Causal Mask
- 2.2 位置编码演化：**Sinusoidal → Learned → RoPE → ALiBi**（重点：RoPE 数学推导 + 为什么赢）
- 2.3 Norm 演化：**LayerNorm → RMSNorm**；**Post-Norm → Pre-Norm**（训练稳定性）
- 2.4 FFN 演化：ReLU FFN → **SwiGLU / GeGLU**（GLU 家族）
- 2.5 残差结构 & 梯度流
- 2.6 Encoder-Decoder 的 train / infer 完整流程

### §3  三大架构范式
- 3.1 **Decoder-only**（GPT、Llama、Qwen、DeepSeek）—— 主流胜出
- 3.2 **Encoder-only**（BERT、RoBERTa）—— 表征/分类任务
- 3.3 **Encoder-Decoder**（T5、BART、Flan-T5）—— 仍在 seq2seq 场景活跃
- 3.4 In-Context Learning 与涌现能力（Emergence）

---

## Part II  注意力效率体系

### §4  KV-Cache & 推理经济学
> **§5–§8 的所有优化都在攻这一节定义的指标。**

- 4.1 KV-Cache 原理与显存公式：`2 · n_layer · n_head · head_dim · seq_len · dtype`
- 4.2 **Roofline 模型**：算力 vs 带宽
- 4.3 **Prefill (compute-bound) vs Decode (memory-bound)** —— 整个 LLM 推理优化的分水岭
- 4.4 Arithmetic Intensity / FLOPs / MFU / HFU 指标体系（训练-推理统一视角）
- 4.5 RDMA & GPUDirect（Disaggregated Serving 的网络基石）

### §5  稠密注意力优化
> 不改变注意力数学等价性的"纯工程加速"。

- 5.1 **FlashAttention 1 / 2 / 3**：IO-aware tiling、warpgroup、FP8（H100/H200）
- 5.2 **KV 维度压缩派**：MHA → **MQA → GQA → MLA**（DeepSeek）
- 5.3 工程实现：PyTorch SDPA、xFormers、CuDNN Attention

### §6  稀疏 / 压缩注意力
> 引入近似但保证质量。

- 6.1 经典稀疏模式：Longformer、BigBird、Streaming-LLM、H2O
- 6.2 DeepSeek 稀疏路线演化：
  - **NSA**（2025-02，Native Sparse Attention，原生可训练）
  - **DSA**（V3.2，2025-09，Lightning Indexer + 细粒度 top-k）
  - **CSA + HCA**（V4，2026-04，分层压缩；CSA 细粒度、HCA 把 128 token 压成 1 个 entry）
- 6.3 **MoBA**（Moonshot，2025）：把 MoE 思路用到 attention block 路由

### §7  线性注意力 / SSM / 混合架构 ⭐ 新增
> Transformer 之外的"次线性注意力"流派 + 主流的混合方案。

- 7.1 线性注意力派：**RWKV (v6/v7)**、**RetNet**、**Lightning Attention**（MiniMax-01）、GLA、DeltaNet
- 7.2 SSM 派：**Mamba** → **Mamba-2**（与线性注意力的统一视角）
- 7.3 混合架构：**Jamba**（Transformer+Mamba+MoE）、Zamba、**Nemotron-H**、**Hunyuan-TurboS**、**MiniMax-01**
- 7.4 选型判断：什么任务/上下文长度下选 hybrid，什么时候纯 Transformer 仍占优

### §8  长上下文（训练 + 外推）
- 8.1 训练侧：长上下文 curriculum、数据合成
- 8.2 推理侧位置外推：**Position Interpolation → NTK-aware → YaRN → LongRoPE**
- 8.3 推理侧 KV 节省：KV 量化、KV 驱逐（H2O、SnapKV）、chunked prefill
- 8.4 [**Lost in the Middle 产生的原理**](notes/08-长上下文/04-lost-in-the-middle.md)：causal 掩码 + softmax 稀释 + RoPE 衰减 三根因（U 型召回）

---

## Part III  模型容量扩展

### §9  MoE
- 9.1 路由：Top-K Routing、Expert Choice Routing
- 9.2 负载均衡：Aux Loss → **Aux-Loss-Free**（DeepSeek-V3）
- 9.3 **细粒度专家 + 共享专家**（DeepSeek 关键改进）
- 9.4 **Expert Parallelism (EP)** 与 All-to-All 通信
- 9.5 MoE vs Dense 的 FLOPs / 显存 / 推理时延权衡

---

## Part IV  训练体系

### §10  预训练 + 并行化
- 10.1 预训练流程：data → tokenization → 训练 → checkpointing
- 10.2 **混合精度**：FP32 / FP16 / BF16 / FP8
- 10.3 **显存四块**：参数 + 梯度 + 优化器状态 + 激活（16N 公式 / Adam=12N）
- 10.4 **并行化体系**：
  - **DP / TP / PP / SP / EP / CP（Context Parallel）**
  - **ZeRO 1/2/3 / FSDP**
  - Megatron-LM 实现
  - 3D/4D/5D Parallelism
- 10.5 **知识蒸馏**（属于训练，不是推理）

### §11  后训练（Alignment + Reasoning）
> 这条线决定模型"对齐"和"会推理"。**串读建议先看 [§11.0 章节导论](notes/11-后训练/00-章节导论.md)**（算法主线 + 两条正交话题的双轴结构）。

- 11.1 **SFT**（Supervised Fine-Tuning）
- 11.2 **PEFT 家族**：**LoRA → QLoRA → DoRA → AdaLoRA**（SFT/对齐的轻量化分支）
- 11.3 **RLHF**：Reward Model + **PPO**
- 11.4 **DPO** 及其变种（IPO / KTO / SimPO）
- 11.5 **Reasoning 训练**：
  - **RLVR**（Verifiable Reward）
  - **GRPO**（DeepSeek-R1）
  - 推理蒸馏（R1 → 小模型）
  - **DAPO、GSPO**（2025-2026 演进）
- 11.6 **Agentic RL**：工具使用、多步推理、环境交互
- 11.7 Constitutional AI / RLAIF

---

## Part V  推理体系

### §12  推理优化
> 围绕 §4 的 Roofline 拆解，逐层压成本。

- 12.1 **量化**：
  - **PTQ**：GPTQ / AWQ / SmoothQuant
  - **QAT**
  - 精度档：W8A8 / W4A16 / **FP8 / NVFP4**
- 12.2 **Serving 系统**：
  - **vLLM + PagedAttention**
  - **Continuous Batching**
  - **SGLang + RadixAttention**
  - CUDA Graphs
- 12.3 **推测解码**：Draft Model / **Medusa** / **EAGLE 1/2/3** / Lookahead Decoding
- 12.4 **分离式推理**（Disaggregated Inference）：**Mooncake / DistServe / Splitwise**（prefill-decode 分离）
- 12.5 长上下文推理（承 §8.3）
- 12.6 **Test-time Scaling**：
  - CoT / Self-Consistency / Best-of-N
  - **Tree Search**（ToT / MCTS / rStar）
  - o1/o3、R1 范式：训练 + 推理双侧 scaling
  - "更长 ≠ 更好"：Short-m@k 等反直觉发现

---

## Part VI  评测

### §13  评测体系
- 13.1 通用：MMLU / MMLU-Pro / BBH
- 13.2 数学：GSM8K / MATH / **AIME**
- 13.3 代码：HumanEval / MBPP / **SWE-bench / SWE-bench Verified**
- 13.4 长上下文：RULER / Needle-in-a-Haystack / LongBench
- 13.5 Agent：GAIA / WebArena / OSWorld
- 13.6 人评 / 偏好：**Arena (Chatbot Arena) Elo**
- 13.7 评测污染（contamination）与"过拟合榜单"问题

---

## Part VII  面试考点

### §14  面试考点（Q&A + 手撕 + 系统设计）
> 把前 13 章压缩成"30 秒标准答案 + 加分项 + 高频追问"。

- 14.1 [基础架构高频题](notes/14-面试考点/01-基础架构高频题.md)（§1-§3）
- 14.2 [注意力 & 推理经济学题](notes/14-面试考点/02-注意力-推理经济学题.md)（§4-§5）
- 14.3 [长上下文 / 稀疏 / SSM 题](notes/14-面试考点/03-长上下文-稀疏-ssm题.md)（§6-§8）
- 14.4 [MoE & 训练并行题](notes/14-面试考点/04-moe-训练并行题.md)（§9-§10）
- 14.5 [后训练 & Reasoning 题](notes/14-面试考点/05-后训练-reasoning题.md)（§11）
- 14.6 [推理优化题](notes/14-面试考点/06-推理优化题.md)（§12）
- 14.7 [手撕代码 & 系统设计题](notes/14-面试考点/07-手撕代码-系统设计题.md)（跨章）

---

## 学习建议（不展开，待下一轮讨论）

- 优先级排序：§2 §4 §5 §6 §9 §10 §11 §12 是面试/工程的高频深水区
- §7 §8 是 2025–2026 增量重点
- §1 §3 §13 是必备底座但相对易掌握
- **§14 是面试冲刺最后两周的重点：每节口播一遍 + 手撕代码至少写 3 段**

## 后续动作

- [x] 为每章建立 `notes/<chapter>.md`
- [ ] 每章列必读论文清单（→ materials.md）
- [x] 每章列面试高频考点（→ §14）
- [ ] 一个月日程拆解（待下一轮讨论）
