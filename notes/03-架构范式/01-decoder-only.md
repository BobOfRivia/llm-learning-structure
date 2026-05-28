# 3.1 Decoder-only

[← 返回框架](../../README.md) · [📎 materials.md → §3.1](../../materials.md)

---

## 一、定义与结构

只有 **causal self-attention + FFN** 的堆叠，目标 next-token prediction（CLM）。

```
Tokens: [x₁, x₂, x₃, ..., x_T]
            │
        Embedding + (RoPE)
            │
   ┌────────▼────────┐
   │  N × Block:     │
   │  ┌──────────┐   │
   │  │ Pre-Norm │   │
   │  │ Causal   │   │
   │  │ Attn     │←─ KV-cache
   │  │ +residual│   │
   │  └────┬─────┘   │
   │  ┌────▼─────┐   │
   │  │ Pre-Norm │   │
   │  │ FFN /    │   │
   │  │ SwiGLU   │   │
   │  │ +residual│   │
   │  └──────────┘   │
   └────────┬────────┘
        Final Norm
            │
        LM head (tied to embedding)
            │
       Softmax → next-token distribution
```

**关键特征**：
- 单一 attention 类型（causal self-attn）
- 训练 = next-token CE，整段并行
- 推理 = autoregressive，一个 token 一个 token
- 输入与输出在同一序列上对齐

---

## 二、代表模型谱系

| 时间 | 模型 | 关键贡献 |
|------|------|---------|
| 2018 | **GPT-1** | 117M，首次证明 generative pretraining + finetune |
| 2019 | **GPT-2** | 1.5B，zero-shot 多任务能力，Pre-Norm 普及 |
| 2020 | **GPT-3** | 175B，**in-context learning 涌现** ⭐ |
| 2022 | **InstructGPT / ChatGPT** | RLHF，开启对齐时代 |
| 2023 | **GPT-4** | MoE 推测，多模态 |
| 2023 | **Llama-1 / 2** | 开源 LLM 基石（RoPE + RMSNorm + SwiGLU + GQA） |
| 2023 | **Mistral 7B** | sliding window + GQA |
| 2024 | **Llama-3 / 3.1** | 8k → 128k 上下文，15T tokens |
| 2024 | **DeepSeek-V2 / V3** | MLA + DeepSeekMoE + FP8 训练 |
| 2024 | **Qwen-2.5 / 3** | 152k 词表 + 长上下文 + 数学 |
| 2024-25 | **o1 / R1 / o3** | 强化学习驱动的推理模型 |
| 2026 | **DeepSeek-V4** | CSA + HCA 稀疏 + 128-token 压缩 |

---

## 三、Decoder-only 现代标配（"Llama 范式"）

2023 年起开源社区基本固化为一套设计：

| 模块 | 选择 | 章节 |
|------|------|------|
| 位置编码 | **RoPE** | §2.2 |
| 归一化 | **RMSNorm** | §2.3 |
| Norm 位置 | **Pre-Norm + Final Norm** | §2.3 |
| 激活 | **SwiGLU** | §2.4 |
| Attention | MHA / **GQA** / MLA | §5.2 |
| 训练目标 | **next-token CE** | §1.4 |
| Tokenizer | **BBPE**（tiktoken）或 SentencePiece-BPE | §1.1 |
| 训练精度 | BF16 / FP8 | §10.2 |

> 这套配置在 Llama-2 后被几乎所有开源模型沿用，差异主要在 attention 变体、MoE、词表上。

---

## 四、为什么 Decoder-only 赢了

### 4.1 训练效率
- 单一注意力类型 → 实现简单 + kernel 优化彻底
- 每个 token 都贡献 loss（MLM 只 15%、Span Corruption 部分位置）
- **数据效率高**

### 4.2 推理简洁
- KV-cache 单一类型，工程简单
- prefill + decode 两阶段非常清晰（§4.3）
- 适合 batch / 流式输出

### 4.3 In-Context Learning（决定性）
- 可以把 few-shot examples 直接放进 context
- attention 在 context 内任意检索
- 这是 encoder-decoder 结构上做不到的能力（§3.4）

### 4.4 通用性
- 同一架构覆盖：对话、reasoning、code、翻译、摘要、agent
- 用 prompt 切换任务，不需要任务特定 head
- 多模态延伸自然（视觉/语音 token 拼到序列）

### 4.5 Scaling 优雅
- 增大 N（层数 / 宽度）、D（数据）就是收益
- 不需要重设计 encoder/decoder 平衡
- 与 MoE、长上下文等扩展方向都兼容

---

## 五、与其它范式的对比

| 维度 | Decoder-only | Encoder-only (BERT) | Encoder-Decoder (T5) |
|------|--------------|---------------------|----------------------|
| Attention | causal | bidirectional | 两端独立 + cross-attn |
| 训练目标 | CLM | MLM | Span corruption |
| 表征任务 | 中（需 prompt） | ⭐ 最强 | 中 |
| 生成任务 | ⭐ 最强 | 不能直接生成 | 强（定长输入场景） |
| In-context learning | ⭐ 有 | 无 | 弱 |
| 训练数据利用率 | ⭐ 每 token | 15% mask | 部分 span |
| 工程复杂度 | ⭐ 低 | 低 | 中 |
| 现代主流 | ⭐⭐⭐ | 检索 / classifier | 翻译 / 语音 |

> Wang et al. 2022 [What Language Model Architecture and Pretraining Objective Work Best?](https://arxiv.org/abs/2204.05832) 在统一算力下系统对比了 8 种组合，最终 **causal decoder + CLM** 在 zero-shot 上稳定胜出。

---

## 六、不变与变化

### 不变的（10 年没换）
- next-token prediction
- 残差 + Norm + FFN 三件套
- attention 作为核心信息混合机制
- causal mask

### 在变的
- attention 变体（MHA → GQA / MLA / DSA / CSA）
- FFN 变体（dense → MoE → fine-grained MoE）
- 位置编码（learned APE → Sinusoidal → RoPE + YaRN）
- Norm（LN → RMSNorm；Post-Norm → Pre-Norm）
- 训练目标补充（CLM → SFT → RLHF → RLVR）
- 长上下文支持（2k → 128k → 1M+）

---

## 七、Decoder-only 的"理论极限"

社区不断质疑：
- **数据墙**：高质人类语料 ~10²-10³T tokens 见顶 → 合成数据 / 多模态
- **架构瓶颈**：causal attention 是否还是最优？→ Mamba / Jamba 等混合架构（§7）
- **推理能力**：next-token 学不出 system-2？→ test-time compute / RLVR
- **效率天花板**：$O(L^2)$ attention → 稀疏 / linear

但每一次"decoder-only 该被替代"的论断都被新工程突破（FlashAttention、MoE、RoPE、RLVR）打回。

---

## 关键问答

**Q1**：Decoder-only 为什么数据利用率高？
- 每个 token 都贡献 CE loss（MLM 只在 mask 位置）
- 同样 D tokens，CLM 学到的有效信号比 MLM 多 ~6×

**Q2**：双向 attention 不是更有信息吗？为什么 decoder-only 还赢？
- BERT 的双向是为了表征任务，但**生成**和 **in-context learning** 是 causal 的天然形态
- 实证：相同算力下，causal LM 在 zero/few-shot 上更强
- 多任务通用性碾压了 BERT 的单点表征优势

**Q3**：Llama 范式（RoPE + RMSNorm + SwiGLU + GQA）会持续多久？
- 2023 至今几乎所有开源模型都沿用
- 最大变化点是 attention（GQA → MLA → 稀疏 / linear）
- FFN 走向 MoE，但单专家仍是 SwiGLU
- 短期看 stack 稳定，attention 与 MoE 是主要演化方向

**Q4**：Decoder-only 模型怎么做表征任务（embedding / 检索）？
- 主流做法：用最后一个 token 的 hidden state（如 OpenAI Ada）
- 或 mean-pooling 所有 token
- **E5-Mistral / BGE-M3 / Qwen-3 Embedding** 等用对比学习微调 LLM 做 embedding
- 在多语言长上下文场景上已超过 BERT 系编码器

**Q5**：为什么 LM head 通常和 embedding 共享权重（weight tying）？
- 节省 V × d 参数（V=128k, d=4096 时省 0.5B 参数）
- 训练初期能加速收敛
- Llama 系列默认 tied，少数大模型（GPT-3）不 tied

**Q6**：Decoder-only 能做"理解"类任务（NLU）吗？
- 完全可以，prompt 工程或加 classifier head
- 工业上很多 NLU 应用已经从 BERT 切到小 LLM + few-shot
- 推理上贵一些，但灵活性收益更大

---

## 参考资料

- [Radford et al. 2018 — GPT-1: Improving Language Understanding by Generative Pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Radford et al. 2019 — GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Brown et al. 2020 — GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Touvron et al. 2023 — Llama: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Touvron et al. 2023 — Llama 2](https://arxiv.org/abs/2307.09288)
- [Llama-3 技术报告](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [Qwen-2.5 技术报告](https://arxiv.org/abs/2412.15115)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [Wang et al. 2022 — What Language Model Architecture and Pretraining Objective Work Best?](https://arxiv.org/abs/2204.05832)
- [Karpathy — nanoGPT (经典 decoder-only 参考实现)](https://github.com/karpathy/nanoGPT)
- [Anthropic — Transformer Circuits](https://transformer-circuits.pub/)
