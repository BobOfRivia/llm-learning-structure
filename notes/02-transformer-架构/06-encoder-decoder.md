# 2.6 Encoder-Decoder 训练与推理流程

[← 返回框架](../../README.md) · [📎 materials.md → §2.6](../../materials.md)

> **本节聚焦"机制":3 种 attention、teacher forcing、cross-attn、推理 KV 复用等工程细节。范式层面的特征、适用场景与历史地位详见 [§3.3 Encoder-Decoder 范式](../03-架构范式/03-encoder-decoder.md)。**

---

## 一、为什么这一节单独存在

虽然现代 LLM 主流是 decoder-only，但 encoder-decoder 仍是：
- 经典 Transformer 的原始形态（理解架构起点）
- 翻译 / 摘要 / Whisper / T5 / Flan-T5 / NLLB / Seq2Seq 任务的最佳实践
- 多模态（图文、语音）中 encoder 处理输入模态

理解这套训练 / 推理流程能反衬出 decoder-only 的设计取舍。

---

## 二、结构总览

```
        Source                          Target
       (x₁..x_S)                       (y₁..y_T)
           │                               │
       Encoder                         Decoder
       ┌──────────┐                   ┌──────────────────┐
       │ Self-Attn│                   │  Masked Self-Attn│ ← causal
       │ + FFN ×N │                   │  + Cross-Attn   ←┼─── encoder K,V
       │ (双向)   │                   │  + FFN ×N        │
       └──────────┘                   └──────────────────┘
           │                               │
       enc memory                        LM head
           └──────── K,V 喂给 cross-attn ──┘
```

**3 种 attention**：
1. Encoder self-attention：双向（看完整 source）
2. Decoder self-attention：causal（只看 target 历史）
3. Decoder cross-attention：Q 来自 decoder，K/V 来自 encoder 输出

---

## 三、训练流程

### 3.1 Teacher Forcing

训练时 decoder 输入是 **shifted target**：
```
Source:   <bos> Bonjour le monde <eos>
Target in: <bos> Hello world
Target out:    Hello world <eos>
```

每个位置预测下一个 token，loss 是位置平均的 CE。
**整段一次性 forward**（可并行所有时间步，因为有 causal mask）。

### 3.2 Cross-Attention

decoder 每层先做 masked self-attn，再做 cross-attn：
$$Q = W_Q^{dec} \cdot h^{dec}, \quad K = W_K^{enc} \cdot h^{enc}, \quad V = W_V^{enc} \cdot h^{enc}$$

- encoder 输出对 **所有 decoder 位置共享**（一次性算出）
- decoder 每个位置都能"看"整段 source（无 mask）

### 3.3 训练目标

标准 seq2seq：CLM on decoder + 整段 NLL loss
$$\mathcal{L} = -\sum_{t=1}^{T} \log p(y_t | y_{<t}, x)$$

T5 系列用 **span corruption**（§1.4）作为预训练目标。

---

## 四、推理流程

### 4.1 两阶段

```
[Stage 1] Prefill encoder
  - 一次性算 encoder 全部输出 h^enc
  - 缓存为 cross-attn 的 K,V

[Stage 2] Autoregressive decode
  - decoder 从 <bos> 开始一个 token 一个 token 生成
  - 每步:
      1. 当前 token 过 self-attn (用历史 KV cache)
      2. 过 cross-attn (用 encoder K,V — 不变!)
      3. 过 FFN
      4. LM head → softmax → 采样下一 token
```

### 4.2 关键差异：两套 KV-cache

| 类型 | 内容 | 推理时变化 |
|------|------|-----------|
| **Encoder KV (cross-attn 用)** | 整段 source 一次算出 | **不变** |
| **Decoder self-attn KV** | 历史 target | 每生成一 token 增长一行 |

decoder-only 模型只有一种 KV-cache（无 encoder K/V），实现更简单。

### 4.3 Beam Search

机翻 / 摘要任务历史上常用 beam search（保留 top-k 候选）：
- 现代 LLM 几乎都用采样（temperature / top-p）
- beam search 仍是机翻 BLEU 评测的黄金做法

---

## 五、encoder-decoder vs decoder-only

| 维度 | Encoder-Decoder | Decoder-only |
|------|----------------|--------------|
| 参数 | 同尺寸下编码器 + 解码器各占一半 | 全部用于 LM |
| 输入处理 | encoder 双向理解 | causal 逐 token |
| 多任务（生成 + 理解）| 自然 | 需要 prompt 设计 |
| In-context learning | 弱（cross-attn 限制） | 强 ⭐ |
| 训练复杂度 | 2 套 attention 类型 | 1 套 |
| KV-cache | 2 套 | 1 套 |
| 主流采用 | T5 / BART / Whisper / NLLB | GPT / Llama / Claude / 几乎所有现代 LLM |

**核心结论**：通用对话与推理任务上 decoder-only 已胜出（in-context learning 是关键）。
但在**给定固定输入 → 生成输出**的场景（语音识别、机器翻译、文档摘要），encoder-decoder 仍有结构优势。

---

## 六、典型 encoder-decoder 模型

### 6.1 T5（Raffel 2019）
- "Text-to-Text Transfer Transformer"
- 用 span corruption 预训练
- 后续 Flan-T5 / UL2 / mT5 多语言版

### 6.2 BART（Lewis 2019）
- 用多种噪声训练（token masking、permutation、deletion）
- 摘要 / 翻译任务表现强

### 6.3 Whisper（Radford 2022）
- 语音 encoder（mel spectrogram → CNN+Transformer）
- 文本 decoder
- 多任务 token（语言 / 翻译 / 转写 / 时间戳）

### 6.4 NLLB（Meta 2022）
- 200 种语言机翻
- MoE encoder-decoder

### 6.5 多模态扩展
- Flamingo / IDEFICS / InstructBLIP：视觉 encoder + LLM decoder（cross-attn 注入）
- Qwen-VL / LLaVA：视觉 token 直接拼到 decoder 输入（"flatten" 路线）

---

## 七、Prefix LM（一种折中）

PaLM / GLM 用过的混合方案：
- 同一个 decoder，但 prefix 段用双向 attention，target 段用 causal
- 同时具备"双向理解" + "自回归生成"
- 工程上需要 mask 设计精细，且已被 decoder-only 完全压制

---

## 关键问答

**Q1**：cross-attention 的 K、V 来自 encoder 输出，Q 来自 decoder。为什么这样设计？
- decoder 当前位置在"询问"（Q），向 encoder 的全部位置（K、V）取信息
- 这是 attention 的"软检索"语义
- 反过来（encoder Q + decoder K/V）会破坏因果，且违背"信息流方向：source → target"

**Q2**：encoder 输出在生成全程都不变吗？
- 是。encoder 在 prefill 阶段算一次，cross-attn 的 K/V cache 全程复用
- 这点和 decoder-only 不同（decoder-only 的 KV 每步增长）

**Q3**：encoder-decoder 为什么 in-context learning 弱？
- decoder 只能通过 cross-attn 间接访问 examples
- 例子要先被 encoder 压缩成定长 hidden states，信息有损
- decoder-only 直接把 examples 放入 context，attention 可任意检索

**Q4**：T5 为什么没"赢"？
- 训练目标（span corruption）与下游生成不完全一致
- 同等算力下 decoder-only + CLM 更通用
- 2022 GPT-3 后市场惯性彻底倒向 decoder-only

**Q5**：现代多模态模型还用 encoder-decoder 吗？
- Whisper 语音、部分 OCR 模型仍用
- 通用 VLM（GPT-4V、Qwen-VL、LLaVA）多走"视觉编码器输出 → LLM token"路线，本质是 decoder-only
- Flamingo 用过 cross-attn 注入，后被简化

**Q6**：encoder-decoder 的训练 token / FLOPs 怎么算？
- 6ND 法则仍适用，但 N 是 encoder + decoder 总参数
- cross-attn 算 FLOPs 时 Q 长 = decoder L，K/V 长 = encoder L

---

## 参考资料

- [Vaswani et al. 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Raffel et al. 2019 — T5](https://arxiv.org/abs/1910.10683)
- [Lewis et al. 2019 — BART](https://arxiv.org/abs/1910.13461)
- [Sutskever et al. 2014 — Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Bahdanau et al. 2014 — Neural Machine Translation by Jointly Learning to Align and Translate (attention 起源)](https://arxiv.org/abs/1409.0473)
- [Radford et al. 2022 — Whisper](https://arxiv.org/abs/2212.04356)
- [Meta — NLLB 200 多语言机翻](https://arxiv.org/abs/2207.04672)
- [Wang et al. 2022 — What Language Model Architecture and Pretraining Objective Work Best?](https://arxiv.org/abs/2204.05832)
- [Tay et al. 2022 — UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)
- [Karpathy — minGPT vs encoder-decoder 对比代码](https://github.com/karpathy/minGPT)
