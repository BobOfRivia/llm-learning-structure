# 3.3 Encoder-Decoder

[← 返回框架](../../README.md) · [📎 materials.md → §3.3](../../materials.md)

---

> 本节聚焦"作为一种范式"的特征、适用场景与现状。
> 详细的训练 / 推理流程、cross-attention 细节见 [§2.6](../02-transformer-架构/06-encoder-decoder.md)。

---

## 一、范式定义

**两个独立 stack** 通过 cross-attention 桥接：
```
   Source ── Encoder（双向）──┐
                              ▼
   Target ── Decoder（causal + cross-attn）── Output
```

设计哲学：**给定一段输入 → 生成另一段输出**（seq2seq）。

| 元素 | Encoder | Decoder |
|------|---------|---------|
| Attention | 双向 self-attn | causal self-attn + cross-attn |
| 训练 | 与 decoder 联合（端到端 CE） | next-token CE |
| 输入 | source 序列 | shifted target |
| 输出 | hidden states（不直接预测 token） | softmax → token |

---

## 二、为什么这个范式出现得早

历史顺序：Encoder-Decoder（2014 Sutskever 用 LSTM 起步）→ Transformer Encoder-Decoder（2017）→ BERT Encoder-only（2018）→ GPT Decoder-only（2018）。

最初做的就是**机器翻译**：源语言 → 目标语言，自然形成 encoder-decoder 结构。Transformer 论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 本身就是为翻译设计的。

---

## 三、代表模型

| 模型 | 年份 | 关键贡献 |
|------|------|---------|
| **Transformer (原版)** | 2017 | 架构定义 |
| **BART** | 2019 | denoising autoencoder：mask span / permute / delete |
| **T5** | 2019 | text-to-text 统一范式 + span corruption |
| **mT5 / Flan-T5** | 2020-22 | 多语言 / 指令微调 |
| **mBART / NLLB** | 2020-22 | 多语言翻译 |
| **Pegasus** | 2019 | 摘要专用：gap-sentence prediction |
| **UL2 / Flan-UL2** | 2022 | mixture-of-denoisers，统一 CLM/PrefixLM/MLM |
| **Whisper** | 2022 | 语音 encoder + 文本 decoder ⭐ |
| **Flamingo** | 2022 | 视觉 encoder + LLM decoder（cross-attn 注入） |
| **CodeT5+ / AlphaCode** | 2022-23 | 代码 encoder-decoder |
| **Seamless M4T** | 2023 | 语音-文本-语音多语翻译 |

---

## 四、各种"denoising"训练目标

encoder-decoder 比 decoder-only 多一个自由度：**输入可以被破坏后让模型还原**。

| 目标 | 模型 | 做法 |
|------|------|------|
| Span corruption | T5 | mask 整段，sentinel 占位 |
| Token masking | BART | 单 token mask |
| Token deletion | BART | 删除若干 token，让模型还原位置 |
| Sentence permutation | BART | 打乱句子顺序 |
| Document rotation | BART | 旋转起点 |
| Gap-sentence | Pegasus | 整句 mask（摘要专用） |
| Mixture-of-denoisers | UL2 | R/S/X-denoiser 混合 |

> **Wang et al. 2022** 系统对比：在零样本对齐场景下，causal LM + CLM 胜出；
> 在 supervised fine-tuning 场景下，**span corruption + encoder-decoder 仍最优**。

---

## 五、Encoder-Decoder vs Decoder-only：什么时候应该选哪个

### 5.1 选 encoder-decoder 的情况

- **输入和输出是不同模态/语言**：语音→文本（Whisper）、源语言→目标语言（NLLB）
- **输入定长大、输出定长小**：摘要 / OCR
- **输入需要双向理解**：纯生成不能损失任何输入信息
- **算力受限 + 任务固定**：T5 / BART 微调比同尺寸 LLM 便宜
- **encoder 与 decoder 模态不同**（视觉/语音 encoder + 文本 decoder）

### 5.2 选 decoder-only 的情况

- **通用对话 / 多任务**：in-context learning 关键
- **变长生成 / agent**：需要 reasoning 链
- **持续上下文累积**：连续会话
- **scaling 到很大**：单一 stack 更易工程化

> 现实分界：**输入空间 ≠ 输出空间**时选 encoder-decoder；**同语义空间下文本生成**选 decoder-only。

---

## 六、Cross-Attention 的注入式应用

把 encoder 改成**任意外部模态编码器**：
- 视觉：ViT → cross-attn → LLM（Flamingo / IDEFICS）
- 语音：Conformer / Whisper-encoder → cross-attn → LLM
- 检索：retriever 输出 → cross-attn 注入（Retro / RAG-fusion）

但近 2 年的趋势是**flatten 到 decoder**（把视觉/语音也 tokenize 后拼到序列），原因：
- 工程简单（单 stack）
- 可与 in-context learning 协同
- LLaVA / Qwen-VL / Gemini 都走这条路

> Flamingo 的 cross-attn 方案虽然优雅但已被简化路线压制。

---

## 七、Prefix LM（一种 hybrid）

让 **同一个 decoder** 用不同的 mask 模式：
```
[Prefix bidirectional][Target causal]
```
- Prefix 段用双向 attention
- Target 段用 causal
- 只需一个 stack，但能"理解 + 生成"

代表：GLM-130B、PaLM-1。
现已被 decoder-only 完全压制，但思想被 UL2、ChatGLM 早期版本沿用。

---

## 八、为什么 encoder-decoder 没成为通用 LLM 主流

| 弱点 | 影响 |
|------|------|
| 训练目标与生成不一致 | T5 span corruption 后 fine-tune 阶段仍要 fine-tune |
| In-context learning 弱 | examples 被 encoder 压缩成定长 hidden states，信息有损 |
| 参数效率低 | encoder 与 decoder 各占一半，但有些任务不需要双向 encoder |
| Scaling 困难 | encoder/decoder 平衡是额外超参 |
| 工程复杂 | 两套 attention + cross-attn + 两套 KV-cache |

**结论**：通用对话 LLM 走 decoder-only；定向 seq2seq 任务（翻译 / 语音 / 摘要）保留 encoder-decoder。

---

## 九、现代多模态模型的归属

| 模型 | 范式 | 备注 |
|------|------|------|
| Whisper | Encoder-Decoder | 语音 encoder + 文本 decoder |
| Flamingo | Encoder-Decoder（弱） | 视觉 encoder + cross-attn LLM |
| LLaVA / Qwen-VL | Decoder-only | 视觉 token 拼接 |
| Gemini | Decoder-only（推测） | 多模态 native |
| GPT-4o | Decoder-only | omni-token |
| InstructBLIP | 混合 | Q-Former + LLM |
| Seamless M4T | Encoder-Decoder | 语音-文本-语音 |

---

## 关键问答

**Q1**：encoder 与 decoder 的参数比例怎么选？
- 经典 T5：encoder ≈ decoder（对称）
- BART：decoder 略大于 encoder
- 任务定向：摘要等 input-heavy 任务 → encoder 占比高；翻译 → 对称
- 没有 scaling law 给出明确比例

**Q2**：T5 用 span corruption 不用 CLM 的理由？
- 想让 encoder 学到强表征（双向）
- span 任务模拟"完形填空 + 生成"双重信号
- 但 fine-tune 时 task gap 仍存在 → Flan-T5 用大量指令数据补救

**Q3**：BART 和 T5 选哪个？
- BART：单一 corruption（token mask 为主），decoder 偏强
- T5：span corruption 为核心，encoder 偏强
- 实际工业上 T5 / Flan-T5 更主流（生态成熟）

**Q4**：encoder-decoder 还有未来吗？
- 通用 LLM：基本没有
- 定向 seq2seq（语音 / 翻译 / OCR）：长期保留
- 视觉模型（如部分 image-to-text）也仍在用

**Q5**：Whisper 为什么坚持 encoder-decoder？
- 语音特征（mel spectrogram）与文本是完全不同的模态
- 语音 encoder 处理连续高维特征，文本 decoder 处理离散 token
- 用 cross-attn 桥接是最自然的方式

**Q6**：encoder-decoder 的"cross-attention layer"和 decoder self-attn 在工程上如何复用？
- 实现上是两个独立 attention 模块
- 共享 QKV 投影**没有意义**（K/V 来源不同）
- 但 Norm、FFN 模块可共享

---

## 参考资料

- [Vaswani et al. 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Sutskever et al. 2014 — Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Raffel et al. 2019 — T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)
- [Lewis et al. 2019 — BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Tay et al. 2022 — UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)
- [Zhang et al. 2019 — PEGASUS: Pre-training with Gap-Sentences for Summarization](https://arxiv.org/abs/1912.08777)
- [Radford et al. 2022 — Whisper](https://arxiv.org/abs/2212.04356)
- [Alayrac et al. 2022 — Flamingo: a Visual Language Model](https://arxiv.org/abs/2204.14198)
- [NLLB Team 2022 — No Language Left Behind](https://arxiv.org/abs/2207.04672)
- [Wang et al. 2022 — What Language Model Architecture and Pretraining Objective Work Best?](https://arxiv.org/abs/2204.05832)
- [Chung et al. 2022 — Scaling Instruction-Finetuned Language Models (Flan)](https://arxiv.org/abs/2210.11416)
