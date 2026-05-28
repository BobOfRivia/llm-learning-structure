# 3.2 Encoder-only

[← 返回框架](../../README.md) · [📎 materials.md → §3.2](../../materials.md)

---

## 一、定义与结构

只有 **双向 self-attention + FFN** 的 N 层堆叠，没有 causal mask。
训练目标：**MLM（Masked Language Modeling）+ 可选的 NSP / SOP**。

```
Tokens: [CLS] x₁ x₂ [MASK] x₄ ... [SEP]
            │
        Embedding + Position + Segment
            │
   ┌────────▼────────┐
   │  N × Block:     │
   │  Bidirectional  │
   │  Self-Attention │
   │  + FFN          │
   └────────┬────────┘
            │
       Hidden states
            │
   ┌────────┴─────────┐
   │                  │
[CLS] head        MLM head (预测被 mask 的 token)
(分类 / 表征)
```

**关键特征**：
- attention 是**完全双向**（mask 矩阵全 0）
- 输入与输出长度相同
- **不能直接做生成**（没有 causal 结构）
- 输出可作"表征向量"（[CLS] / mean pooling）

---

## 二、代表模型谱系

| 时间 | 模型 | 关键贡献 |
|------|------|---------|
| 2018 | **BERT** | MLM + NSP，定义了"预训练 + 微调"范式 |
| 2019 | **RoBERTa** | 去掉 NSP，更长训练，更大 batch；BERT 最佳工程版 |
| 2019 | **ALBERT** | 参数共享 + factorized embedding，小模型化 |
| 2019 | **DistilBERT** | 蒸馏到 66M |
| 2020 | **ELECTRA** | 用判别式 RTD 替代 MLM，效率高 |
| 2019 | **XLM / XLM-R** | 多语言版 |
| 2019 | **SciBERT / BioBERT / FinBERT** | 领域版 |
| 2020-21 | **DeBERTa / DeBERTa-v3** | 解耦位置编码 + ELECTRA-style 训练，SuperGLUE 巅峰 |
| 2024 | **ModernBERT** | RoPE + GeGLU + FlashAttention 升级，长上下文 8k |

---

## 三、训练目标

### 3.1 MLM（Masked Language Modeling）

随机选 15% token，对每个被选中的：
- 80% 替换为 `[MASK]`
- 10% 替换为随机 token
- 10% 保持不变

为什么这个比例？
- 全部用 `[MASK]` → 训推不一致（推理时没有 `[MASK]`）
- 加入随机/保持 → 强迫模型对所有位置都建模

损失：
$$\mathcal{L}_{MLM} = -\mathbb{E}_{M} \sum_{i \in M} \log p(x_i \mid x_{\setminus M})$$

### 3.2 NSP（Next Sentence Prediction，已弃用）

给定两段 A、B，预测 B 是否真接在 A 后。
RoBERTa 实验证明 NSP **几乎无用**，去掉后效果更好。
SOP（Sentence Order Prediction，ALBERT）是 NSP 的精炼版。

### 3.3 RTD（Replaced Token Detection，ELECTRA）

更高效的替代：
- 小 generator 模型在每个位置生成候选 token
- 大 discriminator 判断每个 token 是真还是被替换的
- 损失对所有位置生效 → 比 MLM 数据效率高 4×

---

## 四、双向 attention 的代价

| 维度 | 优点 | 缺点 |
|------|------|------|
| 表征质量 | 每个 token 能看到完整上下文，**句子 / 段落 embedding 最强** | — |
| 训练效率 | — | 每 batch 只学 15% 位置（MLM） |
| 生成 | — | **不能自回归生成** |
| In-context learning | — | 结构上做不到 |
| KV-cache | — | 无意义（没有自回归） |

> ELECTRA / DeBERTa-v3 通过 RTD 解决数据效率，但生成与 ICL 的硬伤无法绕过。

---

## 五、Encoder-only 的应用场景

### 5.1 仍占优势的领域

- **检索 / Embedding**：BGE / E5 / GTE 等顶级 retriever 多以 BERT 系为基（虽然现在被 LLM-based embedding 追上）
- **分类 / NER / 关系抽取**：低延迟、低显存场景
- **重排序（Reranker）**：交叉 encoder 输入 (query, doc) 出分数
- **特定领域 NLU**：FinBERT / BioBERT 等
- **关键词 / 实体抽取**：作为生产 pipeline 的低成本模型

### 5.2 已被 LLM 取代的领域

- 摘要、翻译、问答生成
- 多任务零样本能力
- 通用知识问答

---

## 六、ModernBERT（2024）— 编码器的回归

由 Answer.AI / LightOn / HuggingFace 联合推出（[Warner et al. 2024](https://arxiv.org/abs/2412.13663)），把现代 LLM 工程经验回移到编码器：

| 升级 | 来源 |
|------|------|
| **RoPE** | Llama |
| **GeGLU** | T5-v1.1 |
| **alternating global / local attention** | Mistral |
| **8192 token context** | RoPE + 长上下文训练 |
| **FlashAttention 2** | — |
| **去掉 padding token**（unpadding） | — |
| **2 T tokens 训练**（远超 BERT 3.3B） | — |

**结果**：在分类 / 检索 / 代码搜索上全面超过 BERT-large，且延迟更低。
> "Encoder is back."

---

## 七、Decoder-only 做 Embedding 的崛起

2024 起一个反向趋势：用 decoder-only LLM 做 embedding：

- **E5-Mistral-7B-Instruct**：把 Mistral 7B 微调成 retriever
- **BGE-M3 / Qwen-3-Embedding / NV-Embed**：基于 LLM 的多语言通用 embedding
- **GritLM**：generative + embedding 双任务训一个模型

做法：
- 取最后一个 token / 加专门的 `<eos>` token 的 hidden state
- 用 InfoNCE 对比学习微调
- 长上下文 (~8k+) 和多语言能力比 BERT 系强

**结论**：在通用 embedding 上 LLM-based 已超过 BERT-based；但小模型 / 低延迟 / 领域 NLU 场景，编码器仍有性价比。

---

## 八、Encoder-only 与 Decoder-only 的本质差异

| 维度 | Encoder-only | Decoder-only |
|------|--------------|--------------|
| Attention | 双向 | causal |
| 训练目标 | MLM / RTD | next-token CLM |
| 训练并行 | 整段并行 | 整段并行 |
| 推理生成 | ❌ | ✅ autoregressive |
| 表征强度 | ⭐ 高 | 通过 prompt / pooling 可达 |
| 通用性 | 单点强 | 全任务覆盖 |
| 主流地位 | 小模型 / 特定 NLU | 通用 LLM ⭐ |

---

## 关键问答

**Q1**：BERT 训推不一致的具体表现？
- 训练时 15% token 被替换为 `[MASK]`
- 推理时没有 `[MASK]` → 模型见过的输入分布和实际推理分布不一致
- 80/10/10 比例（mask/random/keep）是部分缓解
- ELECTRA 的 RTD 彻底解决（用判别式目标）

**Q2**：为什么 RoBERTa 比 BERT 强这么多？
- 更长训练（10x 数据）
- 去掉 NSP
- 动态 mask（每个 epoch 重新选 mask 位置）
- 更大 batch（8k）
- 表明 BERT 训练严重欠拟合，工程改进收益巨大

**Q3**：ELECTRA 为什么数据效率高？
- MLM 只在 15% 位置算 loss
- RTD 在 **100% 位置**做二分类（真 / 替换）
- 训练信号密度高 4×
- 是 ELECTRA 在小算力下超 BERT 的根本原因

**Q4**：DeBERTa 的"解耦"是什么？
- 把 token embedding 和 position embedding 拆开
- 计算 attention 时同时用 content-content、content-position、position-content 三种交互
- DeBERTa-v3 + ELECTRA 是 SuperGLUE 长期 SOTA

**Q5**：现在做 NLU 还应该用 BERT 吗？
- 低延迟、固定任务：ModernBERT / DeBERTa-v3 仍最优
- 多任务、需推理：Qwen-2.5-1.5B 等小 LLM
- Embedding：根据语料选 — 多语言通用走 Qwen-3-Embedding / BGE-M3，单语 BERT 仍可

**Q6**：BERT 能做生成吗？
- 直接不能（无 causal 结构）
- 改造路线：
  - BART：BERT encoder + GPT decoder（已属 encoder-decoder）
  - UniLM：用 mask 矩阵切换 attention 方向
  - 直接迭代 unmask：质量差
- 工业不再选 BERT 做生成

---

## 参考资料

- [Devlin et al. 2018 — BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Liu et al. 2019 — RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Lan et al. 2019 — ALBERT: A Lite BERT](https://arxiv.org/abs/1909.11942)
- [Clark et al. 2020 — ELECTRA: Pre-training Text Encoders as Discriminators](https://arxiv.org/abs/2003.10555)
- [He et al. 2021 — DeBERTa V3](https://arxiv.org/abs/2111.09543)
- [Warner et al. 2024 — Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT)](https://arxiv.org/abs/2412.13663)
- [BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity (2024)](https://arxiv.org/abs/2402.03216)
- [E5-Mistral — Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368)
- [GritLM — Generative Representational Instruction Tuning](https://arxiv.org/abs/2402.09906)
- [HuggingFace BERT 教程](https://huggingface.co/learn/nlp-course/chapter1/4)
- [Jay Alammar — The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
