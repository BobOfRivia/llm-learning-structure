# 1.1 Tokenization

[← 返回框架](../../README.md) · [📎 materials.md → §1.1](../../materials.md)

---

## 一、为什么需要 Tokenization

模型只能处理离散 ID 序列，文本必须先映射为整数。三种粒度：

| 粒度 | 优点 | 致命缺点 |
|------|------|----------|
| **词 (word)** | 语义完整 | OOV 严重；中文/代码不友好；词表巨大 |
| **字符 (char)** | 无 OOV；词表极小 | 序列过长；难以学到组合语义 |
| **子词 (subword)** | 折中 | 算法选择多，效果敏感 |

**结论**：现代 LLM 几乎全用 subword（BPE 系）。

---

## 二、主流算法

### 2.1 BPE（Byte Pair Encoding）

**起源**：Sennrich 2016 用于 NMT。

**训练流程**（贪心、自底向上合并）：
1. 把语料切到字符级
2. 统计相邻 pair 频次
3. 选频次最高的 pair 合并成新 token
4. 重复直到达到目标词表大小 V

**示例**：
```
初始: l o w </w>, l o w e r </w>, n e w e s t </w>
合并 (e,s)→es, (es,t)→est, (l,o)→lo ...
```

**推理**：贪心从左到右匹配最长 token。

### 2.2 BBPE（Byte-level BPE）

**关键创新**：先把文本编成 UTF-8 **字节序列**（256 种），再在字节上跑 BPE。

- **GPT-2 / GPT-3 / GPT-4 全部使用**
- 优势：**永无 OOV**（任何 Unicode 都能表示）、语言无关
- 劣势：中文/日文等需多字节，token 利用率低 → 后来扩词表缓解

### 2.3 WordPiece（BERT）

- Google 提出，BERT 使用
- 与 BPE 区别：合并准则不是频次，而是 **似然增益**
  $$\text{score}(A, B) = \frac{\text{freq}(AB)}{\text{freq}(A) \cdot \text{freq}(B)}$$
- 用 `##` 前缀标记非词首子词（如 `playing → play, ##ing`）

### 2.4 Unigram LM

- Kudo 2018 提出，**自顶向下**：从大词表开始，逐步删除使似然下降最少的 token
- 概率模型：每个 token 有概率 p(t)，句子概率 ∏ p(tᵢ)
- 推理时用 Viterbi 找最优分词
- **T5、Llama-1/2/3 使用**（通过 SentencePiece 实现）

### 2.5 SentencePiece（实现框架）

不是新算法，而是 Google 开源的**实现库**，提供 BPE 和 Unigram 两种算法。

**关键设计**：
- **把空格当普通字符**（用 `▁` 表示空格）→ **完全语言无关**，不需要预分词
- 直接吃原始文本（不像 BERT WordPiece 需要先按空格 tokenize）

### 2.6 tiktoken

OpenAI 开源的高性能 BPE **推理库**（Rust 实现 + Python 绑定）。
- `cl100k_base`：GPT-3.5 / GPT-4 用的词表（100k）
- `o200k_base`：GPT-4o 用的词表（200k，多语言效果更好）
- 训练用 OpenAI 内部代码，开源的只有 encoder

---

## 三、关键设计决策

### 3.1 词表大小（V）

| 模型 | V | 备注 |
|------|---|------|
| BERT | 30k | WordPiece |
| GPT-2 | 50k | BBPE |
| GPT-3 | 50k | BBPE |
| Llama-1 / 2 | 32k | SP-BPE（英文为主） |
| **Llama-3** | **128k** | tiktoken 风格，多语言 + 代码 |
| Qwen-2.5 | 152k | 强中英文 |
| GPT-4o | ~200k | `o200k_base` |
| DeepSeek-V3 | 129k | BBPE |

**词表大小 V 的权衡**：
- ↑V：单 token 信息密度高 → 序列短 → 训推快 + 长上下文友好
- ↑V：embedding 层参数膨胀（V × d_model）
- ↑V：每个 token 训练样本变少（数据稀疏）
- 经验：词表大小约为参数量的 0.01–0.05%

### 3.2 特殊 Token

- `<bos>` `<eos>`：序列开始/结束
- `<pad>`：填充
- `<unk>`：未知（BBPE/SentencePiece 通常没有）
- **Chat template tokens**：`<|im_start|>`, `<|im_end|>`（ChatML）；`[INST]`, `[/INST]`（Llama）
- **工具调用 tokens**：`<|tool_call|>` 等

### 3.3 数字 Tokenization

**坑**：BBPE 默认会把 `12345` 切成不规则的子串。

**主流对策**：
- **逐位切分**（GPT-4、Llama-3）：每个数字单独成 token
- **3 位分组**（部分模型）：`123,456,789` → `123`, `456`, `789`
- 这是数学能力的隐藏瓶颈

### 3.4 中文 Tokenization

- 中文无空格 → SentencePiece 把句子当字符串处理
- 词表中通常包含**字 + 常见词**两层
- 早期 Llama 中文 token 利用率差（一字 ≥ 2 token）；Llama-3 / Qwen 显著改善

---

## 四、常见坑 / 面试考点

1. **Glitch tokens**：训练语料中频率极低但保留在词表的 token（如 `SolidGoldMagikarp`），模型在它们上的行为不可预测。
2. **不同 tokenizer 的 token 数差异巨大**：中文文本，Llama-2 大约 1 字 ≈ 2.5 token，Qwen 大约 1 字 ≈ 1 token。
3. **拼写鲁棒性**：`teh` vs `the` 是完全不同的 token。
4. **大小写**：BBPE 区分 `Hello`/`hello`/`HELLO`，会占不同 token。
5. **数字算术能力**：与 tokenization 强相关（参考 Llama-3、GPT-4 都改成逐位）。
6. **训练自定义 tokenizer**：HuggingFace `tokenizers` 库；中英文混合训练时需控制语料配比。

---

## 关键问答

**Q1**：BPE / WordPiece / Unigram 的本质区别？
- 训练方向：BPE/WP 自底向上合并，Unigram 自顶向下删除
- 合并准则：BPE 用频次，WordPiece 用似然增益
- 推理：BPE/WP 贪心匹配，Unigram 用 Viterbi

**Q2**：为什么 GPT 系列坚持 BBPE 而不是 SentencePiece？
- 历史惯性 + BBPE 永无 OOV 的字节级保证；OpenAI 自己用 tiktoken 把 BBPE 工程化到极致。

**Q3**：Llama-3 把词表从 32k 扩到 128k 带来什么？
- 多语言/代码 token 效率提升（中文 ~2× 提升）
- 长上下文实际能装更多信息
- embedding 层增大约 4×，但相对总参数占比小

**Q4**：为什么数字 token 化策略影响数学能力？
- 不规则切分使模型无法稳定学到位值（123 和 1234 的 "23" 含义不同）
- 逐位切分让位值结构暴露给 attention，更易学习进位

**Q5**：tokenizer 改了能继续用原模型吗？
- 不能直接用。embedding/LM-head 维度变了。常见做法：
  - 全量重训
  - "Embedding surgery"：把新词表中老词的 embedding 拷过来，新增的随机初始化后微调

---

## 参考资料

- [Sennrich et al. 2016 — Neural Machine Translation of Rare Words with Subword Units (原始 BPE)](https://arxiv.org/abs/1508.07909)
- [Radford et al. 2019 — GPT-2 paper (BBPE)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Kudo & Richardson 2018 — SentencePiece](https://arxiv.org/abs/1808.06226)
- [Kudo 2018 — Subword Regularization (Unigram LM)](https://arxiv.org/abs/1804.10959)
- [BERT — Devlin et al. 2018 (WordPiece)](https://arxiv.org/abs/1810.04805)
- [HuggingFace Tokenizers 教程](https://huggingface.co/learn/nlp-course/chapter6/1)
- [tiktoken 仓库](https://github.com/openai/tiktoken)
- [Karpathy — Let's build the GPT Tokenizer (视频)](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Llama-3 技术报告（含词表设计讨论）](https://arxiv.org/abs/2407.21783)
