# 4.1 KV-Cache 原理与显存公式

[← 返回框架](../../README.md) · [📎 materials.md → §4.1](../../materials.md)

---

## 一、为什么需要 KV-Cache

自回归生成的核心痛点：**每生成一个新 token，理论上要重做整段 attention**。

第 $t$ 步生成时：
$$\text{Attention}(Q_t, K_{1..t}, V_{1..t})$$

- $Q_t$：当前 token 的 query（只算这一个位置）
- $K_{1..t}, V_{1..t}$：所有历史 token 的 key / value

观察：**$K_{1..t-1}$ 和 $V_{1..t-1}$ 在第 $t-1$ 步已经算过**，且**不会随后续 token 变化**（causal 模型中 K/V 只依赖于自身的 hidden state）。

→ **缓存历史 K, V，每步只算新 token 的 $K_t, V_t$ 并 append。**

```
没有 KV-cache:
  step 1:  算 K₁,V₁                          → 1 个 token 计算
  step 2:  重算 K₁,K₂,V₁,V₂                  → 2 个
  step 3:  重算 K₁..K₃, V₁..V₃               → 3 个
  ...
  生成 T 个 token 总开销: O(T²) 次 K/V 投影

有 KV-cache:
  step 1:  算 K₁,V₁,存                       → 1 个
  step 2:  算 K₂,V₂,append                   → 1 个
  step t:  算 K_t,V_t,append                 → 1 个
  生成 T 个 token 总开销: O(T) 次 K/V 投影
```

> KV-cache 把 decode 从 $O(T^2)$ 计算降到 $O(T)$，是 LLM 推理工程的基石。

---

## 二、KV-Cache 的存储位置与单条目大小

每层 attention 模块都缓存自己的 K、V。

**单 token、单层的 KV 大小**（MHA）：
$$\text{size}_{token, layer} = 2 \times h \times d_h \times \text{bytes}$$

其中：
- 2：K 和 V
- $h$：head 数
- $d_h$：每 head 维度
- $h \times d_h = d$（模型隐维度）
- bytes：FP16/BF16 = 2，FP8 = 1

**整段序列、整模型的 KV 显存公式**：
$$\boxed{\text{KV-cache total} = 2 \times L \times d \times \text{layers} \times B \times \text{bytes}}$$

| 符号 | 含义 |
|------|------|
| $L$ | 序列长度 (prompt + generated) |
| $d$ | hidden dim |
| layers | 层数 |
| $B$ | batch size |
| bytes | 数据类型字节数 |

---

## 三、典型数字感觉（必须背下来）

**Llama-3 70B（MHA，BF16）**：
- layers = 80, d = 8192, bytes = 2
- 单 token KV ≈ 2 × 8192 × 80 × 2 = **2.6 MB / token**
- 长度 8192：≈ **21 GB**（单条!）
- batch=32, L=8192：≈ **670 GB** → 显然撑不住

**Llama-3 70B GQA（KV head = 8）**：
- KV head 从 64 → 8，KV-cache **缩小 8×**
- 单 token ≈ **0.33 MB**
- batch=32, L=8192：≈ **84 GB**（可承受）

**GQA 是过去 3 年最重要的推理优化之一**，原因就是 KV-cache 直接除以 grouping。

---

## 四、GQA / MQA / MLA 对 KV-Cache 的压缩

| 方案 | KV head 数 | KV 占用比 | 性能影响 |
|------|------------|-----------|----------|
| MHA | h（=query head） | 1× | baseline |
| **GQA** | h/g（一般 g=8） | 1/g | 几乎无损 ⭐ |
| **MQA** | 1 | 1/h | 略损 |
| **MLA** (DeepSeek) | 1 个低秩 latent | ~1/10+ | 持平甚至更好 |

公式变化（GQA，g 组）：
$$\text{KV} = 2 \times L \times \frac{d}{g} \times \text{layers} \times B \times \text{bytes}$$

MLA（Multi-head Latent Attention）走的是另一条路：不是减少 head，而是用 **低秩压缩**：
$$c_t^{KV} = W^{DKV} h_t \quad (\dim c \ll \dim K, V)$$

只 cache $c_t^{KV}$（极小），用时再投影回 K/V。

> 详见 §5.2 注意力变体。

---

## 五、KV-Cache 的内存布局：PagedAttention

核心比喻：**把操作系统"虚拟内存分页"的思路搬到 KV-Cache 上**。

### 5.1 传统连续布局的问题

朴素做法：为每个请求预先分一段**连续显存**，长度按"最坏情况"预留（如 max_seq_len = 4096）。

```
req1 (实际只生成 500 token):   [████░░░░░░░░░░░░░░░░]  浪费 87%
req2 (生成 3000 token):        [████████████████████]   占满
req3 (实际只生成 200 token):   [██░░░░░░░░░░░░░░░░░░]   浪费 95%
```

三种浪费叠加：
- **预留浪费**：还没生成的 token 也得提前占位
- **内部碎片**：预留长度 > 实际长度，剩余部分用不上
- **外部碎片**：请求结束释放出的"洞"难以拼成新请求需要的连续大块

实测显存有效利用率只有 **20-40%**。

### 5.2 PagedAttention 的核心思想

把 KV-cache 切成**固定大小的小块**（通常 16 token / block），按需分配，**不要求物理上连续**。

每个请求维护一张 **block_table**，记录"逻辑第 i 段 → 全局物理池里的哪一页"：

```
逻辑视角（每请求看到的是连续序列）：
  req1:  [token 0..15] [token 16..31] [token 32..47] ...
            ↓             ↓              ↓        ← block_table 做映射
          page 7        page 2         page 9

物理视角（GPU 全局共享池）：
  pages:  [0][1][2][3][4][5][6][7][8][9]...
              ↑           ↑    ↑
            其他请求      req1 req1
```

- 上层逻辑仍然顺序（attention 计算不变）
- 物理上 page 散落在共享池
- attention kernel 通过 block_table 完成 gather

### 5.3 两个直接收益

**(1) 显存利用率 90%+**

不预留，要一个 block 分一个。最坏内部碎片 = 1 个 block - 1 个 token（≈ 15 token）。

**(2) Prefix sharing（前缀共享）**

多个请求若共享同一段 prompt（如同一个 system prompt、few-shot 示例），它们的 block_table 可以**指向同一组物理 page**：

```
req1: [page 7][page 2][page 9]   ← system prompt 在 page 7,2
req2: [page 7][page 2][page 5]   ← 直接复用前两页
req3: [page 7][page 2][page 11]
```

- 物理上 page 7,2 只存一份
- 共享前缀的场景下 KV 占用显著下降
- 这就是 **prefix caching**（vLLM、SGLang、TensorRT-LLM 都内置）

### 5.4 工程影响

- batched throughput 提升 3-5×（vs 连续布局）
- 现代推理引擎（SGLang、TensorRT-LLM、TGI、MLC）几乎都实现了某种形式的分页

> 一句话：**逻辑连续、物理分页、按需分配、可跨请求共享前缀。**

### 5.5 vLLM —— 不只是 PagedAttention

**先澄清关系**：

| 名字 | 是什么 |
|------|--------|
| **PagedAttention** | 一项**算法/技术**：KV-Cache 的分页内存管理 |
| **vLLM** | 一个**开源推理引擎**（UC Berkeley，SOSP 2023），PagedAttention 是它的招牌技术之一 |

vLLM 的高吞吐**不是单靠 PagedAttention**，而是几个机制配合。其中和 PagedAttention 同等重要的是 **Continuous Batching**。

#### Continuous Batching（连续批处理 / in-flight batching）

传统 **static batching**：组好一个 batch 一起跑，等 batch 内**最长**的那条请求生成完，才能开始下一个 batch。

```
static batching (槽位是固定的)：
  step:    1  2  3  4  5  6  7  8  9  10
  req A:   ░  ░  ░  ░  ░  ░  ░  ░  ░  ░   (10 token)
  req B:   ░  ░  ░  ▓  ▓  ▓  ▓  ▓  ▓  ▓   (3 token 就结束, 后面空转 7 步)
  req C:   ░  ░  ░  ░  ░  ▓  ▓  ▓  ▓  ▓   (5 token 后空转)
  → GPU 在 ▓ 的位置全在空转
```

**continuous batching**：每个 decode step 都重新调度。**一旦某条请求生成完 EOS，立刻把它的槽位让给排队中的新请求**：

```
continuous batching：
  step:    1  2  3  4  5  6  7  8  9  10
  req A:   ░  ░  ░  ░  ░  ░  ░  ░  ░  ░
  req B:   ░  ░  ░  D  D  D  D  D  D  D   ← B 完成后立刻塞 D 进来
  req C:   ░  ░  ░  ░  ░  E  E  E  E  E   ← C 完成后立刻塞 E
  → GPU 几乎不空转
```

这才是让 throughput 真正起飞的东西。**PagedAttention 让 KV 可以被任意拼装，是 continuous batching 在显存层面的前提**（否则新请求来了没地方放 KV）。两者是配套的。

#### vLLM 还做了什么

| 模块 | 作用 |
|------|------|
| PagedAttention | KV 分页管理（§5.2） |
| Continuous batching | 每 step 重排，槽位不空转 |
| Prefix caching | 跨请求共享前缀 page（§5.3） |
| Scheduler | prefill / decode 优先级与抢占 |
| Quantization | AWQ / GPTQ / FP8 / KV INT8 |
| Speculative decoding | 小模型起草 + 大模型校验 |
| Multi-LoRA | 单引擎挂多个 LoRA 适配器 |
| TP / PP | 张量并行 / 流水并行 |
| OpenAI-compatible server | 直接当 `/v1/chat/completions` 用 |

#### 生态地位

- 2023 年发布后迅速成为**开源 LLM 推理的事实标准**
- 主要竞品：**SGLang**（更强的结构化前缀复用 + RadixAttention）、**TensorRT-LLM**（NVIDIA 官方，单卡极致性能）、**TGI**（HuggingFace）
- 大部分模型发布后 24h 内会有 vLLM 适配

> 一句话：**PagedAttention 解决"显存怎么放"，Continuous Batching 解决"调度怎么排"，两者合起来才是 vLLM 高吞吐的真正来源。**

---

## 六、KV-Cache 量化

KV 是显存大头，量化收益显著：
- **FP16 → INT8**：尺寸 ×0.5，几乎无损
- **FP16 → INT4 / FP4**：尺寸 ×0.25，需要 per-head / per-token scale
- **KV-cache offload**：把不活跃 cache 移到 CPU/SSD（DeepSpeed, FlexGen）

代表工作：
- **KIVI**（[Liu et al. 2024](https://arxiv.org/abs/2402.02750)）：K 按 per-channel、V 按 per-token 2-bit 量化
- **SmoothQuant** 思想可迁移到 KV
- **Kivi 2-bit / 4-bit** 已被 vLLM、SGLang 内置

---

## 七、KV-Cache 复用与丢弃

进阶话题（与第 6 章稀疏注意力交叉）：
- **Streaming LLM**（[Xiao et al. 2023](https://arxiv.org/abs/2309.17453)）：保留 attention sink（前几个 token）+ 最近窗口
- **H2O**（[Zhang et al. 2023](https://arxiv.org/abs/2306.14048)）：保留 high attention score 的 KV
- **SnapKV / PyramidKV**：分层不同压缩率
- **CSA / NSA / DSA**：基于稀疏 attention 的 KV 选择性使用（DeepSeek 系）

---

## 关键问答

**Q1**：KV-Cache 为什么不存 Q？
- 当前 step 只算当前 Q 一次，下一 step Q 重新算
- Q 不参与对历史的 attention 计算（历史 token 不再做 forward）
- 缓存 Q 没有收益

**Q2**：Encoder-decoder 的 cross-attn 也用 KV-cache 吗？
- 是，但 encoder K/V **全程不变**，prefill 阶段算一次后永久不动
- 与 decoder self-attn 的"增长式"cache 不同（§2.6）

**Q3**：为什么 KV-cache 是 decode 的瓶颈？
- decode 阶段每步只有 1 个 token 的计算，但要读取所有历史 KV
- arithmetic intensity = FLOPs / Bytes ≈ 1 → memory-bound（§4.4）
- GPU 算力 100x 大于带宽时，"算"不是瓶颈，"读 KV"才是

**Q4**：KV-Cache 量化为什么 K 与 V 要分开处理？
- K 进入 softmax，量化误差被指数放大 → 更敏感
- V 是线性求和，量化更宽容
- KIVI 论文实证：K per-channel + V per-token 最稳

**Q5**：长上下文（128k）下 KV-cache 显存压力多大？
- Llama-3 70B GQA + 128k context ≈ 42 GB（单条）
- 单卡 A100 80G 只能放 ~1 条
- → 必须靠 MLA / 稀疏 attention / KV 量化 / offload 缓解

**Q6**：vLLM 的 PagedAttention 解决了什么核心问题？
- 传统连续 KV cache 需要预留最大长度 → 浪费 + 限制 batch
- 分页后按需分配 + 跨请求共享前缀
- 工业上 batched throughput 翻 3-5×

---

## 参考资料

- [Pope et al. 2022 — Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [Shazeer 2019 — Fast Transformer Decoding (MQA)](https://arxiv.org/abs/1911.02150)
- [Ainslie et al. 2023 — GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2 技术报告 (MLA)](https://arxiv.org/abs/2405.04434)
- [Kwon et al. 2023 — vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Xiao et al. 2023 — Efficient Streaming LLMs with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [Liu et al. 2024 — KIVI: 2-bit KV Cache Quantization](https://arxiv.org/abs/2402.02750)
- [Zhang et al. 2023 — H2O: Heavy-Hitter Oracle for Efficient KV Cache](https://arxiv.org/abs/2306.14048)
- [Karpathy — let's reproduce GPT-2 (含 KV-cache 实现讲解)](https://www.youtube.com/watch?v=l8pRSuU81PU)
