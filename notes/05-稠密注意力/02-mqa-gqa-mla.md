# 5.2 MQA / GQA / MLA —— KV head 怎么压

[← 返回框架](../../README.md) · [📎 materials.md → §5.2](../../materials.md)

---

## 〇、为什么要压 KV head

§4.1 的核心结论：KV-cache 显存 = $2 \cdot L \cdot d \cdot \text{layers} \cdot B \cdot \text{bytes}$。

里面的 $d = h \cdot d_h$（head 数 × head 维）。Query 的 head 数必须够多（多视角才有效），但 **KV head 数完全可以独立减少**——这是 MQA/GQA/MLA 这一条线的核心思想。

```
       Q heads             KV heads (压这里)
MHA:   h  ↔  h            完全对应，KV 显存最大
MQA:   h  ↔  1            所有 Q 共享 1 套 KV，KV 显存 1/h
GQA:   h  ↔  h/g          分 g 组，每组共享，平衡点
MLA:   h  ↔  低秩 latent   不存 KV，只存压缩向量
```

---

## 一、MHA（baseline）

标准多头 attention（[Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)）。

```
    Q (h heads)        K (h heads)        V (h heads)
       │                  │                  │
       ├──────────────────┤                  │
       │  attention(Q_i, K_i, V_i) for each head i
       │                                     │
       └──────────────► concat → W_O → output
```

- Q、K、V 各有 $h$ 个 head，每 head 独立计算 attention
- KV-cache 大小 = $2 \cdot L \cdot h \cdot d_h$ = $2Ld$ per layer
- 表达能力最强，但 KV 显存最大

---

## 二、MQA — Multi-Query Attention（[Shazeer 2019](https://arxiv.org/abs/1911.02150)）

**激进做法**：所有 query head 共享**同一组** K, V。

```
    Q (h heads)        K (1 head)        V (1 head)
       │                  │                  │
       ├──── 全部 h 个 Q 都和这一组 K,V 算 attention
       │
       └──► concat → W_O
```

收益：
- KV-cache 缩到 $\frac{1}{h}$（如 h=64 → 64×）
- decode 阶段读 KV bytes 大幅下降
- arithmetic intensity ↑

代价：
- 表达能力下降，**长文本和 reasoning 上明显掉点**
- PaLM 用过，但后期模型很少纯 MQA

---

## 三、GQA — Grouped-Query Attention（[Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)）

**MHA 和 MQA 的中间方案**：$h$ 个 Q head 分成 $g$ 组，每组共享一组 K, V。

```
    Q (h heads)               K (h/g heads)       V (h/g heads)
       │                            │                  │
   [Q1,Q2,...Q_{h/g}] → 组 1 → 共享 (K_1, V_1)
   [Q_{h/g+1},...]    → 组 2 → 共享 (K_2, V_2)
   ...
   (共 g 组)
```

公式：
$$\text{KV-cache} = 2 \cdot L \cdot \frac{d}{g} \cdot \text{layers} \cdot B \cdot \text{bytes}$$

典型配置（Llama-2 70B / Llama-3 系列）：
- Q heads = 64
- KV heads = 8（**g = 8**）
- KV-cache 缩到 **1/8**
- 性能几乎无损（MMLU 等评测仅低 0.2-0.5 分）

收益总结：
- 实现简单（在 MHA kernel 上微改）
- 训练时直接配 GQA，不用从 MHA 蒸馏
- 当前几乎所有开源大模型（Llama-2/3, Mistral, Qwen2/3, Yi, GLM-4）都用 GQA

> **GQA 是 2023-2024 年最成功的工程权衡**：实现复杂度 ≈ MHA，KV-cache ≈ MQA，质量 ≈ MHA。

---

## 四、MLA — Multi-head Latent Attention（[DeepSeek-V2/V3](https://arxiv.org/abs/2405.04434)）

DeepSeek 的另辟蹊径：不是"减 head"，而是**用低秩压缩存 KV**。

### 4.1 核心思想

把 K、V 的"通过权重得到"过程改成：
1. 先把 hidden state $h_t$ 投到一个**小的 latent**向量 $c_t$
2. 实际 cache 的是 $c_t$（极小）
3. 用时再从 $c_t$ 解压出 K、V

```
hidden state h_t ────W^DKV───→  c_t (低秩 latent, dim≈512)   ←─ 这个才进 KV-cache
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
                  W^UK · c_t = K_t       W^UV · c_t = V_t      ←─ 用时再展开
```

公式：
$$c_t^{KV} = W^{DKV} h_t \quad (\dim c \ll d)$$
$$K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV}$$

### 4.2 RoPE 的特殊处理

K 进 attention 前要乘 RoPE 旋转矩阵，而 RoPE **不与低秩可交换**（旋转不能与上投影合并）。MLA 的解法：**K 拆成两部分**：
- 大部分维度走 low-rank 路径（不带 RoPE）
- 一小部分维度（"rope head"）保留单独的 K，附 RoPE

V 没有 RoPE 问题，纯 low-rank。

### 4.3 推理时的进一步技巧：吸收上投影

decode 阶段不需要显式展开 $K_t$。可以把 $W^{UK}$ "吸收" 到 $W^Q$ 里：
$$Q^\top K = (W^Q h)^\top (W^{UK} c) = h^\top (W^Q)^\top W^{UK} c$$

让 $W^{UK}$ 在编译期与 $W^Q$ 融合 → KV-cache 实际只读 $c$，不需要再做上投影。

### 4.4 性能数字

| 模型 | KV-cache / token | 相比 MHA |
|------|------------------|---------|
| MHA (Llama-3 70B, h=64) | 2.6 MB | 1× |
| GQA (Llama-3 70B, g=8) | 0.33 MB | 1/8 |
| **MLA (DeepSeek-V2 236B)** | **0.07 MB** | **1/40** ⭐ |

→ 128k context 下：MHA 需要 ~340 GB，MLA 只需 ~9 GB。

### 4.5 训练代价

- 参数多一组 $W^{DKV}, W^{UK}, W^{UV}$（但都很小）
- 实现复杂度比 GQA 高（需要专门的 MLA kernel）
- 训练 loss 与 MHA 接近，DeepSeek 实测略好（更强的正则）

---

## 五、四者对比

| 维度 | MHA | MQA | GQA | MLA |
|------|------|-----|-----|-----|
| KV head 数 | h | 1 | h/g | 低秩 latent |
| KV-cache 占用 | $2Ld$ | $2Ld/h$ | $2Ld/g$ | $\sim Ld_c$（$d_c\ll d$）|
| 质量 vs MHA | = | ↓↓ | ≈（-0.3 MMLU） | =/↑ |
| 实现复杂度 | 低 | 低 | 低 | **高** |
| 训练成本 | baseline | 略低 | 略低 | 略高（多套权重） |
| 代表模型 | GPT-2/3, Llama-1 | PaLM, Falcon-40B | Llama-2/3, Mistral, Qwen2/3 | DeepSeek-V2/V3 |

---

## 六、关于"为什么 KV head 可以这么压"的直觉

为什么 Q head 不能压、KV head 却可以压？

- Q 是"问题"，需要多视角发问 → 多 head 必要
- K, V 是"被问到的信息"，多 Q head 可以问同一份信息的不同方面 → KV 可以共享
- 实证：把 Q head 缩成 1 个会大幅掉点，但 KV head 缩到 1/8 几乎无损
- 这反映了 attention 的非对称性——Q 是"查询"，KV 是"知识存储"

---

## 七、对推理引擎的影响

- vLLM、SGLang、TensorRT-LLM 都内置了 GQA kernel（实质是 MHA kernel 的 grouped 版本）
- MLA 需要专门 kernel：DeepSeek 官方 FlashMLA、SGLang 也支持
- GQA + PagedAttention + FlashAttention 是当前推理引擎的"标配三件套"

---

## 关键问答

**Q1**：为什么 GQA 比 MQA 普及？
- MQA KV/8 倍压缩 vs GQA 通常 8 组 → KV-cache 一样小
- 但 GQA 几乎无损，MQA 在 reasoning 上明显掉点
- 工程上"几乎免费的优化"远比"激进但有副作用"受欢迎

**Q2**：把 MHA 模型蒸馏成 GQA / MQA 怎么做？
- Llama-2 报告里有：把 KV head 平均合并（取均值或线性合并）
- 然后用少量 token 继续训练（< 5% 原训练量）
- 通常恢复到 -0.5 MMLU 以内
- DeepSeek 也用过类似的 MLA-to-GQA / GQA-to-MLA 蒸馏

**Q3**：MLA 训练时如何避免 RoPE 不兼容？
- K 解耦为 (K_nope + K_rope) 两路
- K_nope 走低秩：$K_{nope} = W^{UK} c$
- K_rope 是独立的 head，保留 RoPE
- Attention 时拼接：$K = [K_{nope} | K_{rope}]$

**Q4**：MLA 推理时为什么"几乎不用展开 K"？
- $Q^T K$ 中可以把 $W^{UK}$ 提到 $W^Q$ 那侧
- 等价于读取 $c$（小）而不是 $K$（大）
- 这一步在 DeepSeek-V2 paper 中是 "absorbing the up-projection matrix"

**Q5**：长上下文（128k+）下哪个更优？
- MLA ⭐⭐⭐：KV/40 倍压缩，是 DeepSeek 能做 128k 的关键
- GQA ⭐⭐：还要靠量化（INT8 KV）才能撑 128k
- MQA ⭐：可以，但质量掉
- MHA：基本不可行（KV 显存爆炸）

**Q6**：训练时 GQA 比 MHA 快吗？
- forward 几乎一样快（KV head 少不显著影响 FLOPs，因为 Q 还是 h 个）
- 训练显存略省（KV bytes 在激活里也省了）
- 主要收益在推理 decode

**Q7**：MLA 为什么也能提升训练质量？
- 低秩压缩起了正则效应（类似 LoRA 在 fine-tune 时的效果）
- $c$ 维度选择得当时，能强迫模型学到更紧凑的 KV 表示
- DeepSeek-V2 论文实测：MLA 略胜 MHA

---

## 参考资料

- [Vaswani et al. 2017 — Attention Is All You Need (MHA)](https://arxiv.org/abs/1706.03762)
- [Shazeer 2019 — Fast Transformer Decoding (MQA)](https://arxiv.org/abs/1911.02150) ⭐
- [Ainslie et al. 2023 — GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) ⭐
- [DeepSeek-V2 技术报告 (MLA 首次提出)](https://arxiv.org/abs/2405.04434) ⭐
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [Llama-2 技术报告（GQA 设计原因）](https://arxiv.org/abs/2307.09288)
- [Llama-3 技术报告](https://arxiv.org/abs/2407.21783)
- [FlashMLA GitHub (DeepSeek)](https://github.com/deepseek-ai/FlashMLA)
- [Sebastian Raschka — Multi-Head/Multi-Query/Grouped-Query Attention](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)
