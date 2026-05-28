# 2.4 FFN（SwiGLU / GeGLU）

[← 返回框架](../../README.md) · [📎 materials.md → §2.4](../../materials.md)

---

## 一、FFN 在 Transformer 中的角色

每个 Transformer block：
```
x → Attn → +x → Norm
       ↓
      FFN → +x → Norm → next layer
```

**Attention 做"信息交互"，FFN 做"信息加工"。**

- Attention 让 token 之间交换信息（混合 sequence 维）
- FFN 是逐 token 的两层 MLP（混合 feature 维）
- **FFN 占模型大约 2/3 的参数与 FLOPs**（在 MHA + 4× FFN 配置下）

---

## 二、原版 FFN（Vaswani 2017）

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

- $W_1: d \to 4d$（升维）
- $W_2: 4d \to d$（降维）
- 中间宽度 $d_{ff} = 4d$ 是经验值

**直觉**：升维到一个更大的空间做 ReLU 选择性激活，再投影回去 → 等价于一个查表式的"知识存储"。

> Geva et al. 2021 [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) 论证了 FFN 起的就是 "associative memory" 作用。

---

## 三、激活函数的演化

| 激活 | 公式 | 优点 | 主流采用 |
|------|------|------|----------|
| ReLU | $\max(0, x)$ | 简单 | 原版 Transformer |
| GELU | $x \cdot \Phi(x)$ | 平滑、负区间有梯度 | BERT / GPT-2 / GPT-3 |
| Swish (SiLU) | $x \cdot \sigma(x)$ | 类似 GELU 但更便宜 | EfficientNet |
| **SwiGLU** | gated Swish | 经验上最好 | **Llama / Qwen / DeepSeek / 主流 LLM** |
| GeGLU | gated GELU | 类似 SwiGLU | T5-v1.1 / PaLM |
| ReGLU | gated ReLU | 简单门控 | — |

---

## 四、GLU 家族（核心创新）

**GLU = Gated Linear Unit**（Dauphin 2016），把 FFN 拆成"主路径 × 门控路径"。

### 4.1 标准 GLU 公式

$$\text{GLU}(x) = (W_1 x) \odot \sigma(W_2 x)$$
- $W_1$ 主路径（线性）
- $W_2$ 门控路径（sigmoid 决定开关）

### 4.2 SwiGLU（Llama 系采用）

把 sigmoid 换成 **Swish**（= SiLU）：
$$\text{SwiGLU}(x) = \text{Swish}(W_{gate} x) \odot (W_{up} x)$$
$$\text{FFN}_{\text{SwiGLU}}(x) = W_{down} \cdot \text{SwiGLU}(x)$$

由 Noam Shazeer 在 [GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202) 提出。
Llama-1 大规模验证 → 之后几乎所有开源 LLM 都跟随。

### 4.3 完整结构（Llama / Qwen / DeepSeek）

```
        x  (d_model)
         │
  ┌──────┴──────┐
  ▼             ▼
 W_gate        W_up        (各自 d_model → d_ff)
  │             │
 SiLU           │
  │             │
  └──── × ──────┘          (element-wise 乘)
         │
       W_down              (d_ff → d_model)
         │
        out
```

注意：相比原 FFN（2 个矩阵），**SwiGLU FFN 有 3 个矩阵**。

### 4.4 维度调整

参数量与原 FFN 持平时：
- 原 FFN：$d_{ff} = 4d$，参数 $= 2 \cdot d \cdot 4d = 8d^2$
- SwiGLU：3 个矩阵，要持平 → $d_{ff} = \frac{8d}{3} \approx 2.67d$
- Llama 实际取 **$d_{ff} \approx \frac{8d}{3}$ 向上取整到 256/128 的倍数**

例如 Llama-3 8B：$d = 4096, d_{ff} = 14336 \approx 3.5d$（略大于 8/3）。

### 4.5 为什么 SwiGLU 更好

- **门控**让网络学会"选择"哪部分激活通过 → 表达能力更强
- Swish 在负区间仍有梯度（不像 ReLU 死神经元）
- 实证：相同 FLOPs 下 SwiGLU 的 perplexity 比 GELU 低 0.5-1%
- Shazeer 原文："We offer no explanation as to why... divine benevolence."

---

## 五、FFN 的工程细节

### 5.1 参数 / FLOPs 占比

对 dense LLM（MHA 配置）：
- Attention 参数（Q/K/V/O 投影）≈ $4d^2$
- FFN 参数 ≈ $8d^2$（标准）或 $\frac{8 \cdot 3}{2} d^2 = 12d^2$（SwiGLU 持平时）
- **FFN ≈ 2/3 总参数**

每 token FLOPs 同分布。这就是为什么：
- 量化 / 稀疏化优化首要瞄准 FFN
- MoE 把 FFN 改成专家集合（§9）能省 4-8 倍激活参数

### 5.2 显存

FFN 中间激活 $h = d_{ff} \cdot L$，比 attention 的 $L^2$ 在短序列下更耗显存。
- 这是 activation checkpointing 必砍的对象
- FlashAttention 解决的是 attention 的 $L^2$ 显存，**对 FFN 无效**

### 5.3 与量化的关系

- AWQ / GPTQ 对 FFN 的 W_gate / W_up / W_down 都量化
- W_down 的输入是激活，输出是残差路径 → 量化敏感
- SmoothQuant 通过把激活 outlier 转移到权重缓解

---

## 六、MoE 视角的 FFN

把单个 FFN 换成"N 个 FFN + 路由"：
$$\text{MoE-FFN}(x) = \sum_{i \in \text{Top-K}} g_i(x) \cdot \text{FFN}_i(x)$$

- 训练参数 N×（如 256 个专家）
- 推理只激活 K 个（如 8 个）→ 激活参数 = K × FFN 参数
- DeepSeek-V3 的 256 专家 / 激活 8 即此模式（§9 详）

---

## 关键问答

**Q1**：FFN 的中间维度为什么是 4d？
- 经验值，Vaswani 2017 直接用的
- 一些研究表明 2d - 8d 都可工作，4d 是精度/计算的折中
- SwiGLU 因为 3 矩阵，常取 2.67d 持平参数

**Q2**：SwiGLU 为什么效果更好（直观解释）？
- 门控提供 multiplicative interaction（乘法非线性比加法非线性表达更强）
- Swish 在负区间不像 ReLU 死，比 GELU 略便宜
- 实证：在 reasoning / math benchmark 上提升明显

**Q3**：FFN 占总 FLOPs 多少？
- dense MHA: FFN ≈ 2/3 总参数和 FLOPs
- GQA: FFN 占比更高（因为 attention 投影变少）
- MoE: 激活 FFN 减少 → attention 占比反而上升

**Q4**：FFN 与"知识存储"的关系？
- Geva et al.（2021）解释 FFN 是 key-value memory：W_1 是 key，W_2 是 value
- 编辑 FFN 权重可以"修改"模型已知的事实（ROME / MEMIT 方法）
- 这是为什么 MoE 用专家 FFN 能水平扩展知识

**Q5**：为什么 SwiGLU 有 3 个矩阵而原 FFN 只有 2 个？
- GLU 需要"主线性路径"和"门控路径"两条独立投影
- 加上输出投影一共 3 个
- 为持平参数，d_ff 缩小到 8d/3

**Q6**：能不能只压 FFN 中间维度？
- 可以，"narrow FFN" 模型（如 GPTQ 等量化方案、低秩 FFN）
- DeepSeek-V3 用细粒度专家把"小 FFN × 多个"实现 sparse compression
- 极端情况下 FFN 可以用低秩分解（SVD-LoRA）替代

---

## 参考资料

- [Vaswani et al. 2017 — Attention Is All You Need (原版 FFN)](https://arxiv.org/abs/1706.03762)
- [Dauphin et al. 2016 — Language Modeling with Gated Convolutional Networks (GLU)](https://arxiv.org/abs/1612.08083)
- [Shazeer 2020 — GLU Variants Improve Transformer (SwiGLU / GeGLU)](https://arxiv.org/abs/2002.05202)
- [Hendrycks & Gimpel 2016 — Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415)
- [Ramachandran et al. 2017 — Searching for Activation Functions (Swish)](https://arxiv.org/abs/1710.05941)
- [Geva et al. 2021 — Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [Meng et al. 2022 — Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262)
- [Llama 2 paper (SwiGLU 工程实践)](https://arxiv.org/abs/2307.09288)
- [Mistral 7B paper](https://arxiv.org/abs/2310.06825)
- [Jianlin Su 博客 — Gated Linear Units 解读](https://kexue.fm/archives/9812)
