# 2.3 Norm（LayerNorm / RMSNorm / Pre-Post Norm）

[← 返回框架](../../README.md) · [📎 materials.md → §2.3](../../materials.md)

---

## 一、为什么需要归一化

深层网络中，激活的分布会逐层漂移（covariate shift），导致：
- 梯度消失 / 爆炸
- 优化曲面变形 → 难以训练
- 训练对学习率敏感

归一化 = 在每层把激活拉回稳定分布（均值方差受控），让深层网络可训。

> Transformer 中**没有** BatchNorm，因为：
> - 序列长度变化 → BN 统计不稳定
> - 训练/推理 batch 维度差异大
> - LayerNorm 在 token 维度内做，与 batch 无关

---

## 二、LayerNorm（Ba et al. 2016）

对每个 token 的 d 维向量独立归一化：

$$y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中
$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

- $\gamma, \beta \in \mathbb{R}^d$：可学习缩放与偏置（**affine** 参数）
- $\epsilon$：数值稳定（一般 $10^{-5}$）
- 沿**最后一维**（feature 维）算

**计算量**：每 token 一次均值 + 方差 + 标准化 → $O(d)$。

---

## 三、RMSNorm（Zhang & Sennrich 2019）⭐ 现代主流

省略**均值中心化**，只用 RMS 缩放：

$$y = \gamma \odot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

- 没有 $\mu$，没有 $\beta$（一般也不要 affine 偏置）
- 计算量比 LayerNorm 少 **~30-50%**
- 实证：**效果几乎与 LN 持平**，但更快 / 更省

**采用者**：Llama 系列 / Qwen / Mistral / DeepSeek / Gemma / 几乎所有 2023 后的开源模型。

**为什么 RMS 够用？**
- 均值漂移在残差网络里实证不显著
- 缩放控制方差是主导因素
- 减一步运算在 LLM 这种深网络中累积可观

---

## 四、Pre-Norm vs Post-Norm ⭐ 训练稳定性的关键

### 4.1 Post-Norm（原版 Transformer）

```
x → Attn → +x → Norm → FFN → +x → Norm → out
```

公式：$x_{l+1} = \text{Norm}(x_l + \text{Sublayer}(x_l))$

- 原版 Attention Is All You Need 用的
- **训练不稳定**：深层网络梯度爆炸 / 消失
- 需要 **warmup** 和小心调 LR
- BERT、GPT-1 用

### 4.2 Pre-Norm（现代主流）

```
x → Norm → Attn → +x → Norm → FFN → +x → out  → final Norm
```

公式：$x_{l+1} = x_l + \text{Sublayer}(\text{Norm}(x_l))$

- 残差路径直接传，**梯度信号不被 Norm 衰减**
- 训练显著更稳定，可以堆很深
- 末尾通常加一个 **final RMSNorm**（Llama 风格）
- GPT-2 起所有现代 LLM 都用

### 4.3 为什么 Pre-Norm 更稳定？

数学上：
- Post-Norm 梯度逐层乘以一个 ≤1 的因子（Norm 的 Jacobian） → 衰减或爆炸
- Pre-Norm 残差是恒等通路，梯度可以无衰减传到任意层
- 但代价：**模型的有效深度变浅**（残差通路占主导）→ 大模型上略损精度

### 4.4 DeepNorm（折中方案）

Microsoft 2022 提出，Post-Norm 加缩放：
$$x_{l+1} = \text{LN}(\alpha \cdot x_l + \text{Sublayer}(x_l))$$
$\alpha = (2N)^{1/4}$（N 是层数）。
保留 Post-Norm 的精度优势，同时能训到 1000 层。

GLM-130B / DeepSeek-V1 中用过，最新模型基本回归 Pre-Norm + RMSNorm。

### 4.5 Sandwich Norm / QK-Norm

- **Sandwich**：在 sublayer 前后都加 Norm（PaLM、Gemma 用）
- **QK-Norm**：对 Q 和 K 单独 Norm（Olmo、部分长上下文模型用），避免 attention logits 爆炸

---

## 五、对比总表

| 方案 | 中心化 | 缩放 | Affine | 计算量 | 主流模型 |
|------|--------|------|--------|--------|----------|
| LayerNorm | ✅ | ✅ | $\gamma, \beta$ | 100% | BERT / GPT-2 / 早期 |
| RMSNorm | ❌ | ✅ | $\gamma$ | ~60-70% | Llama / Qwen / DeepSeek / 现代主流 |
| DeepNorm | LN + 放缩 | — | — | LN 同 | GLM-130B |
| QK-Norm | 对 Q/K 单独 norm | — | — | 少量额外 | Olmo, 部分长上下文 |

| 位置 | 稳定性 | 训练难度 | 主流采用 |
|------|--------|----------|----------|
| Post-Norm | 差 | 高 | 早期 (BERT / GPT-1) |
| **Pre-Norm** | 好 | 低 | **现代主流** |
| Sandwich | 中 | 中 | PaLM / Gemma |
| DeepNorm | 好 | 中 | GLM-130B |

---

## 六、混合精度下的实现细节

在 FP16 / BF16 训练时：
- Norm 内部统计计算（均值、方差、RMS）必须在 **FP32** 中做，否则数值不稳定
- 否则容易出现 NaN（特别是 FP16 的 overflow）
- BF16 因为指数位多更友好，但仍建议 stat 在 FP32

```python
# 伪代码（RMSNorm）
def rms_norm(x, gamma, eps=1e-6):
    # x: BF16
    x_fp32 = x.to(torch.float32)
    rms = (x_fp32.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x_fp32 * rms).to(x.dtype) * gamma
```

---

## 关键问答

**Q1**：为什么 Transformer 不用 BatchNorm？
- 序列内 token 不独立同分布，按 batch 维度做没有意义
- 序列长度可变 → 训练 / 推理统计分布漂移
- LN/RMS 沿 feature 维 → 与 batch 大小、序列长度都无关

**Q2**：RMSNorm 为何能省去均值？
- 残差网络中均值漂移并不严重（残差累积已经在做"加回去"）
- 实验证明去掉中心化对最终 loss 几乎无影响
- 节省一次 reduce 操作 + 简化反向

**Q3**：Pre-Norm 为什么训练更稳？
- 残差通路 $x_l \to x_{l+1}$ 是恒等映射，梯度信号无衰减
- Post-Norm 把 Norm 放在残差**之后**，每层都对总残差做缩放 → 梯度逐层衰减或爆炸
- 这是 GPT-2 之后所有大模型回归 Pre-Norm 的根本原因

**Q4**：Pre-Norm 有什么缺点？
- "有效深度"变浅：很多深层 sublayer 的贡献被残差稀释
- 同参数量下，Post-Norm 在能训出来的情况下精度略高
- 工业上为了可训性还是选 Pre-Norm

**Q5**：QK-Norm 是为了解决什么问题？
- 长上下文 / 大模型下 $QK^\top$ 容易出现极端值 → softmax 饱和
- 对 Q, K 单独 Norm 让 logits 分布更稳
- ViT-22B、Olmo、Gemma-2 都用了类似 trick

**Q6**：Norm 放在哪里影响 KV-cache 吗？
- 不影响 cache 内容（cache 的是 K、V 投影后的张量）
- 但实现上 Norm 必须在 RoPE 之前（避免破坏旋转结构）

---

## 参考资料

- [Ba et al. 2016 — Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Zhang & Sennrich 2019 — Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [Xiong et al. 2020 — On Layer Normalization in the Transformer Architecture (Pre vs Post)](https://arxiv.org/abs/2002.04745)
- [Wang et al. 2022 — DeepNet: Scaling Transformers to 1000 Layers](https://arxiv.org/abs/2203.00555)
- [Henry et al. 2020 — Query-Key Normalization](https://arxiv.org/abs/2010.04245)
- [Dehghani et al. 2023 — ViT-22B (QK-Norm 工业验证)](https://arxiv.org/abs/2302.05442)
- [Llama 2 paper (RMSNorm + Pre-Norm 工程实践)](https://arxiv.org/abs/2307.09288)
- [Jianlin Su 博客 — RMSNorm 解读](https://kexue.fm/archives/8620)
- [HuggingFace transformers — modeling_llama.py RMSNorm 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
