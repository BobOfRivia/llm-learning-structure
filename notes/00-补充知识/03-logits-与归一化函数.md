# 补充 3：logits 与归一化函数（softmax / sigmoid）

[← 返回框架](../../README.md)

> 这是一篇**跨章节复用**的工具笔记。"logits" 这个词在几乎每一章都出现：
> - §2 分类头 / LM head 的输出
> - §5 注意力分数 $QK^\top/\sqrt{d}$
> - §9.1 MoE router 的 $W_g x$
> - §11 RLHF 的 reward 模型输出
> - §12 采样（top-k / top-p / 温度）操作的对象
>
> 集中讲清楚一次：什么是 logits、为什么 DL 里偏爱在 logits 上操作而不是概率上。

---

## 一、严格定义（统计学起源）

"logit" 来自 **logistic function（sigmoid）的反函数**：

$$\text{logit}(p) = \log \frac{p}{1-p}, \quad p \in (0, 1)$$

这叫**对数几率（log-odds）**——把概率 $p \in (0,1)$ 映射回 $\mathbb{R}$。

它和 sigmoid 是互逆的：

$$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \xleftrightarrow{\text{互逆}} \quad \text{logit}(p) = \log \frac{p}{1-p}$$

所以：**如果 $\sigma(z) = p$，则 $z = \text{logit}(p)$。** —— $z$ 就是概率 $p$ 的 "logit 表示"。

> 这个名字最早来自二分类逻辑回归（logistic regression）。在多分类（softmax）场景里被泛化保留了下来。

---

## 二、深度学习的宽口径用法

现代 DL 里 "logits" 不再严格指 log-odds，而是泛化为：

> **logits = "网络最后一个线性层的原始输出，还没经过 softmax/sigmoid 这种归一化的值"**

性质：

| 性质 | 说明 |
|---|---|
| 取值范围 | $\mathbb{R}$（可正可负，无上下界） |
| 单位 | 没有概率含义，**只有相对大小有意义** |
| 经过 softmax → | 概率分布（和为 1） |
| 经过 sigmoid → | 独立的 [0, 1] 概率 |
| 差值更有意义 | softmax 对 logits **加常数不变**（$\text{softmax}(z) = \text{softmax}(z + c)$）—— "绝对值"不重要，"差值"重要 |
| 缩放有意义 | $\text{softmax}(z/T)$ 改变分布尖锐度（温度采样） |

---

## 三、三阶段术语：logits → score → weight

在 MoE / softmax 分类 / attention 这些场景里，从"原始输出"到"最终使用值"通常经过三步，对应三个名字：

```
W · x  ──→  z  ──→  s  ──→  w
linear   logits   score   weight
        (R)    (归一化)  (聚合用)

  raw ──→ softmax/sigmoid ──→ TopK + normalize（如有）
```

| 阶段 | 名字 | 范围 | 谁产生 | 谁用 |
|---|---|---|---|---|
| ① | **logits** $z$ | $\mathbb{R}$ | 线性层 $W x$ | sampling、loss（log-sum-exp）、mask（设 -∞） |
| ② | **scores / probs** $s$ | $[0,1]$ | softmax 或 sigmoid | 排序、TopK、判别 |
| ③ | **weights** $w$ | 通常 $\sum w_i = 1$ | TopK + 归一化 | 加权求和（attention、MoE 输出） |

> ⚠️ 这三个名字在不同章节略有混用：
> - 分类任务里只有 ①logits 和 ②probs 两个阶段
> - MoE 路由里三个阶段都明确（§9.1）
> - Attention 里 $QK^\top/\sqrt{d}$ 是 logits、$\text{softmax}(\cdot)$ 是 attention weights（直接当 weight 用，无 TopK）

---

## 四、三个常见场景

### 4.1 分类任务

```
[batch, d] ──W_cls──→ [batch, C]   ← logits
                          │
                          ▼ softmax
                      [batch, C]   ← 概率分布
                          │
                          ▼ cross-entropy with label
                       loss
```

PyTorch 里 `F.cross_entropy(logits, label)` 直接吃 logits，内部用 log-sum-exp 算 log-softmax，比"先 softmax 再 log"数值稳定得多。

### 4.2 语言模型（LM head）

```
hidden state [B, T, d] ──W_emb^T──→ [B, T, V]   ← logits（V = vocab size）
                                       │
                                       ▼ softmax
                                   [B, T, V]   ← 每个位置的下一 token 概率
```

HuggingFace `model(input_ids).logits` 拿到的就是 `[B, T, V]` 这个张量。采样、top-k、top-p、温度调节、repetition penalty 都在 logits 上做。

### 4.3 MoE Router

```
x [d] ──W_g──→ z [N]              ← logits
                  │
                  ▼ softmax 或 sigmoid
              s [N]                ← score
                  │
                  ▼ TopK + normalize
              w [N]                ← weight
                  │
                  ▼ Σ w_i · E_i(x)
              y [d]                ← MoE 输出
```

→ §9.1 §4.1 里写的 $s_i = \text{score}(W_g x)$，**$W_g x$ 就是 logits**。DSv3 §5.4 的 bias $b_i$ 也是直接加在 logits 上：$\tilde{z}_i = z_i + b_i$。

### 4.4 Attention

$$\text{attn}(Q,K,V) = \text{softmax}\Big(\underbrace{\frac{QK^\top}{\sqrt{d}}}_{\text{attention logits}}\Big) V$$

Causal mask 通过把被屏蔽位置的 **logits 设为 $-\infty$** 实现——softmax 后这些位置自动变 0。这是"在 logits 上加 mask"的经典用法。

---

## 五、为什么 DL 里偏爱在 logits 上操作

### 5.1 数值稳定性（最重要的原因）

概率经过 $\log$ 会出 $\log 0 = -\infty$、$\log(\text{tiny}) = \text{underflow}$。logits 在 $\mathbb{R}$ 上活动没有边界问题。

cross-entropy 的标准实现：

$$L = -\log \frac{e^{z_y}}{\sum_j e^{z_j}} = -z_y + \underbrace{\log \sum_j e^{z_j}}_{\text{log-sum-exp}}$$

用 **log-sum-exp trick** 防溢出：

$$\log \sum_j e^{z_j} = z_{\max} + \log \sum_j e^{z_j - z_{\max}}$$

→ 所以 PyTorch / JAX 等框架几乎所有"分类相关的 loss"都接受 logits 而不是 probs 作为输入：

| API | 输入 |
|---|---|
| `F.cross_entropy(input, target)` | input 是 **logits** |
| `F.binary_cross_entropy_with_logits` | input 是 **logits** |
| `F.nll_loss(input, target)` | input 是 **log_softmax 后**（即 log-probs） |
| `F.binary_cross_entropy` | input 是 **probs**（不推荐，容易数值出问题） |

### 5.2 温度采样

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

温度 $T$ 直接除在 logits 上：
- $T \to 0$：分布尖锐（趋向贪心 argmax）
- $T = 1$：原分布
- $T \to \infty$：分布均匀（随机）

→ 温度操作在 logits 上是"线性缩放"，干净简洁。在 probs 上没有这种形式。

### 5.3 Top-k / Top-p 采样

排序、截断都在 logits 上做。softmax 是单调函数，**排 logits 的顺序 = 排 probs 的顺序**，但 logits 上不会下溢。

```python
# 标准实现思路
logits = model(x).logits[:, -1, :]   # [B, V]
logits = logits / temperature         # 温度
logits[logits < topk_threshold] = -inf  # top-k 截断（设 -∞）
probs = logits.softmax(-1)            # 最后再 softmax
next_token = probs.multinomial(1)
```

设 $-\infty$ 这个 trick 反复使用：**只要想"屏蔽某些选项"，把它们 logits 设 $-\infty$，softmax 后自动是 0**，不会扭曲其他位置的相对概率。

### 5.4 加 bias / mask（硬调整）

- **DSv3 aux-loss-free bias**：$\tilde{z}_i = z_i + b_i$，直接加在 logits 上
- **Attention mask**：被屏蔽位置 logits 设 $-\infty$
- **Repetition penalty**：已生成 token 的 logits 减一个 penalty
- **Logit bias**（OpenAI API）：用户可指定特定 token 的 logits 偏移

这些"硬调整"都依赖 logits 在 $\mathbb{R}$ 上的自由度。在 probs 上做这些会破坏归一化（和不为 1）。

---

## 六、容易混淆的术语对照

| 术语 | 含义 | 范围 |
|---|---|---|
| **logits** | softmax/sigmoid **前**的原始线性输出 | $\mathbb{R}$ |
| **scores** | logits 经过 softmax/sigmoid 后的分数（MoE / 检索语境） | $[0, 1]$ 或 $\mathbb{R}$（点积分） |
| **probabilities (probs)** | 经过 softmax 的概率分布 | $[0, 1]$，和为 1 |
| **log-probabilities (log-probs)** | $\log p$，常用于 loss 与 KL | $(-\infty, 0]$ |
| **odds** | $p / (1-p)$ | $[0, +\infty)$ |
| **log-odds = logit** | $\log(p/(1-p))$，**这是 logit 的严格定义** | $\mathbb{R}$ |
| **energy** | EBM / 对比学习里的"负 logit" | $\mathbb{R}$ |

> 在检索 / contrastive 场景里，"score" 经常指未归一化的相似度（如 $q \cdot k$），更像 logits；在 MoE 里 "score" 通常指归一化后的 $s_i$。**具体看上下文**。

---

## 七、一句话记忆

> **logits = "未归一化的原始分数"**。它是网络的直接输出、$\mathbb{R}$ 上的自由变量、损失计算和采样操作的标准入口。后面跟什么（softmax / sigmoid / argmax / topk）决定它被解读成什么（概率 / 独立概率 / 离散选择 / 排序）。

在 MoE router 里：$W_g x$ 是 logits → softmax/sigmoid 把它变成 score → TopK + normalize 把它变成 weight。三个名字对应三个阶段。

---

## 八、本文与各章的连接

- **§2.1 self-attention**：$QK^\top/\sqrt{d}$ 是 attention logits；causal mask = 把 logits 设 $-\infty$
- **§9.1 MoE router**：$W_g x$ 是 logits；§4.1 score 函数把 logits 转 score；§5.4 DSv3 bias 直接加在 logits 上
- **§11 RLHF / DPO**：reward model 输出 logits；策略 ratio $\pi_\theta(a|s)/\pi_\text{ref}(a|s)$ 在 log-space（log-probs）算
- **§12 推理优化**：top-k / top-p / 温度 / repetition penalty 全部在 logits 上做；speculative decoding 比较 logits 而非 probs
- **§13 评测**：PPL 来自 LM logits 的 cross-entropy

---

## 参考资料

- [Wikipedia — Logit](https://en.wikipedia.org/wiki/Logit)（严格定义和统计学背景）
- [PyTorch docs — CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)（明确说明输入是 raw logits）
- [HuggingFace — Generation utilities](https://huggingface.co/docs/transformers/main_classes/text_generation)（看 logits 怎么被 temperature / top_k / top_p 处理）
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)（为什么 logits 上算 loss 数值稳定）
