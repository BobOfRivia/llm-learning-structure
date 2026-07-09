# 2.1 Self-Attention

[← 返回框架](../../README.md) · [📎 materials.md → §2.1](../../materials.md)

---

## 一、核心公式（Scaled Dot-Product Attention）

输入序列 $X \in \mathbb{R}^{L \times d_{model}}$，三个线性投影：
$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V \quad W_* \in \mathbb{R}^{d_{model} \times d_k}$$

注意力输出：
$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

- $QK^\top \in \mathbb{R}^{L \times L}$：每一行是某 query 与所有 key 的相似度
- $M$：mask（causal LM 中 = 下三角的负无穷上方）
- $\sqrt{d_k}$：缩放因子，防止 softmax 进入饱和区

---

## 二、为什么除以 $\sqrt{d_k}$

如果 q、k 各分量独立同分布、方差为 1，则
$$\text{Var}(q^\top k) = d_k$$
点积量级随 $d_k$ 增大 → softmax 输入分布过陡 → 梯度趋零（"硬"分布）。
除以 $\sqrt{d_k}$ 让方差归一到 ~1。

> 论文原话："We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."

---

## 三、Multi-Head Attention（MHA）

$d_{model}$ 切成 $h$ 份，每份 $d_k = d_{model} / h$，独立做 attention 后拼接：

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$

每个 head 学不同的"关注模式"（语法/语义/位置等），等价于在不同子空间做投影。

```
                X (L × d_model)
                 │
         ┌──Q,K,V 投影──┐
         │              │
   ┌─────┼────┐    切成 h 份
   │     │    │
  Q₁    Q₂   …  Qₕ      (each L × d_k)
   │     │    │
  ↓attn↓attn↓attn      并行
   │     │    │
   └──concat──┘
         │
        W_O (混合 head)
         │
        output (L × d_model)
```

**FLOPs 与显存**：
- 注意力矩阵 $L \times L$（量级 $O(L^2)$，长上下文瓶颈）
- 总 FLOPs：$O(L^2 d + L d^2)$（前项是 attn，后项是 QKV/O 投影）

---

## 四、Causal Mask（自回归）

decoder-only LLM 必须屏蔽未来位置：
$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

实现上：上三角填 $-\infty$（或一个很大的负值），softmax 后变 0。

> FlashAttention 2 之后由 kernel 内部处理，不实例化 $L \times L$ 的 mask 矩阵。

---

## 五、与 RNN / CNN 的对比

| 维度 | RNN | CNN | Self-Attention |
|------|-----|-----|----------------|
| 并行性 | 串行 | 并行 | 并行 |
| 长程依赖 | 难（梯度消失） | 受感受野限制 | $O(1)$ 跳数 |
| 每层复杂度 | $O(Ld^2)$ | $O(kLd^2)$ | $O(L^2 d + Ld^2)$ |
| 归纳偏置 | 时间局部性 | 平移不变 | 几乎无（需位置编码） |

**核心优势**：任意两个位置间一步可达 → 长程依赖学习容易。
**核心代价**：$O(L^2)$ 复杂度 → 长上下文昂贵（驱动了 §5–§7 所有优化）。

---

## 六、计算复杂度细分

设 $L$ = 序列长度，$d$ = $d_{model}$，$h$ = head 数，$d_k = d/h$。下表先给单层 MHA 的总账，再逐步拆。

| 步骤 | 形状 | FLOPs | 激活显存 |
|------|------|-------|----------|
| Q,K,V 投影 | $(L,d)\cdot(d,d)\times 3$ | $6Ld^2$ | $3Ld$ |
| $QK^\top$ | $(L,d_k)\cdot(d_k,L)\times h$ | $2L^2 d$ | $hL^2$（attn logits） |
| Softmax | $(L,L)\times h$ | $\sim 5hL^2$（exp/和/除） | $hL^2$（attn 权重） |
| Attn $\cdot V$ | $(L,L)\cdot(L,d_k)\times h$ | $2L^2 d$ | $Ld$ |
| 输出投影 $W_O$ | $(L,d)\cdot(d,d)$ | $2Ld^2$ | $Ld$ |
| **总计** | — | $\boxed{8Ld^2 + 4L^2 d}$ | $O(hL^2 + Ld)$ |

> 因子 2：一次 $(m,k)\cdot(k,n)$ 矩阵乘法 = $2mnk$ FLOPs（每个输出元素一次乘 + 一次加）。前面表格里若按"乘加合一"算 1 FLOP，则去掉这个 2，得 $4Ld^2 + 2L^2 d$——两种写法在文献中都常见，差一个常数因子。

### 6.1 逐项推导

**① Q,K,V 投影**　$Q=XW_Q$，$X\in\mathbb{R}^{L\times d}$，$W_Q\in\mathbb{R}^{d\times d}$
- 单次矩阵乘 FLOPs = $2Ld^2$，三个投影 → $6Ld^2$
- 输出 $Q,K,V\in\mathbb{R}^{L\times d}$，激活显存 $3Ld$
- **性质**：标准 GEMM，compute-bound，GPU 利用率高（>70% TFLOPs）

**② 注意力打分 $S = QK^\top / \sqrt{d_k}$**
- 每个 head：$(L,d_k)\cdot(d_k,L)$ → $2L^2 d_k$ FLOPs；$h$ 个 head 合起来 $2L^2 \cdot h d_k = 2L^2 d$
- 显存上 $S\in\mathbb{R}^{h\times L\times L}$，是**长上下文显存爆炸**的根源（$L=8\text{k}, h=32$ 时单层就 8GB FP16）
- **性质**：$L\ll d$ 时被投影 dominate；$L\gg d$ 时这一步主导

**③ Softmax**
- 每行需要 max（数值稳定）、exp、求和、除——约 $5$ 次操作/元素，总计 $\sim 5hL^2$
- FLOPs 量级小（无 $d$），但 **memory-bound**：读写 $hL^2$ 的 logits/概率矩阵
- 这就是 FlashAttention 要 fuse softmax 进 attention kernel 的根本动机（§5.1）

**④ 加权求和 $O = \text{softmax}(S)\cdot V$**
- 每个 head：$(L,L)\cdot(L,d_k)$ → $2L^2 d_k$；总计 $2L^2 d$（与 ② 同量级）
- 输出 $L\times d$，激活显存 $Ld$
- **性质**：与 ② 对称，同样在长 $L$ 时主导

**⑤ 输出投影 $W_O\in\mathbb{R}^{d\times d}$**
- FLOPs = $2Ld^2$，与单个投影同
- 作用：把 $h$ 个 head 拼出的 $d$ 维向量"混合"——没有 $W_O$ 则各 head 输出仅是相加，head 间无交互

### 6.2 临界点 & 直觉

把投影项与 attention 项相比：
$$\frac{\text{attn FLOPs}}{\text{proj FLOPs}} = \frac{4L^2 d}{8Ld^2} = \frac{L}{2d}$$

- $L < 2d$：**投影主导**（典型短序列训练，~$O(Ld^2)$ 线性）
- $L = 2d$：两者持平（$d=4096$ → $L=8192$）
- $L > 2d$：**attention 主导**（~$O(L^2 d)$ 二次）

实际场景：
| 模型 | $d$ | 临界 $L$ | $L=32\text{k}$ 时 attention 占比 |
|------|-----|---------|-------------------------------|
| LLaMA-7B | 4096 | 8k | ~80% |
| LLaMA-70B | 8192 | 16k | ~67% |
| 32B+ 长文 | 8192 | 16k | 同上 |

→ 这就是为什么 8k 以下做稠密 attention 没问题，32k+ 必须上 FlashAttention / sliding window / linear（§5–§7）。

### 6.3 训练 vs 推理的成本差异

| 阶段 | 主算子 | FLOPs 主项 | 受限于 |
|------|--------|-----------|--------|
| 训练（fwd+bwd）| 完整 GEMM | $\sim 3\times$ 上表 | compute |
| Prefill | 完整 GEMM | 同训练 fwd | compute |
| Decode（1 token）| $Q$ 是 $(1,d)$，$K/V$ 是 $(L_{\text{cache}},d)$ | $O(L_{\text{cache}}\cdot d)$ | **memory**（读 KV-cache） |

decode 阶段 attention 的 FLOPs 极少，但要把整个 KV-cache 从 HBM 拉到 SRAM，算术强度极低——这就是 §4 roofline、§5.2 MQA/GQA/MLA 的全部动机。

---

## 七、训练时 vs 推理时

| 阶段 | 输入 | 计算 | 瓶颈 |
|------|------|------|------|
| **训练** | 整段 $L$ tokens | 一次大矩阵 attention | compute-bound |
| **Prefill** | prompt $L$ tokens | 同上 | compute-bound |
| **Decode** | 1 新 token | Q 是 1 × d，K/V 累积 | **memory-bound**（要从 HBM 读历史 KV-cache） |

这个差异是 §4（KV-cache & roofline）的根因。

---

## 八、Attention 的可视化与可解释性

- **Attention map** 显示每个 query 对历史 keys 的权重分布
- 早期 head 学位置/语法（"上一个 token"、"句首"），深层 head 学语义
- **Induction heads**（Anthropic 2022）：能完成 `[A][B]...[A] → [B]` 模式补全，是 in-context learning 的电路基础
- 但 attention 权重不严格等于"重要性"（值向量也影响）

---

## 九、常见变体（铺垫后续章节）

| 变体 | 节省 | 章节 |
|------|------|------|
| MHA | baseline | — |
| MQA / GQA | KV-cache | §5.2 |
| MLA | KV-cache（低秩） | §5.2 |
| FlashAttention | HBM IO | §5.1 |
| Sliding Window | $O(L^2) \to O(Lw)$ | §6.1 |
| Linear / SSM | $O(L)$ | §7 |

---

## 关键问答

**Q1**：为什么要除以 $\sqrt{d_k}$ 而不是 $d_k$？
- 假设 q、k 是均值 0 方差 1 的独立向量，$q^\top k$ 的方差是 $d_k$，标准差是 $\sqrt{d_k}$
- 除以 $\sqrt{d_k}$ 让点积**标准差**归一到 1，softmax 输入分布稳定
- 除以 $d_k$ 会让方差变成 $1/d_k$，过度衰减

**Q2**：为什么用多头而不是单头扩大维度？
- 不同 head 学不同子空间的关系（语法、长程、位置等）
- 单头大维度等价于一个全连接，缺少这种"专家分工"
- 实证：head 数从 8 → 32 在固定参数下提升明显

**Q3**：Self-attention 是 permutation-equivariant 吗？
- 是。没有位置信息时，置换输入 = 置换输出（行）
- 这是必须加位置编码的根本原因（§2.2）

**Q4**：Attention 等价于一种"软 retrieval"吗？
- 是。Q 是查询，K 是键，V 是值，softmax 是软索引
- 这就是 Memory-augmented NN / Hopfield network 的现代版

**Q5**：为什么 attention 矩阵是 $L \times L$，能不能并行成 $L \times k$？
- 数学上必须每个 query 对所有 keys 算（causal 下是历史）
- 工程上 FlashAttention 通过 tile / online softmax 避免实例化全矩阵
- 算法上稀疏 / linear attention 把它压到 $O(L)$（§6、§7）

---

## 参考资料

- [Vaswani et al. 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Lilian Weng — Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- [Anthropic — Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Anthropic — In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Karpathy — Let's build GPT (video)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/)
- [3Blue1Brown — Attention in transformers (video)](https://www.youtube.com/watch?v=eMlx5fFNoYc)
