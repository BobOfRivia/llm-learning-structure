# 5.1 FlashAttention 1 / 2 / 3

[← 返回框架](../../README.md) · [📎 materials.md → §5.1](../../materials.md)

---

## 一、标准 Attention 的瓶颈

标准 attention（PyTorch naive 实现）：

```
S = Q @ K^T          # (L,L) 矩阵，大！
P = softmax(S)       # (L,L)
O = P @ V            # (L,d)
```

显存与时延都被 $L \times L$ 矩阵卡住：
- **显存**：必须把 $L \times L$ 的 S/P 写回 HBM —— $O(L^2)$ 激活
- **带宽**：S、P 在 HBM 与 SRAM 之间反复读写
- 长序列时（L=8k~128k），HBM 读写量爆炸 → **memory-bound**

直觉：算力（H100 ≈ 1 PFLOPs）很多，但带宽（3.35 TB/s）有限。
真正的瓶颈是"为算 attention 来回搬数据"，而不是"算 softmax"。

---

## 二、FlashAttention-1（Dao 2022）

核心思想（三句话）：
1. **Tiling**：把 Q、K、V 切成 SRAM 能放下的小块（block size $B_r \times B_c$）
2. **Online softmax**：分块逐步累积 softmax，不需要先看全行
3. **Recompute**：backward 时不存 attention matrix，重算（FLOPs 增加 25%，但显存节省巨大）

### 2.1 Online Softmax 关键公式

给定一行已扫了 j 个 block，维护：
- $m_j$：当前最大值
- $\ell_j$：当前归一化分母
- $O_j$：当前累积输出

来一个新 block $S^{(j+1)}$：

$$m_{new} = \max(m_j, \max(S^{(j+1)}))$$
$$\ell_{new} = e^{m_j - m_{new}} \ell_j + \sum e^{S^{(j+1)} - m_{new}}$$
$$O_{new} = \frac{\ell_j e^{m_j - m_{new}}}{\ell_{new}} O_j + \frac{e^{S^{(j+1)} - m_{new}}}{\ell_{new}} V^{(j+1)}$$

→ 全程不需要把 S 矩阵实例化，只在 SRAM 内做计算。

### 2.2 算法骨架

```
for i = 1..N_r:           # 外循环：Q 分块
  load Q_i to SRAM
  m_i = -inf, l_i = 0, O_i = 0
  for j = 1..N_c:         # 内循环：K,V 分块
    load K_j, V_j to SRAM
    S_ij = Q_i @ K_j^T
    update m_i, l_i, O_i  # online softmax
  write O_i to HBM
```

### 2.3 收益

| 指标 | naive | FlashAttention |
|------|-------|----------------|
| HBM 读写 | $O(L^2 d)$ | $O(L^2 d^2 / M)$，M=SRAM 大小 |
| 显存（激活）| $O(L^2)$ | $O(L)$ |
| 实测 wall time | baseline | A100 上 **2-4×** |
| 长序列（16k） | OOM | 正常运行 |

> FlashAttention 是 2022 后 LLM 训练的事实标准，没有它训不到 8k 以上。

---

## 三、FlashAttention-2（Dao 2023）

FA1 已有 ~25% TFLOPs 利用率，但还有 3 个低效问题，FA2 全部解决：

### 3.1 主要改进

| 问题 | FA1 | FA2 优化 |
|------|-----|---------|
| 非 matmul ops（rescale 等）多 | 每 block 都做 rescale | 分块内 rescale，最后一次性归一 |
| 内外循环不优 | 外循环 Q（小），内循环 KV（大）| **外循环 KV→外循环 Q**，更好并行 |
| 单 warp 处理整行 | warp 之间负载不均 | 多 warp 共同处理一行 |
| backward 重算开销 | 全块重算 | 优化重算调度 |

### 3.2 性能数字

A100 BF16：
- FA1：~125 TFLOPs/s
- FA2：~225 TFLOPs/s（≈ 72% MFU on A100）
- vs PyTorch naive：~10×

H100 BF16：
- FA2 利用率 ~35% TFLOPs（因为 H100 加速比涨太快，FA2 没用上 TMA）

→ 这正是 FA3 出场的契机。

---

## 四、FlashAttention-3（Shah, Tri Dao 2024）

为 H100/Hopper 架构定制：

### 4.1 三个 Hopper 新硬件特性

| 特性 | 用途 |
|------|------|
| **WGMMA (Warp Group MMA)** | 异步矩阵乘指令 |
| **TMA (Tensor Memory Accelerator)** | 异步全局↔共享内存搬运 |
| **FP8 with extra accumulator** | 高吞吐量低精度 |

### 4.2 三大优化

1. **Producer-consumer pipeline (Warp Specialization)**：一组 warp 专门搬数（TMA），另一组算（WGMMA），两者并发
2. **GEMM-softmax overlap**：算下一块 QK^T 的同时做当前块的 softmax（用乒乓 buffer 隐藏 softmax 时延）
3. **FP8 with incoherent processing**：FP8 的 outlier 用 Hadamard 矩阵预乘"摊平"，量化误差减半

### 4.3 性能数字

H100 BF16：
- FA2：**~35% MFU**
- FA3：**~75% MFU**（**~740 TFLOPs/s**），是 FA2 的 1.5-2× ⭐

H100 FP8：
- FA3：**~1.2 PFLOPs/s**，几乎对齐峰值

> FA3 是当前训推一体最快的 attention 实现，已被 vLLM/SGLang/PyTorch 整合。

---

## 五、对比表

| 版本 | 出版 | 关键创新 | A100 BF16 | H100 BF16 |
|------|------|---------|-----------|-----------|
| naive | — | — | ~30 TF | ~50 TF |
| FA1 | 2022 | Tiling + online softmax | ~125 TF | — |
| FA2 | 2023 | 改循环、改并行 | ~225 TF | ~350 TF (~35%) |
| FA3 | 2024 | WGMMA + TMA + warp-spec + FP8 | — | ~740 TF (~75%) ⭐ |

---

## 六、对各下游优化的影响

- **训练**：长上下文（>16k）一律标配；recompute 才让训练能上 100k
- **推理 prefill**：FA-decode kernel 也使用 FA2/3 内核（同样 compute-bound）
- **推理 decode**：FA-decode 是不同 kernel（attention M-bound，不是 FA 主战场）
- **稀疏 attention（NSA/DSA）**：FA 的 block-sparse 扩展（block-mask）
- **MLA**：MLA 解压后仍走 FA 路径

---

## 关键问答

**Q1**：FlashAttention 为什么能"减少 HBM 读写"？
- HBM 慢、SRAM 快但小（per-SM ~192KB）
- FA 把 Q/K/V 切到能放进 SRAM 的小块，所有中间 S/P 都在 SRAM 内消化
- 只写最终输出 O 回 HBM
- 标准 attention 必须把 (L,L) 矩阵写回 HBM → FA 跳过这一步

**Q2**：FlashAttention 是 exact 还是 approximate？
- **完全 exact**！跟 naive softmax 数学等价（online softmax 是恒等变换）
- 与 sparse attention 不同（那是 approximate）

**Q3**：FA 在小 L（如 512）上还有收益吗？
- 小 L 上 attention 本来就 compute-bound，FA 收益小（甚至慢一点点，因为 tiling overhead）
- 收益与 L 成正比，L 越大越赚

**Q4**：FA 的 backward 怎么不存 attention matrix？
- 重新 forward 算一遍 S、softmax（增加 25% FLOPs）
- 用 forward 时存下的 $\ell$、$m$（每行 1 标量）快速重建 softmax
- 是经典的 **gradient checkpointing** 思路在 attention 上的应用

**Q5**：FlashAttention 怎么处理 causal mask？
- 对 $j > i$ 的块直接 skip（一半的工作量）
- 边界块（$i = j$）单独处理 triangular mask
- 实测 causal mask 比 full attention 快 ~2×（如预期）

**Q6**：FA3 为什么不能 trivially port 回 A100？
- 用的是 H100 专属指令（WGMMA、TMA、warp-spec）
- A100 没有这些 → 退化为 FA2

---

## 参考资料

- [Dao et al. 2022 — FlashAttention](https://arxiv.org/abs/2205.14135) ⭐
- [Dao 2023 — FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao 2024 — FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Tri Dao 博客（FA3 解读）](https://tridao.me/blog/2024/flash3/)
- [Horace He — Brrr Intro](https://horace.io/brrr_intro.html)
- [Online Softmax (Milakov & Gimelshein 2018)](https://arxiv.org/abs/1805.02867)
- [HazyResearch GitHub flash-attention](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA H100 White Paper（TMA / WGMMA 介绍）](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)
