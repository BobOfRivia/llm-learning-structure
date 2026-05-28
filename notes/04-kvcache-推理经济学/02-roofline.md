# 4.2 Roofline 模型

[← 返回框架](../../README.md) · [📎 materials.md → §4.2](../../materials.md)

---

## 一、Roofline 是什么

Roofline 是一个**性能上界模型**（Williams et al. 2009），把硬件的"能跑多快"由两个数字决定：
- **峰值算力**（FLOPs/s）—— 计算屋顶
- **峰值带宽**（Bytes/s）—— 带宽屋顶

任何 kernel 的可达性能上界：
$$\text{Perf} \le \min\bigl(\text{Peak FLOPs}, \ \text{Bandwidth} \times I\bigr)$$

其中 **$I$ = Arithmetic Intensity = FLOPs / Bytes**（每字节算多少次运算）。

```
Performance (FLOPs/s)
│
│            ┌────────────  Peak FLOPs (compute roof)
│           /
│          / ← compute-bound
│         /
│        /
│       / ← memory-bound 区域
│      /
│     / slope = Bandwidth
│    /
│   /
└──────────────── Arithmetic Intensity I (FLOP/Byte)
   I*
```

**转折点** $I^* = \text{Peak FLOPs} / \text{Bandwidth}$。
- $I < I^*$ → memory-bound，性能正比于带宽 × I
- $I > I^*$ → compute-bound，性能恒定为峰值算力

---

## 二、典型 GPU 的 Roofline 数字

| GPU | BF16 FLOPs | HBM 带宽 | $I^*$ (FLOP/Byte) |
|-----|------------|----------|-------------------|
| **A100 80G** | 312 TF | 2.0 TB/s | ~156 |
| **H100 SXM** | 989 TF | 3.35 TB/s | ~295 |
| **H200** | 989 TF | 4.8 TB/s | ~206 |
| **B200** | 2250 TF | 8.0 TB/s | ~281 |
| **MI300X** | 1307 TF | 5.3 TB/s | ~247 |

> 注意：FP8 的 FLOPs 通常是 BF16 的 2 倍，但 KV-cache 主要是 BF16/FP8 内存读取，$I^*$ 计算要按实际精度算。

**核心结论**：现代 GPU 算力涨得比带宽快很多 → $I^*$ 越来越高 → 越多 workload 落到 memory-bound 区。

---

## 三、Transformer 的 Roofline 分析

### 3.1 训练（大 batch / 长序列）

| Op | FLOPs | Bytes | I |
|------|------|-------|-----|
| 矩阵乘 (GEMM, B≫1) | $2BNd^2$ | $\sim(B+d)Nd$ | **大**（百~千）|
| Attention (FlashAttention) | $2BL^2d$ | $\sim BLd$ | **中-大** |
| LayerNorm / RMSNorm | $\sim BLd$ | $\sim BLd$ | **~1**，memory-bound |
| Element-wise (act / +) | $BLd$ | $BLd$ | **~1**，memory-bound |

→ 训练绝大部分时间花在 GEMM，**compute-bound** 占主导，MFU 可达 30-60%。

### 3.2 推理 Prefill

类似训练（一次性算长序列），**compute-bound**。GEMM batch 维度由 batch × prompt_len 提供。

### 3.3 推理 Decode（每步 1 个 token）

| Op | FLOPs | Bytes | I |
|------|------|-------|-----|
| QKV 投影 | $\sim 8Bd^2$ | $4d^2$ (weight) + $Bd$ | $\approx 2B$ |
| Attention（读 KV-cache）| $4BLd$ | $2BLd$ (read KV) | **~2** ⭐ |
| FFN | $\sim 16Bd^2$ | $\sim 16 d^2$ (weight) | $\approx B$ |
| LM head | $2BdV$ | $dV$ (weight) | $\approx B$ |

**关键观察**：
- decode 每步算的 token 只有 batch B 个 → I ≈ B
- **B 必须很大**（典型 32~256）才能从 memory-bound 区进入 compute-bound 区
- attention 部分的 I 不依赖于 B（每个 batch 的 KV 不同），**永远 memory-bound** ⭐

→ decode 阶段的工程目标全部是"**增大 batch**"和"**减少 KV 读取**"。

---

## 四、用 Roofline 解释经典优化

| 优化 | 改的是什么 | Roofline 上的位置 |
|------|------------|-------------------|
| **FlashAttention** | 减 Bytes（不写 attention 矩阵） | I ↑ → 从 memory 推到 compute |
| **GQA / MQA** | 减 KV bytes | I ↑（attention 部分）|
| **MLA** | 减 KV bytes（低秩） | I ↑↑ |
| **PagedAttention** | 提升有效 batch（减碎片） | I ↑（FFN/GEMM 受益）|
| **Continuous batching** | 增大 batch | I ↑（GEMM 进 compute-bound）|
| **量化 (W8A8, FP8)** | 减 weight bytes | I ↑ + 算力 ↑ |
| **Speculative decoding** | 让 decode 步多 token | I ↑（attention 从 1 算到 k）|
| **稀疏 attention (NSA/DSA)** | 跳过部分 KV 读取 | 直接降 Bytes |

> 几乎所有 LLM 推理优化都是在 **降低 Bytes 或增大 batch** —— 即把 I 推到 $I^*$ 以右。

---

## 五、Roofline 的局限

- 假设 kernel 完美利用硬件（实际 GEMM 也只到 ~80% peak）
- 忽略 latency（小 batch 时启动开销可能更重要）
- 不处理多 GPU 通信（NVLink/InfiniBand 是第三条 roof）
- 不处理热 / 功耗（B200 持续 peak 时降频）

更完整的模型：**HBM Roofline + NVLink Roofline + L2 Roofline** 多层 roofline。

---

## 六、亲手估一次（必练）

**问题**：H100 上跑 Llama-3 70B GQA（KV head=8）decode，batch=64，L=4k，问 attention 部分 compute or memory bound？

- 单 token attention FLOPs（一层）≈ $4 L d_h h$ ≈ $4 \cdot 4096 \cdot 128 \cdot 64$ ≈ 134 MFLOPs
- 单 token attention bytes（KV 读取）≈ $2 \cdot L \cdot d_h \cdot \text{KV-head} \cdot 2$ = $2 \cdot 4096 \cdot 128 \cdot 8 \cdot 2$ ≈ 16 MB
- I ≈ 134M / 16M ≈ **8**
- H100 $I^* \approx 295$
- 8 << 295 → **强 memory-bound** ⭐

即使把 batch 拉到 64，attention 依然 memory-bound。
→ 该场景 attention 的吞吐由 HBM 带宽决定，**不是**由算力决定。

---

## 关键问答

**Q1**：为什么 LLM 推理这么"反直觉地依赖带宽"？
- decode 每步只算 1 个新 token 的运算，但要读全部权重 + 全部 KV
- I = 计算 / 读取 ≈ O(B)，B 不够大就 memory-bound
- 硬件越强（算力涨快于带宽）这个矛盾越尖

**Q2**：FlashAttention 是 compute-bound 优化还是 memory-bound 优化？
- 是 **memory-bound 优化**：减少 HBM 读写，让 attention 不再写 $L^2$ 矩阵
- 在长序列训练中，attention 原本是 memory-bound，FA 把它推到 compute-bound

**Q3**：MFU 高就一定快吗？
- 不一定。MFU 高表示算力利用率高，但如果 workload 应该在 memory-bound 区
- 真正的衡量：tokens/s 和单位 FLOPs 成本
- 但 MFU > 50% 通常意味着 kernel 工程做得不错

**Q4**：Speculative decoding 为什么提升性能？
- target model 一次验证 k 个 draft token
- 每步的 attention 从 1 个 Q × L KV 变成 k 个 Q × L KV
- I 提升 k 倍 → 从 memory-bound 进入 compute-bound 区
- 实质：**用 GPU 闲置算力把 memory-bound 任务"变密"**

**Q5**：为什么 prefill 和 decode 的优化策略完全不同？
- prefill：compute-bound → 看算力（FlashAttn 的 throughput 配方）
- decode：memory-bound → 看 KV 大小、batch、bandwidth
- 工业上把它们分开调度（chunked prefill、分离式服务）

**Q6**：为什么大 batch 在 decode 阶段尤其重要？
- 增大 batch 直接放大 weight reuse → FFN/GEMM 的 I 跟 B 成正比
- 但 attention 的 I 不随 B 增长（每条请求 KV 独立）
- 所以 batch 大到一定后，FFN 进 compute-bound，attention 仍 memory-bound → attention 成为瓶颈
- 这是后续稀疏 attention / MLA 等优化的动机

---

## 参考资料

- [Williams et al. 2009 — Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [Horace He — Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) ⭐
- [Pope et al. 2022 — Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [Dao et al. 2022 — FlashAttention](https://arxiv.org/abs/2205.14135)
- [Dao 2023 — FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [NVIDIA H100 White Paper](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)
- [Semianalysis — GPU Performance Deep Dives](https://www.semianalysis.com/)
