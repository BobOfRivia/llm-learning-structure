# 补充 4：GPU 运算基础（GEMM / Roofline / 内存层级 / Tensor Core）

[← 返回框架](../../README.md)

> 这是一篇**跨章节复用**的工具笔记。LLM 训练 / 推理几乎所有性能问题最终都会落到三个硬件事实上：
> - **GEMM**（通用矩阵乘）是 GPU 上的"主菜",训练 / 推理 95% 以上的算力都在跑它。
> - **Roofline 模型** + **算术强度**(arithmetic intensity)决定一段代码究竟是"算不过来"还是"搬不过来"。
> - **HBM ↔ SMEM ↔ 寄存器** 三级存储 + **Tensor Core** + **kernel launch** 是所有具体优化的物理来源。(SMEM 在文献尤其 FlashAttention 论文里常被称为 `SRAM`,是同一块东西的不同叫法,见 §2.1。)
>
> 后续章节里会反复出现的术语,**这一篇集中讲清楚**:
> - §4.4 MFU 指标 / Roofline / 内存带宽瓶颈
> - §4.3 prefill compute-bound vs decode memory-bound
> - §5 FlashAttention 为什么要 tiling 进 SMEM(论文里称 SRAM)
> - §10.2 / §10.3 TP / DP / PP 中的 communication-bound 分析
> - §12.1 PagedAttention(KV 碎片)、§12.2 Continuous Batching 的「batched GEMM 形状必须固定」的硬约束
> - §12.3 CUDA Graph 重录代价
> - §12.4 量化(FP16 / FP8 / FP4)的算力提升来自 Tensor Core 不同精度的吞吐差

---

## 零、前置：GPU 跟 CPU 有什么不同（小白先读这一节）

如果你之前主要写 CPU 程序，理解 GPU 的第一步是记住一件事：**它们的设计哲学完全相反**。

| | CPU | GPU |
|---|---|---|
| 核数 | 几个 ~ 几十个**大核** | 上万个**小核** |
| 单核能力 | 强（乱序执行、分支预测、大缓存） | 弱（顺序执行、几乎没有分支预测） |
| 擅长 | **顺序逻辑、复杂控制流** | **大量重复、彼此独立的计算** |
| 类比 | 几位博士各自处理复杂任务 | 一万个小学生同时做加减法 |

**为什么矩阵乘特别适合 GPU?** 看 $C_{[M, N]} = A_{[M, K]} \cdot B_{[K, N]}$:输出矩阵的每个元素 $C[i, j] = \sum_k A[i, k] \cdot B[k, j]$ —— **每个 $C[i, j]$ 之间完全独立**,可以同时算。一万个小学生一人负责一格,简直是为 GPU 量身定做。

**LLM 又恰好 99% 算力都在跑矩阵乘**(§1 展开)。这就是为什么:
- LLM 训练 / 推理几乎只在 GPU 上跑,CPU 干不了。
- 衡量 GPU 性能时不看 CPU 那套指标(主频、IPC),而是看"每秒能做多少次矩阵乘"(TFLOPS)。

**本篇要回答的三个问题:**

| 问题 | 在哪一章 |
|---|---|
| GEMM 在**数学上**是什么?为什么 LLM 都在算它? | §1 |
| GEMM 在 GPU **硬件上怎么跑**? 数据在哪、谁来算、怎么协作? | §2(+ §3 精度细节) |
| 一段 GEMM 跑得**快不快、瓶颈在哪**? | §4(Roofline) |

后面 §5~§9 是把上面三件事用到具体场景(batched GEMM、一次 decode、踩坑、公式表)。

---

## 一、GEMM:LLM 算力的"主菜"

### 1.1 LLM 里 99% 的算力都在跑矩阵乘

一个 Transformer block 的 forward 拆开看,每一步都是矩阵乘:

| 操作 | 形状(忽略 batch) | 是不是 GEMM |
|---|---|---|
| QKV 投影 | $[T, d] \times [d, 3d] \to [T, 3d]$ | **GEMM** |
| Attention 内部($QK^\top$、softmax×V) | per-head 的小 GEMM | **GEMM** |
| O 投影 | $[T, d] \times [d, d] \to [T, d]$ | **GEMM** |
| FFN(up + gate + down) | $[T, d] \times [d, d_{ff}]$ 等 | **GEMM**(占比最大) |
| LM head | $[T, d] \times [d, V] \to [T, V]$ | **GEMM**(单层最大) |
| LayerNorm / softmax / RoPE / 激活函数 | 逐元素或归约 | **不是** GEMM |
| Embedding lookup | 查表 | 不是 GEMM(也几乎不耗算力) |

→ 真正吃 FLOPs 的几乎全是 GEMM,**非 GEMM 部分(LayerNorm、softmax、激活、RoPE)合起来通常 < 5% FLOPs**,但它们在 decode 阶段往往是延迟瓶颈(原因见 §4)。

### 1.2 GEMM 的标准形式

GEMM = **GE**neral **M**atrix **M**ultiply,通用矩阵乘。NVIDIA 文档里的标准形式是:

$$C = \alpha \cdot A \cdot B + \beta \cdot C$$

- $A$: $[M, K]$
- $B$: $[K, N]$
- $C$: $[M, N]$
- $\alpha, \beta$ 是标量,LLM 里通常 $\alpha = 1, \beta = 0$,也就是简单的 $C = A \cdot B$。

**三个维度名字一定要记住**:`M / N / K`。后面几乎所有性能分析都用它们。

| 维度 | 含义 | LLM 里通常对应谁 |
|---|---|---|
| M | 输出行数 / batch×seq 维 | $B \cdot T$(prefill)或 $B$(decode,T=1) |
| N | 输出列数 / 输出隐藏维 | $d_{model}$ 或 $d_{ff}$ |
| K | 收缩维 / 输入隐藏维 | $d_{model}$ 或 $d_{ff}$ |

FLOPs 公式:**每个输出元素做 K 次乘加 = 2K FLOPs**(一次乘法 + 一次加法),总 FLOPs:

$$\text{FLOPs}_{GEMM} = 2 \cdot M \cdot N \cdot K$$

> 这个公式贯穿全篇,后面 Roofline / 算术强度 / MFU 都靠它。

---

## 二、GPU 是怎么算 GEMM 的(硬件 + 执行模型)

> §1 告诉了你 GEMM 在数学上是 $C = A \cdot B$。这一章回答:**这个矩阵乘到底是怎么在 GPU 上跑起来的**——数据放在哪、谁来算、怎么协作。

### 2.1 GPU 的硬件长什么样

一块现代 GPU(以 H100 为例)的内部结构:

```
GPU
 ├─ SM 0  (Streaming Multiprocessor,流式多处理器)
 │   ├─ Tensor Core × 4              ─→ 专门跑矩阵乘的电路(§2.2)
 │   ├─ CUDA Core × 128              ─→ 通用标量/向量计算
 │   ├─ Register File (~256 KB)      ─→ 每线程私有,最快
 │   └─ Shared Memory / L1 (~228 KB) ─→ block 内共享,可编程
 ├─ SM 1
 ├─ ...
 ├─ SM 131                            (H100 有 132 个)
 ├─ L2 Cache (~50 MB)                ─→ 全 GPU 共享
 └─ HBM (80~192 GB)                  ─→ 显存,最慢但容量最大
```

几个关键名词:

- **SM(Streaming Multiprocessor,流式多处理器)**:GPU 的"小型核心"。H100 有 132 个 SM,每个 SM 自带几十个计算单元 + 自己的小缓存。所有 SM 并行工作。
- **HBM(High Bandwidth Memory,高带宽显存)**:GPU 的"主存",容量 80~192 GB。模型权重、KV cache、所有 tensor 默认都住这里。
- **SMEM / Shared Memory(共享内存)**:每个 SM 内部的高速小缓存,228 KB 左右。和 CPU 的 L1 不同的一点是:**程序员可以显式控制 SMEM 里放什么**(这一点 §2.4 会大量用到)。
  - ⚠️ **术语提示**:FlashAttention 论文以及很多性能分析文章里说的 **`SRAM`** 就是这块 SMEM(严格说还包括 L1 和寄存器,统称片上 SRAM)。**SRAM 是电路技术名**(Static RAM,跟片外 DRAM/HBM 相对),**SMEM 是 NVIDIA 的功能名**——指同一块物理存储。本篇统一用 SMEM,看到论文里写 SRAM 直接代入即可。
- **L2 Cache**:全 GPU 共享的二级缓存,介于 HBM 和 SMEM 之间。
- **寄存器(Register)**:每个线程私有的最快存储,只能放几十~几百个值。

各层级的延迟和带宽量级(H100):

| 层级 | 容量 | 带宽 | 相对延迟 |
|---|---|---|---|
| 寄存器 | 256 KB/SM | ~33 TB/s/SM | 1 cycle |
| Shared Memory / L1 | 228 KB/SM | ~19 TB/s/SM | ~20 cycles |
| L2 Cache | 50 MB | ~5.5 TB/s | ~200 cycles |
| HBM3 | 80 GB | **3.35 TB/s** | ~500 cycles |
| PCIe 5.0(到 CPU) | — | 64 GB/s | ~微秒级 |
| NVLink 4(到其他 GPU) | — | 900 GB/s | ~微秒级 |

→ **跨级访问的代价跨 5 个数量级**。这就是为什么所有 GPU 优化的核心动作都是同一个:**让数据尽量停在 SMEM 和寄存器里反复用,不要频繁回 HBM 拉**。FlashAttention(§5)、kernel fusion 全是这一思路的不同实现。

### 2.2 Tensor Core:专门跑矩阵乘的硬件单元

§2.1 表格里出现了两种计算单元:CUDA Core 和 Tensor Core。两者差别巨大,而且 Tensor Core 是 LLM 性能的核心,所以单独讲。

| | CUDA Core | Tensor Core |
|---|---|---|
| 一条指令算什么 | 一次标量乘加(FMA) | **一整个小矩阵的乘加** |
| 典型形状 | 标量 / 几个浮点 | $16 \times 16$ 矩阵 MMA |
| 适合干啥 | 控制流、reduction、elementwise(LayerNorm / SiLU) | **GEMM、卷积** |
| 算力差距 | 1× | **一个数量级以上** |

**Tensor Core 一次做的事**(称为 MMA,Matrix-Multiply-Accumulate):

$$D_{16 \times 16} = A_{16 \times 16} \cdot B_{16 \times 16} + C_{16 \times 16}$$

也就是一条硬件指令完成两个 $16 \times 16$ 矩阵相乘 + 累加。本质上是把"几百次标量乘加"压成了一条指令——这就是 GPU 在 GEMM 上能跑那么快的物理根源。

(具体 tile 大小随代际变化:Hopper 上还有 wgmma 异步指令对应 $64 \times N$ 形状;Blackwell 又有新一代 MMA。形状细节不重要,重要的是"一条指令吃一个矩阵片段"这个本质。)

**关键结论**:**一段代码能不能用上 Tensor Core,决定了它能不能接近 GPU 的峰值算力**。不能用 Tensor Core 而只能跑 CUDA Core 的 kernel,算力直接差一个数量级以上。所有现代深度学习框架(PyTorch、JAX)和库(cuBLAS、cuDNN、CUTLASS、Triton)的 GEMM 内核都尽量把计算调度到 Tensor Core 上。

> Tensor Core 在不同**精度**(FP16 / BF16 / FP8 / FP4)下的吞吐和数值特性,见 §3 表格。低精度更快是因为同一条 MMA 指令能塞下更多元素——这是量化(§12.4)能立即提速的原因。

### 2.3 软件抽象:kernel / grid / block / warp / thread

GPU 上跑的代码叫 **kernel(核函数)**——可以理解为"由 CPU 发起、在 GPU 上并行执行的一段函数"。一次 kernel 调用会同时启动成千上万个线程并行跑同一段代码。

这些线程不是平铺的,而是分层组织的:

```
kernel 一次调用 = 一个 grid
grid
 ├─ block 0 ─→ 整个 block 调度到某个 SM 上执行
 │   │           block 内所有线程共享同一块 shared memory
 │   ├─ warp 0  (32 个线程,SIMT 同步执行同一条指令)
 │   ├─ warp 1
 │   └─ ...
 ├─ block 1
 └─ ...
```

四个名词一句话定义:

- **thread(线程)**:执行 kernel 的最小单位,可以理解为"程序员视角下的一份独立程序"。
- **warp(线程束)**:**32 个线程绑在一起,执行同一条指令**(SIMT,Single Instruction Multiple Threads)。这是 GPU 硬件调度的基本单位,程序员无法拆开。
- **block(线程块)**:一组线程(通常 128 ~ 1024 个),**整个 block 跑在同一个 SM 上**,内部可以通过 SMEM 互相通信。跨 block 不能直接通信。
- **grid(网格)**:一次 kernel 调用里所有 block 的集合。

为什么这样设计?——为了**精确匹配硬件**:

| 软件抽象 | 对应硬件 |
|---|---|
| thread | CUDA Core(或 Tensor Core 内部 lane) |
| warp(32 线程) | SM 的调度单位 |
| block | 一个 SM(整块 block 都跑在同一 SM 上) |
| grid | 整块 GPU |

→ Tensor Core 的输入(那个 $16 \times 16$ 的矩阵片段)**正好由一个 warp 协作驱动**——所以 GEMM 的 tile 形状常见是 16 / 32 / 64 / 128 的倍数,就是为了对齐 warp 和 Tensor Core 的硬件粒度。

### 2.4 一个 GEMM 的完整旅程(把上面三节串起来)

现在用一个具体例子把硬件、Tensor Core、软件抽象全串起来。算一次 FFN-up:

$$C_{[2048, 14336]} = A_{[2048, 4096]} \cdot B_{[4096, 14336]}$$

直接搬整张 A、B 进 SMEM 不可能——SMEM 才 228 KB,装不下这种规模的矩阵。所以 **GEMM 必须"切 tile",像拼图一样组装**:

**步骤 1:CPU 发起 kernel,GPU 接管**

PyTorch 里写 `C = A @ B` → 底层调到 cuBLAS 的 GEMM kernel → CPU 通过 driver 把 kernel + 形状参数下发给 GPU → GPU 启动一个 grid。

**步骤 2:把输出 C 切 tile,每块分给一个 block**

把 $C_{[2048, 14336]}$ 切成 $128 \times 128$ 的小块:

$$\text{tile 数} = \frac{2048}{128} \times \frac{14336}{128} = 16 \times 112 = 1792 \text{ 个 tile}$$

→ 启动 1792 个 block,**每个 block 负责算一个 C tile**。这 1792 个 block 被分批调度到 132 个 SM 上(每个 SM 同时跑若干个,跑完一批再换下一批)。

**步骤 3:每个 block 内部,先取"条带",再沿 K 维分段累加**

一个 block 负责的 C tile 在 C 中的位置标作 $(i, j)$——意思是 C 矩阵从第 $i$ 行 / 第 $j$ 列开始的那块 $[128, 128]$ 小方块。这个 block **只关心** A、B 的两个"条带":

| 名字 | 含义 | 形状 | FP16 大小 |
|---|---|---|---|
| $A_{\text{stripe}}$ | A 的第 $i \sim i+127$ 行(横条,全部 K 列) | $[128, 4096]$ | ~1 MB |
| $B_{\text{stripe}}$ | B 的第 $j \sim j+127$ 列(竖条,全部 K 行) | $[4096, 128]$ | ~1 MB |

那么 tile 的精确数学就是这两个条带做矩阵乘:

$$C_{\text{tile}} = A_{\text{stripe}} \cdot B_{\text{stripe}}, \quad \text{形状}\ [128, 4096] \times [4096, 128] = [128, 128]\ \checkmark$$

但**两段条带加起来 2 MB,SMEM 才 228 KB,放不下**。所以再沿 K 维切成 32 一段(CUTLASS 里这个粒度叫 `CtaTileK`,本例取 32):

$$C_{\text{tile}} = \sum_{k_b = 0}^{127} \underbrace{A_{\text{stripe}}[:,\ 32 k_b : 32(k_b+1)]}_{A_{\text{chunk}}\ [128, 32]} \cdot \underbrace{B_{\text{stripe}}[32 k_b : 32(k_b+1),\ :]}_{B_{\text{chunk}}\ [32, 128]}$$

每段 chunk 的形状和大小:
- $A_{\text{chunk}}$:$[128, 32]$ FP16 ≈ **8 KB**
- $B_{\text{chunk}}$:$[32, 128]$ FP16 ≈ **8 KB**
- 一对 ≈ 16 KB,SMEM 轻松装下

伪代码:

```
for k_block in [0, 32, 64, ..., 4064]:   ← K 维分 128 段
    1. 从 HBM 读 A_chunk[128, 32] + B_chunk[32, 128]
       → 过 L2 cache → 落到 SMEM(≈ 16 KB)
    2. block 内的 warp 各自从 SMEM 取 16×16 fragment
       → 装进自己的寄存器
    3. 喂给 Tensor Core 做 MMA:
          D_frag = A_frag @ B_frag + D_frag   ← 累加进寄存器
    4. 累加器留在寄存器,不写回 HBM
```

> 上面这种"先按 tile 取 A/B 条带,再沿 K 维迭代加载 chunk 做累加"的两层切分,就是 NVIDIA CUTLASS 里 **threadblock tiling** 的标准结构(每个 threadblock 拿 A 的 `[M_tile, K]` row stripe、B 的 `[K, N_tile]` column stripe,沿 K 维 mainloop 累加 `[M_tile, K_tile]` × `[K_tile, N_tile]` 的小积)。cuBLAS / Triton 等所有现代 GEMM kernel 都是这一套。

**步骤 4:K 循环结束,写回 HBM**

把寄存器里累加好的 $[128, 128]$ tile 写回 HBM 的 C 矩阵对应位置。**整个 K 循环(128 次迭代)过程中,中间结果一次都没碰 HBM**——只在寄存器和 SMEM 之间流动。

**这张旅程为什么重要**

整个 GEMM 在做的事可以浓缩成一句话:**HBM 太慢,所以策略是"分块搬一次,在 SMEM/寄存器里反复用,算完再搬回去"**。具体到每份数据:

| 数据 | 住在哪 | 被读取的次数 |
|---|---|---|
| A、B、C 整体 | HBM | 每个元素只搬 1~2 次 |
| A_chunk、B_chunk(每段 K) | SMEM | 被 block 内所有 warp 反复读 |
| A_frag、B_frag | 寄存器 | 直接喂 Tensor Core,一条 MMA 指令吃一次 |
| 累加器 | 寄存器 | K 循环 128 次都不落 HBM |

→ 这是后面所有 GPU 优化的母版。FlashAttention(§5)把 attention 也变成"分块进 SMEM、不落 HBM";MoE(§9)减少要搬的权重;Continuous Batching(§12.2)增大 M 让每次搬运被更多次乘法分摊。**核心思想全是同一条**——减少 HBM 流量。

### 2.5 kernel launch overhead

最后一个硬件事实:CPU 启动一个 GPU kernel 要走一遍 driver + runtime,实测延迟约 **5~20 μs**。

> 这个数字看起来小,但在 decode 阶段(每次 forward 只生成 1 token)、一个 70B 模型一次 forward 要跑 ~80 层 × 5+ 个 kernel(QKV / attn / O / FFN-up / FFN-down / norm / ...),每次 forward 就是 ~400 次 launch,即 ~5 ms 纯调度开销——和真正的 GPU 算时同量级。

解法:

| 方案 | 思路 |
|---|---|
| **kernel fusion** | 把多个小 kernel 编译成一个大 kernel,减少 launch 次数 |
| **CUDA Graph**(§12.2) | 把一整个 forward 录制成图,后续 replay 只交一次给 driver,launch 摊销到接近 0 |
| **persistent kernel** | 一个长寿命 kernel 在 SM 上常驻,自己循环吃任务 |

注意 CUDA Graph 有个硬约束:**所有 tensor 形状必须固定**,任何 shape 变化都得重新录制。这就是 §12.2 里"batched GEMM 的 batch 维 B 必须编译期定死"的物理来源(详见 §5)。

---

## 三、Tensor Core 各精度的吞吐与数值特性

> §2.2 介绍了 Tensor Core 这个硬件单元。这一章展开它在**不同数据精度**下的吞吐差异和数值特性——这是量化(§12.4)能立刻提速的硬件根源。

### 3.1 各精度的吞吐(以 H100 SXM 为例,单位 TFLOPS)

| 精度 | CUDA Core | Tensor Core(dense) | Tensor Core(2:4 稀疏) |
|---|---|---|---|
| FP32 | 67 | — | — |
| TF32 | — | 989 | 1979 |
| BF16 / FP16 | — | **1979** | 3958 |
| FP8(E4M3 / E5M2) | — | **3958** | 7916 |
| INT8 | — | 3958 | 7916 |

Blackwell(B200) 进一步加了 FP6 / FP4:

| 精度 | B200 TC dense |
|---|---|
| FP16 / BF16 | 2250 |
| FP8 | 4500 |
| FP6 | 4500 |
| **FP4** | **9000** |

**规律**:**每降一档精度,Tensor Core 吞吐翻倍**(FP16→FP8→FP4)。原因:一条 MMA 指令的总 bit 是定长的,精度减半 → 同一条指令能塞两倍的元素。**这是硬件设计,不是软件优化**——这就是为什么量化(§12.4)能直接获得算力提升。

### 3.2 各精度的数值特性

| 精度 | 总 bit | 指数 | 尾数 | 动态范围(量级) | LLM 用途 |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | $\pm 10^{38}$ | master weight、optimizer state |
| TF32 | 19 | 8 | 10 | $\pm 10^{38}$ | A100 / H100 上 FP32 的默认替代 |
| BF16 | 16 | 8 | 7 | $\pm 10^{38}$ | **训练首选**(动态范围同 FP32) |
| FP16 | 16 | 5 | 10 | $\pm 10^{4}$ | 早期训练 / 推理(易溢出,要 loss scaling) |
| FP8 E4M3 | 8 | 4 | 3 | $\pm 10^{2}$ | **推理常用 + Hopper 训练** |
| FP8 E5M2 | 8 | 5 | 2 | $\pm 10^{4}$ | 训练 backward gradient 用 |
| FP4 | 4 | 2 | 1 | $\pm 8$ | Blackwell 推理 + 训练实验 |

记忆口诀:**指数定动态范围,尾数定相对精度**。BF16 的指数位数和 FP32 一样多,所以训练里几乎完全替代 FP16——同样 16 bit,BF16 不容易溢出。

> 涉及训练稳定性 / 溢出 / loss scaling 见 §10.5。

---

## 四、Roofline:给一个 GEMM 性能打分

> 现在我们知道 GEMM 是什么(§1)、它在 GPU 上怎么跑(§2)、Tensor Core 能多快(§3)。这一章回答最实用的问题:**一段代码到底跑得多快?瓶颈是算力(Tensor Core 满载)还是带宽(HBM 搬不过来)?**
>
> 本章的输入就是 §1 的 GEMM:**FLOPs 来自公式 $2MNK$,Bytes 来自 §2.4 那张旅程里的 HBM 流量**。

### 4.1 算术强度(Arithmetic Intensity)

定义:

$$I = \frac{\text{FLOPs}}{\text{Bytes moved from HBM}}$$

**单位 FLOP/byte**。它衡量"每搬一字节数据,顺便算多少次"。

直觉:

- $I$ **大** → 数据搬一次反复算很多次 → 算力是瓶颈 → **compute-bound**
- $I$ **小** → 数据搬来用一次就扔 → 带宽是瓶颈 → **memory-bound**

回看 §2.4 的旅程:A_chunk 搬一次进 SMEM 被多个 warp 反复读、累加器在寄存器里循环 128 次而不落 HBM——这些"反复用"都在抬高 $I$。所以**写得好的 GEMM 算术强度可以很高**;反过来,elementwise kernel(LayerNorm、激活函数)每个元素读一次算一次就扔,$I$ 只有 1~2,**天然 memory-bound**。

### 4.2 Roofline 图

把"实际可达吞吐 $P$" 画成 $I$ 的函数:

```
 P (TFLOPs/s)
   ▲
   │                  ┌─────────────────  ← compute roof (硬件峰值)
   │                /
   │              /
   │            /
   │          /  ← memory roof = BW × I
   │        /     斜率 = 内存带宽
   │      /
   │    /
   │  /
   │/_________________________________→  I (FLOPs/byte)
       ↑
   I_crit (临界算术强度)
   = compute_peak / bandwidth
```

公式:

$$P(I) = \min(P_{\text{peak}}, \; \text{BW} \cdot I)$$

**临界点** $I_{\text{crit}} = P_{\text{peak}} / \text{BW}$:算术强度超过它就是 compute-bound,达不到就是 memory-bound。

| GPU(FP16 Tensor Core) | $P_{\text{peak}}$ (TFLOPS) | BW (TB/s) | $I_{\text{crit}}$ (FLOPs/byte) |
|---|---|---|---|
| A100 80GB | 312 | 2.0 | **156** |
| H100 SXM | 1979 | 3.35 | **~590** |
| H200 | 1979 | 4.8 | **~412** |
| B200(FP8) | 4500 | 8.0 | **~563** |

→ H100 上,一段代码的算术强度只要小于 ~590,就是 memory-bound——**这是非常高的门槛**。也就是说,**绝大多数 LLM 推理 kernel 都是 memory-bound**。

### 4.3 LLM 各阶段在 Roofline 上的位置

把 GEMM 公式套进 Roofline。FLOPs 来自 §1.2($2MNK$),Bytes 来自 §2.4(主要是读权重 B,因为权重最大):

| 场景 | 形状(M, N, K) | FLOPs | Bytes(主要是权重 B) | 算术强度 $I$ | 受限于 |
|---|---|---|---|---|---|
| Prefill 一个 FFN | M=2048, N=14336, K=4096 | 2·MNK ≈ 240 GFLOPs | 读 W = K·N·2 ≈ 117 MB | $\approx 2050$ | **算力**(compute-bound) |
| Decode 一个 FFN(batch=1) | M=1, N=14336, K=4096 | 2·MNK ≈ 117 MFLOPs | 读 W ≈ 117 MB | $\approx 1.0$ | **HBM 带宽**(memory-bound) |
| Decode 一个 FFN(batch=64) | M=64, N=14336, K=4096 | 7.5 GFLOPs | 读 W ≈ 117 MB | $\approx 64$ | **仍是带宽**(64 < 590) |
| Decode 一个 FFN(batch=1024) | M=1024, N=14336, K=4096 | 120 GFLOPs | 读 W ≈ 117 MB | $\approx 1024$ | **算力**(compute-bound) |

**两个关键直觉**:

1. **prefill 天然 compute-bound,decode 天然 memory-bound**。原因是 GEMM 的 $M$ 维度差了 2~3 个数量级:prefill 的 M 是 batch × 长 prompt,decode 的 M 是 batch × 1。同一块权重 B,prefill 阶段被反复用了几千次,decode 只被用了 1 次——**算术强度直接差了同样的倍数**。
2. **增大 batch 是把 decode 从 memory-bound 推向 compute-bound 的唯一手段**——这就是 continuous batching / vLLM 存在的根本动机(§12.2)。

> 详细推导见 §4.3「prefill vs decode 的二元性」,这里只埋下论据。

### 4.4 一个典型的踩坑:看到 GPU 利用率 100%,以为已经压满

`nvidia-smi` 显示的 GPU-Util 只反映"有没有 kernel 在跑",**不区分 compute-bound 还是 memory-bound**。

- 跑一个纯访存 kernel,GPU-Util 也是 100%,但 Tensor Core 几乎全空闲,**MFU 可能只有 5%**。
- 真正要看的是 **SM Active / Tensor Active / DRAM Throughput**(Nsight Compute 里能看到)。

→ "GPU 利用率高" ≠ "算力压满"。MFU(§4.4)才是真指标。

---

## 五、Batched GEMM:为什么 batch 维必须编译期定死

### 5.1 GEMM 的 batched 扩展

把同一个矩阵乘对 batch 维堆叠 B 份:

$$C_{b, :, :} = A_{b, :, :} \cdot B_{b, :, :}, \quad b = 0, 1, \dots, B-1$$

GPU 上的实现叫 **batched GEMM** 或 **batched matmul**(cuBLAS `gemmBatched`,PyTorch `bmm`)。它**不是** B 次普通 GEMM——而是把 B 份合并成一个 kernel 调用,共享 launch 开销和 Tensor Core 的初始化。

### 5.2 形状必须固定的硬约束

batched GEMM 在编译/调度阶段就把 $(B, M, N, K)$ 和数据布局烧进 kernel,理由全部回到 §2:

1. **Tensor Core 的 tile 形状是编译期决定的**(M_tile / N_tile / K_tile 是 kernel 模板参数)。
2. **SMEM 的 staging buffer 大小**也是编译期定死的。
3. **CUDA Graph** 一旦录制,所有 tensor shape 不能变。

→ **运行时改 B 或 M = kernel 重新 dispatch / CUDA Graph 重录 = 50~200 ms 延迟尖峰**。

这就是 §12.2 里 continuous batching 不能"想加请求就加请求"、PagedAttention 要把 batch 维改成"逻辑指针"的物理根源——**底层的 batched kernel 不允许形状变化**。

vLLM / SGLang 的实际做法是**预录多档 CUDA Graph**(batch ∈ [1, 2, 4, 8, 16, 32, 64, 128] 等),运行时按当前 batch 数选最近的一档,把"重录代价"摊销到启动期。跨档时(比如 batch=64 突然涨到 65,需要升级到 96 那档)有 50~150 ms 的延迟尖峰;prefill / decode 切换更贵,~200 ms。

---

## 六、把所有概念串起来:一次 decode iteration 的物理图景

以 Llama-3 70B,batch=64,decode 一步 为例:

```
host (CPU)
   │  1. 调度器决定哪些请求进入这一步 batch
   │  2. 找到匹配 batch=64 的 CUDA Graph,replay()
   ▼
GPU
 ┌─ kernel 1: embed lookup       [B=64, 1] → [B=64, 1, d]
 │
 ├─ for layer in 80 layers:
 │    ├─ kernel 2: RMSNorm        elementwise
 │    ├─ kernel 3: QKV GEMM       [64, d] × [d, 3d]  ─→ 形状压 Tensor Core
 │    ├─ kernel 4: RoPE           elementwise
 │    ├─ kernel 5: FlashAttention 读 KV[B=64, layer, ...]
 │    │                            tiling 进 SMEM(论文里称 SRAM),不落 HBM
 │    ├─ kernel 6: O GEMM         [64, d] × [d, d]
 │    ├─ kernel 7: RMSNorm
 │    ├─ kernel 8: FFN-up GEMM    [64, d] × [d, d_ff]  ─→ 最大 GEMM
 │    ├─ kernel 9: SiLU           elementwise
 │    └─ kernel 10: FFN-down GEMM [64, d_ff] × [d_ff, d]
 │
 └─ kernel 11: LM head GEMM       [64, d] × [d, V]
                                    │
                                    ▼
                              [64, V] logits
```

按 Roofline 分类,这一步里:

| kernel 类型 | 占总时间 | bound on |
|---|---|---|
| GEMM(QKV/O/FFN-up/FFN-down/LM head) | ~75% | **HBM 带宽**(decode + batch=64 仍 memory-bound) |
| FlashAttention | ~15% | **HBM 带宽**(读 KV cache) |
| elementwise(norm/RoPE/SiLU) | ~5% | 带宽 + kernel launch |
| 调度 + launch | ~5% | CPU + driver |

→ 整体瓶颈是**显存带宽 + kernel launch**——这是为什么:
- **量化(§12.4)** 立竿见影:权重从 16-bit 压到 8-bit,HBM 读取量直接减半。
- **MoE(§9)** 在 decode 阶段比 dense 更划算:每个 token 只激活一部分 expert,搬运的权重更少。
- **speculative decoding / MTP**(§12.5)有效:多个候选 token 共享一次权重读取。
- **CUDA Graph** 重要:省 kernel launch。

---

## 七、常见踩坑与口诀

1. **「显卡跑满了」≠「Tensor Core 跑满了」**
   - `nvidia-smi` 的 GPU-Util 只看是否有 kernel,看不出 compute / memory bound。要 Nsight Compute 里看 SM Active / Tensor Active / DRAM Throughput。

2. **「FLOPs 大就慢」是错的**
   - FFN-up 比 QKV FLOPs 大,但二者都是 memory-bound 时,**真正决定时间的是权重大小**(HBM 读取量),不是 FLOPs。

3. **「加 batch 一定提速」也错**
   - batch 提升只在 memory-bound 区段有效。一旦推到 compute-bound(算术强度 > 临界值),继续加 batch 只会增大延迟,吞吐不再涨。

4. **「FP16 比 BF16 算力高」是错的**
   - 它们 Tensor Core 吞吐**完全相同**。差别在数值动态范围:BF16 训练稳定性远好于 FP16。

5. **CUDA Graph 不是免费午餐**
   - 一次录制要十几毫秒。频繁重录(动态 shape、动态 batch)反而会拖慢。所以 vLLM 才需要预录多档。

6. **算术强度只看 GEMM 不够**
   - 严格的 Roofline 还要算 attention(KV 读取)、norm(elementwise 全是带宽)。decode 阶段 attention 的 KV 读取常常和权重读取量级相当。

---

## 八、口袋公式表

| 公式 | 用途 |
|---|---|
| $\text{FLOPs}_{GEMM} = 2MNK$ | 任意 GEMM 的算力需求 |
| $\text{Bytes}_{GEMM} \approx 2(MN + NK + MK)$(FP16,3 个矩阵都读写) | 算 GEMM 的 HBM 流量 |
| 权重主导的 decode:$\text{Bytes} \approx 2 \cdot N \cdot K$(只读权重) | decode FFN 时 |
| $I = \text{FLOPs} / \text{Bytes}$ | 算术强度 |
| $I_{\text{crit}} = P_{\text{peak}} / \text{BW}$ | 判断 compute / memory bound |
| $P_{\text{achievable}} = \min(P_{\text{peak}}, \text{BW} \cdot I)$ | Roofline 上界 |
| $\text{MFU} = P_{\text{achieved}} / P_{\text{peak}}$ | 实际跑出的算力比例(§4.4) |
| $\text{decode TBW per token} \approx \text{model size in bytes}$ | 一个 batch=1 decode token 要把整个模型从 HBM 读一遍 |
| 临界 batch:$B^* \approx I_{\text{crit}}$(单层 FFN) | 让 decode 从 memory-bound 推到 compute-bound 的 batch 阈值 |

---

## 九、后续章节如何引用本篇

| 章节 | 用到本篇哪部分 |
|---|---|
| §4.3 prefill vs decode | 四、Roofline:prefill compute-bound / decode memory-bound 的判据 |
| §4.4 MFU | 四、Roofline + 八、公式表:MFU = 实测 / 峰值,峰值取本篇 §3 的 TC 吞吐 |
| §5 FlashAttention | 二、内存层级 + 二.四、GEMM 旅程:tiling 进 SMEM(论文称 SRAM)减少 HBM 访问的逻辑同源 |
| §9 MoE | 六、decode 物理图景里"权重搬运量决定时间"——MoE 减权重搬运 |
| §10.2 / §10.3 TP / DP / PP | 二、NVLink/PCIe 带宽数字;通信 ≈ memory-bound 的另一种形式 |
| §12.1 PagedAttention | 五、batched GEMM 形状必须固定 → 才需要把 KV 分块虚拟化 |
| §12.2 Continuous Batching | 五、batched GEMM + 二.五、CUDA Graph 重录代价 |
| §12.3 CUDA Graph | 二.五、kernel launch overhead + 五、shape 固定约束 |
| §12.4 量化 | 三、Tensor Core 不同精度的吞吐表 |
| §12.5 speculative / MTP | 六、多 token 共享一次权重读取 → 把 memory-bound 改善 |

---

> **本篇就到这里。** 再往下(具体 kernel 源码、cuBLAS / CUTLASS 用法、Triton DSL、PTX、warp 级 wmma 指令)属于"GPU 编程"而非"LLM 系统"范畴,本系列不展开。需要时直接读官方文档:
> - cuBLAS / CUTLASS 文档
> - NVIDIA Hopper / Blackwell whitepaper
> - Aleksa Gordić 的 [Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
> - GPU MODE / CUDA MODE 的录播
