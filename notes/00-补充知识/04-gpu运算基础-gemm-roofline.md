# 补充 4：GPU 运算基础（GEMM / Roofline / 内存层级 / Tensor Core）

[← 返回框架](../../README.md)

> 这是一篇**跨章节复用**的工具笔记。LLM 训练 / 推理几乎所有性能问题最终都会落到三个硬件事实上：
> - **GEMM**（通用矩阵乘）是 GPU 上的"主菜",训练 / 推理 95% 以上的算力都在跑它。
> - **Roofline 模型** + **算术强度**(arithmetic intensity)决定一段代码究竟是"算不过来"还是"搬不过来"。
> - **HBM ↔ SRAM ↔ 寄存器** 三级存储 + **Tensor Core** + **kernel launch** 是所有具体优化的物理来源。
>
> 后续章节里会反复出现的术语,**这一篇集中讲清楚**:
> - §4.4 MFU 指标 / Roofline / 内存带宽瓶颈
> - §4.3 prefill compute-bound vs decode memory-bound
> - §5 FlashAttention 为什么要 tiling 进 SRAM
> - §10.2 / §10.3 TP / DP / PP 中的 communication-bound 分析
> - §12.1 PagedAttention(KV 碎片)、§12.2 Continuous Batching 的「batched GEMM 形状必须固定」的硬约束
> - §12.3 CUDA Graph 重录代价
> - §12.4 量化(FP16 / FP8 / FP4)的算力提升来自 Tensor Core 不同精度的吞吐差

---

## 一、为什么所有讨论都从 GEMM 开始

### 1.1 LLM 里 99% 的算力都在跑矩阵乘

一个 Transformer block 的 forward 拆开看,每一步都是矩阵乘:

| 操作 | 形状(忽略 batch) | 算的是什么 |
|---|---|---|
| Embedding lookup | 查表(不是 GEMM) | $V \times d$ 里取一行 |
| QKV 投影 | $[T, d] \times [d, 3d] \to [T, 3d]$ | **GEMM** |
| $QK^\top$ | $[T, h_d] \times [h_d, T] \to [T, T]$ | **GEMM**(per head) |
| softmax × V | $[T, T] \times [T, h_d] \to [T, h_d]$ | **GEMM**(per head) |
| O 投影 | $[T, d] \times [d, d] \to [T, d]$ | **GEMM** |
| FFN up + gate | $[T, d] \times [d, d_{ff}] \to [T, d_{ff}]$ ×2 | **GEMM**(占比最大) |
| FFN down | $[T, d_{ff}] \times [d_{ff}, d] \to [T, d]$ | **GEMM** |
| softmax / LayerNorm / RoPE / 激活 | 逐元素或 reduction | **非** GEMM,统称 "elementwise / norm ops" |
| LM head | $[T, d] \times [d, V] \to [T, V]$ | **GEMM**(最大的一个) |

→ 真正吃 FLOPs 的几乎全是 GEMM,**非 GEMM 部分(LayerNorm、softmax、激活、RoPE)合起来通常 < 5% FLOPs**,但它们在 decode 阶段往往是延迟瓶颈(原因见 §三)。

### 1.2 GEMM 的标准形式

NVIDIA 文档里的 GEMM 是:

$$C = \alpha \cdot A \cdot B + \beta \cdot C$$

- $A$: $[M, K]$
- $B$: $[K, N]$
- $C$: $[M, N]$

**三个维度名字一定要记住**:`M / N / K`。后面几乎所有性能分析都用它们。

| 维度 | 含义 | LLM 里通常对应谁 |
|---|---|---|
| M | 输出行数 / batch×seq 维 | $B \cdot T$(prefill)或 $B$(decode,T=1) |
| N | 输出列数 / 输出隐藏维 | $d_{model}$ 或 $d_{ff}$ |
| K | 收缩维 / 输入隐藏维 | $d_{model}$ 或 $d_{ff}$ |

FLOPs 公式:**每个输出元素做 K 次乘加 = 2K FLOPs**,总 FLOPs:

$$\text{FLOPs}_{GEMM} = 2 \cdot M \cdot N \cdot K$$

> 这是后面 Roofline / 算术强度 / MFU 计算的**唯一基础**。

---

## 二、GPU 的执行模型(只讲到能看懂性能分析为止)

### 2.1 硬件层级

```
GPU
 ├─ SM 0  (Streaming Multiprocessor)
 │   ├─ Tensor Core × 4              ─→ 跑矩阵乘
 │   ├─ CUDA Core × 128              ─→ 跑标量/向量
 │   ├─ Register File (~256 KB)      ─→ 每线程私有,最快
 │   └─ Shared Memory / L1 (~228 KB) ─→ block 内共享,可编程
 ├─ SM 1
 ├─ ...
 ├─ SM 131 (H100 有 132 个)
 ├─ L2 Cache (~50 MB)                ─→ 全 GPU 共享
 └─ HBM (80~192 GB)                  ─→ 显存,最慢
```

延迟与带宽量级(H100 为例):

| 层级 | 容量 | 带宽 | 相对延迟 |
|---|---|---|---|
| 寄存器 | 256 KB/SM | ~33 TB/s/SM | 1 cycle |
| Shared Memory / L1 | 228 KB/SM | ~19 TB/s/SM | ~20 cycles |
| L2 Cache | 50 MB | ~5.5 TB/s | ~200 cycles |
| HBM3 | 80 GB | **3.35 TB/s** | ~500 cycles |
| PCIe 5.0(到 CPU) | — | 64 GB/s | ~微秒级 |
| NVLink 4(到其他 GPU) | — | 900 GB/s | ~微秒级 |

→ **跨级访问的代价跨 5 个数量级**。FlashAttention(§5)、kernel fusion、tiling 全是为了**让中间结果停在 SRAM/寄存器,不要落 HBM**。

### 2.2 软件层级:grid → block → warp → thread

CUDA 把一个 kernel 的执行切成:

```
grid
 ├─ block 0 ─→ 调度到某个 SM 上,block 内所有线程共享 shared memory
 │   ├─ warp 0  (32 个线程,SIMT 同步执行)
 │   ├─ warp 1
 │   └─ ...
 ├─ block 1
 └─ ...
```

记住三件事就够分析性能:

1. **warp = 32 个线程**,SM 调度单位。任何时刻一个 warp 内 32 个线程跑同一条指令(SIMT)。
2. **block 是 SM 占用单位**,block 内可共享 shared memory,跨 block 不能直接通信。
3. **Tensor Core 一次吃一个 fragment**(典型 $16 \times 16$ 矩阵片段),由一个 warp 协作驱动。

→ 这就解释了为什么 GEMM 的 tile 形状常见是 16 / 32 / 64 / 128 的倍数:**对齐 Tensor Core 的硬件粒度**。

### 2.3 kernel launch overhead

CPU 启动一个 GPU kernel 要走一遍 driver + runtime,实测延迟约 **5~20 μs**。

> 这个数字看起来小,但在 decode 阶段(每次 forward 只生成 1 token)、一个 70B 模型一次 forward 要跑 ~80 层 × 5+ 个 kernel(QKV / attn / O / FFN-up / FFN-down / norm / ...),每次 forward 就是 ~400 次 launch,即 ~5 ms 纯调度开销——和真正的 GPU 算时同量级。

解法:

| 方案 | 思路 |
|---|---|
| **kernel fusion** | 把多个小 kernel 编译成一个大 kernel,减少 launch 次数 |
| **CUDA Graph**(§12.2) | 把一整个 forward 录制成图,后续 replay 只交一次给 driver,launch 摊销到接近 0 |
| **persistent kernel** | 一个长寿命 kernel 在 SM 上常驻,自己循环吃任务 |

注意 CUDA Graph 有个硬约束:**所有 tensor 形状必须固定**,任何 shape 变化都得重新录制。这就是 §12.2 里"batched GEMM 的 batch 维 B 必须编译期定死"的物理来源。

---

## 三、Roofline 模型:你的代码是 compute-bound 还是 memory-bound

### 3.1 算术强度(Arithmetic Intensity)

定义:

$$I = \frac{\text{FLOPs}}{\text{Bytes moved from HBM}}$$

**单位 FLOP/byte**。它衡量"每搬一字节数据,顺便算多少次"。

直觉上:

- $I$ **大** → 数据搬一次反复算很多次 → 算力是瓶颈 → **compute-bound**
- $I$ **小** → 数据搬来用一次就扔 → 带宽是瓶颈 → **memory-bound**

### 3.2 Roofline 图

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

**临界点** $I_{\text{crit}} = P_{\text{peak}} / \text{BW}$:

| GPU(FP16 Tensor Core) | $P_{\text{peak}}$ (TFLOPS) | BW (TB/s) | $I_{\text{crit}}$ (FLOPs/byte) |
|---|---|---|---|
| A100 80GB | 312 | 2.0 | **156** |
| H100 SXM | 1979 | 3.35 | **~590** |
| H200 | 1979 | 4.8 | **~412** |
| B200(FP8) | 4500 | 8.0 | **~563** |

→ H100 上,一段代码的算术强度只要小于 ~590,就是 memory-bound——**这是非常高的门槛**。也就是说,**绝大多数 LLM 推理 kernel 都是 memory-bound**。

### 3.3 LLM 各阶段在 Roofline 上的位置

| 场景 | 形状(M, N, K) | FLOPs | Bytes(读 W + A) | 算术强度 $I$ | 受限于 |
|---|---|---|---|---|---|
| Prefill 一个 FFN | M=2048, N=14336, K=4096 | 2·MNK ≈ 240 GFLOPs | 读 W = K·N·2 ≈ 117 MB | $\approx 2050$ | **算力**(compute-bound) |
| Decode 一个 FFN(batch=1) | M=1, N=14336, K=4096 | 2·MNK ≈ 117 MFLOPs | 读 W ≈ 117 MB | $\approx 1.0$ | **HBM 带宽**(memory-bound) |
| Decode 一个 FFN(batch=64) | M=64, N=14336, K=4096 | 7.5 GFLOPs | 读 W ≈ 117 MB | $\approx 64$ | **仍是带宽**(64 < 590) |
| Decode 一个 FFN(batch=1024) | M=1024, N=14336, K=4096 | 120 GFLOPs | 读 W ≈ 117 MB | $\approx 1024$ | **算力**(compute-bound) |

**两个关键直觉**:

1. **prefill 天然 compute-bound,decode 天然 memory-bound**。原因是 GEMM 的 $M$ 维度差了 2~3 个数量级:prefill 的 M 是 batch × 长 prompt,decode 的 M 是 batch × 1。
2. **增大 batch 是把 decode 从 memory-bound 推向 compute-bound 的唯一手段**——这就是 continuous batching / vLLM 存在的根本动机(§12.2)。

> 详细推导见 §4.3「prefill vs decode 的二元性」,这里只埋下论据。

### 3.4 一个典型的踩坑:看到 GPU 利用率 100%,以为已经压满

`nvidia-smi` 显示的 GPU-Util 只反映"有没有 kernel 在跑",**不区分 compute-bound 还是 memory-bound**。

- 跑一个纯访存 kernel,GPU-Util 也是 100%,但 Tensor Core 几乎全空闲,**MFU 可能只有 5%**。
- 真正要看的是 **SM Active / Tensor Active / DRAM Throughput**(Nsight Compute 里能看到)。

→ "GPU 利用率高" ≠ "算力压满"。MFU(§4.4)才是真指标。

---

## 四、Tensor Core 与精度

### 4.1 Tensor Core 是什么

NVIDIA 从 Volta(2017)开始给每个 SM 加了**专用矩阵乘单元**——Tensor Core。它一次做一个**小矩阵乘累加**(MMA, matrix-multiply-accumulate):

$$D_{16 \times 16} = A_{16 \times 16} \cdot B_{16 \times 16} + C_{16 \times 16}$$

(具体 tile 大小随代际变化,H100 上还有 wgmma 异步指令对应 $64 \times N$ 形状)

**关键事实**:Tensor Core 的算力比标量 CUDA Core 高一个数量级。所以一段代码能不能用上 Tensor Core,决定了它能不能接近峰值算力。

### 4.2 各精度的吞吐(以 H100 SXM 为例,单位 TFLOPS)

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

→ 量化(§12.4)把模型从 FP16 压到 FP8/INT8/FP4 的"加速"是这么来的:**同一块 Tensor Core 在低精度下吞吐翻倍**。这不是软件优化,是硬件设计。

### 4.3 各精度的数值特性

| 精度 | 总 bit | 指数 | 尾数 | 动态范围(量级) | LLM 用途 |
|---|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | $\pm 10^{38}$ | master weight、optimizer state |
| TF32 | 19 | 8 | 10 | $\pm 10^{38}$ | A100/H100 上 FP32 的默认替代 |
| BF16 | 16 | 8 | 7 | $\pm 10^{38}$ | **训练首选**(动态范围同 FP32) |
| FP16 | 16 | 5 | 10 | $\pm 10^{4}$ | 早期训练 / 推理(易溢出,要 loss scaling) |
| FP8 E4M3 | 8 | 4 | 3 | $\pm 10^{2}$ | **推理常用 + Hopper 训练** |
| FP8 E5M2 | 8 | 5 | 2 | $\pm 10^{4}$ | 训练 backward gradient 用 |
| FP4 | 4 | 2 | 1 | $\pm 8$ | Blackwell 推理 + 训练实验 |

记忆口诀:**指数定动态范围,尾数定相对精度**。BF16 的指数和 FP32 一样多,所以训练里几乎完全替代 FP16。

> 涉及训练稳定性 / 溢出 / loss scaling 见 §10.5。

---

## 五、Batched GEMM:为什么 batch 维必须编译期定死

### 5.1 GEMM 的 batched 扩展

把同一个矩阵乘对 batch 维堆叠 B 份:

$$C_{b, :, :} = A_{b, :, :} \cdot B_{b, :, :}, \quad b = 0, 1, \dots, B-1$$

GPU 上的实现叫 **batched GEMM** 或 **batched matmul**(cuBLAS `gemmBatched`,PyTorch `bmm`)。它**不是** B 次普通 GEMM——而是把 B 份合并成一个 kernel 调用,共享 launch 开销和 Tensor Core 的初始化。

### 5.2 形状必须固定的硬约束

batched GEMM 在编译/调度阶段就把 $(B, M, N, K)$ 和数据布局烧进 kernel,理由:

1. **Tensor Core 的 tile 形状是编译期决定的**(M_tile / N_tile / K_tile 是 kernel 模板参数)。
2. **shared memory 的 staging buffer 大小**也是编译期定死的。
3. **CUDA Graph** 一旦录制,所有 tensor shape 不能变。

→ **运行时改 B 或 M = kernel 重新 dispatch / CUDA Graph 重录 = 50~200 ms 延迟尖峰**。

这就是 §12.2 里 continuous batching 不能"想加请求就加请求"、PagedAttention 要把 batch 维改成"逻辑指针"的物理根源——**底层的 batched kernel 不允许形状变化**。

### 5.3 一组数字直觉

H100 上,SGLang / vLLM 内部典型的"换 batch 重录 CUDA Graph"开销:

| batch 变化 | 是否需要重录 | 延迟 |
|---|---|---|
| 已录好 batch=64,新请求填满到 64 | **否**(沿用) | ~0 |
| batch=64 → 65(超过已录) | **是**(必须升级到下一档,如 96) | 50~150 ms |
| 跨 prefill / decode 切换 | **是** | ~200 ms |

→ vLLM 的"多档 CUDA Graph"策略(预录 batch=[1, 2, 4, 8, 16, 32, 64, 128] 等)就是把"重录代价"摊销到启动期。

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
 │    │                            tiling 进 SRAM,不落 HBM(§5)
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
| §4.3 prefill vs decode | 三、Roofline:prefill compute-bound / decode memory-bound 的判据 |
| §4.4 MFU | 三、Roofline + 八、公式表:MFU = 实测 / 峰值,峰值取本篇表格中的 TC 吞吐 |
| §5 FlashAttention | 二、内存层级 + 三、tiling 进 SRAM 减少 HBM 访问的逻辑 |
| §9 MoE | 六、decode 物理图景里"权重搬运量决定时间"——MoE 减权重搬运 |
| §10.2 / §10.3 TP / DP / PP | 二、NVLink/PCIe 带宽数字;通信 ≈ memory-bound 的另一种形式 |
| §12.1 PagedAttention | 五、batched GEMM 形状必须固定 → 才需要把 KV 分块虚拟化 |
| §12.2 Continuous Batching | 五、batched GEMM + 二、CUDA Graph 重录代价 |
| §12.3 CUDA Graph | 二、kernel launch overhead + 五、shape 固定约束 |
| §12.4 量化 | 四、Tensor Core 不同精度的吞吐表 |
| §12.5 speculative / MTP | 六、多 token 共享一次权重读取 → 把 memory-bound 改善 |

---

> **本篇就到这里。** 再往下(具体 kernel 源码、cuBLAS / CUTLASS 用法、Triton DSL、PTX、warp 级 wmma 指令)属于"GPU 编程"而非"LLM 系统"范畴,本系列不展开。需要时直接读官方文档:
> - cuBLAS / CUTLASS 文档
> - NVIDIA Hopper / Blackwell whitepaper
> - Aleksa Gordić 的 [Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)
> - GPU MODE / CUDA MODE 的录播
