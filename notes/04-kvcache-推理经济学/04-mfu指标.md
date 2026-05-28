# 4.4 算力利用率：FLOPs / Arithmetic Intensity / MFU / MBU

[← 返回框架](../../README.md) · [📎 materials.md → §4.4](../../materials.md)

---

## 〇、这一节想回答什么问题

一句话：**你花了几万美元一张的 H100，凭什么相信它真的在干活？**

H100 BF16 峰值算力 ≈ 989 TFLOPs/s（每秒近一千万亿次浮点运算）。但你跑训练，实测往往只能用到 300–500 TFLOPs/s。剩下那一半算力去哪了？是被算法压不出来，还是被带宽卡住，还是通信吃掉了？

这一节就是一套**给 GPU "称重"的工具**：

- **FLOPs**：这个任务一共要做多少计算
- **Arithmetic Intensity**：每搬 1 字节数据能让芯片做多少计算（决定瓶颈）
- **MFU**：实际用上了多少算力
- **MBU**：实际用上了多少带宽
- **HFU**：MFU 的"含重算"变种

---

## 一、最重要的比喻：把 GPU 想成一个工厂

整节都要用这个比喻，先建立直觉。

```
   ┌─────────────────────────────────────────────┐
   │  GPU 工厂                                    │
   │                                              │
   │   仓库 (HBM 显存, ~80 GB)                    │
   │      │                                       │
   │      │   ↕ 卡车 (内存带宽, 3.35 TB/s)        │
   │      ▼                                       │
   │   加工机器 (Tensor Cores, 989 TFLOPs/s)      │
   │                                              │
   └─────────────────────────────────────────────┘
```

- **加工机器**：算力，用 FLOPs/s 衡量
- **卡车**：内存带宽，用 Bytes/s 衡量

每个"工件"（一次计算任务）有两个属性：
- 要做多少**加工**（FLOPs）
- 要搬多少**原料**（Bytes，要从显存读 / 写的字节数）

**永远只有两种瓶颈**：
- 加工重、原料少 → 机器忙、卡车闲 → **compute-bound**（算力不够）
- 加工轻、原料多 → 机器闲、卡车忙 → **memory-bound**（带宽不够）

LLM 训练大批次是前者，LLM 推理 decode 是后者。这条线索贯穿整节。

---

## 二、第一个指标：FLOPs —— 这个任务有多少加工量

FLOPs = **Fl**oating-point **Op**eration**s**，浮点运算总次数。一次乘法或一次加法都算 1 FLOP。

注意区分：**FLOPs**（总次数，s 是复数）vs **FLOPs/s**（每秒次数，速率）。

记两个口诀，能在白板上 30 秒估出一个训练/推理任务的算量：

### 2.1 训练：6ND 法则

$$\text{Train FLOPs} \approx 6 \times N \times D$$

- $N$ = 模型参数量
- $D$ = 训练用的总 token 数
- **6 怎么来**：forward 2N + backward 4N ≈ 6N（每参数大约要做 6 次乘加）

### 2.2 推理 decode：每生成 1 个 token 约 2N FLOPs

$$\text{Decode FLOPs per token} \approx 2N$$

- 只 forward，所以是训练的 1/3
- 生成 1000 token 总算量 ≈ 2000 × N FLOPs

### 2.3 Attention 部分（不在 2N 里）

$$\text{Attn FLOPs per token} \approx 4 \cdot L \cdot d \cdot \text{layers}$$

attention 的算量与序列长度 $L$ 成正比，所以**长 context 下 attention 占比上升**。这正是 FlashAttention、稀疏 attention 等优化想解决的问题。

### 2.4 例子：Llama-3 70B 训练要多久

- $N = 7 \times 10^{10}$，$D = 1.5 \times 10^{13}$
- 总算量 = $6ND \approx 6.3 \times 10^{24}$ FLOPs
- 一张 H100 BF16 峰值 $\approx 10^{15}$ FLOPs/s
- 理论 GPU-秒：$6.3 \times 10^9$ ≈ **2000 GPU-年**
- 1024 张卡：理论 ~2 年；实际 MFU ~40% → ~5 年才能跑完

这条估算路径就是这节指标的实际用法。

---

## 三、第二个指标：Arithmetic Intensity —— 决定瓶颈的关键

很多人卡在这个概念上。它其实就是一句话：

$$I = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{\text{加工量}}{\text{搬运量}}$$

**每搬 1 字节原料，能让机器做多少 FLOPs？**

### 3.1 为什么这个比值决定瓶颈

硬件本身有一个"算力 ÷ 带宽"的比值，叫 **roofline 拐点**：

H100：989 TFLOPs ÷ 3.35 TB/s ≈ **295 FLOPs/byte**

意思是：H100 每送进 1 字节，机器能消化 295 FLOPs。

- 任务 $I > 295$ → 加工太重，卡车送进来芯片消化得完 → **compute-bound**
- 任务 $I < 295$ → 加工太轻，机器在等卡车 → **memory-bound**

（拐点的完整 roofline 模型见 §4.2。）

### 3.2 典型场景的 $I$ 差几个数量级

| 场景 | $I$（FLOPs/byte） | 瓶颈 |
|------|----------|------|
| 训练大 batch 的 GEMM | 几千 | compute-bound ✓ |
| Prefill（一次算整段 prompt） | 几百–几千 | compute-bound ✓ |
| Decode（batch=1，每步 1 token） | **~1** | **重度 memory-bound** ✗ |
| Decode（batch=128） | ~128 | 仍然 memory-bound（< 295） |

→ **这就是为什么 LLM 推理 decode 永远在等带宽，不在等算力。** 也是为什么 KV-cache 量化、GQA、speculative decoding 这些优化都在压"搬运量"——压一字节就直接换 295 字节的算力空间。

### 3.3 Attention 的 $I$（FlashAttention 之后）

$$I_{\text{attn}} \approx \frac{L}{2 \cdot \text{bytes}}$$

序列越长，attention 这一段的 $I$ 越大，越接近 compute-bound。这就是 FlashAttention 在长上下文场景下性能格外亮眼的根因。

---

## 四、第三个指标：MFU —— 机器到底有多忙

**M**odel **F**LOPs **U**tilization，由 [PaLM 论文](https://arxiv.org/abs/2204.02311) 定义。

口语版："你真正用上的算力 / 硬件峰值算力"。

$$\text{MFU} = \frac{\text{实测 FLOPs/s}}{\text{硬件峰值 FLOPs/s}}$$

实战上从训练日志里反推：

$$\text{MFU} = \frac{\text{实测 tokens/s} \times \text{每 token 理论 FLOPs}}{\text{GPU 数} \times \text{峰值 FLOPs}}$$

### 4.1 工业典型值（要背下来）

**训练**：

| 模型 / 设置 | MFU |
|------|------|
| GPT-3 175B（OpenAI 2020 估算）| ~20% |
| PaLM 540B（Google 2022 TPU）| 46% |
| Llama-2 70B（Meta, A100）| ~50% |
| Llama-3 70B / 405B（H100）| 38–43% |
| DeepSeek-V3 FP8（H800）| ~30%（FP8 峰值大 → 实际 TF/s 更高）|

**推理**：

- prefill：30–50%
- decode：通常 5–15%

### 4.2 关键认知：decode 的低 MFU 不是问题

decode memory-bound，**机器本来就该闲着**。用 MFU 评价 decode，等于用"工厂机器使用率"评价一家快递公司——指标错了。

要看的是 **MBU**。

---

## 五、第四个指标：MBU —— 卡车到底有多忙

**M**emory **B**andwidth **U**tilization：你真正吃到的带宽占硬件峰值的百分比。

$$\text{MBU} = \frac{\text{实测 Bytes/s}}{\text{硬件峰值带宽}}$$

不同阶段"做得好"长这样：

| 阶段 | MFU | MBU |
|------|------|------|
| Training | 40% | ~50% |
| Prefill | ~35% | ~30% |
| **Decode** | **<10%（正常）** | **>70%（达标）** |

Decode MBU > 70% 才叫"做对了"。如果 MBU < 50%，说明你浪费带宽（kernel launch 开销、内存不连续、KV gather 低效……）。

---

## 六、第五个指标：HFU —— 含 recompute 的 MFU

**H**ardware **F**LOPs **U**tilization。当训练用了 **gradient checkpointing**（梯度检查点）省显存时，backward 阶段要把 forward 重算一遍。这些"重算"的 FLOPs 算给了硬件，但不算给"模型本该做的工作"。

- **MFU**：分子只算"模型必需"的 FLOPs（不含重算）
- **HFU**：分子算硬件实际做的全部 FLOPs（含重算）
- **HFU > MFU** 永远成立

举例：MFU 40%、重算多占 30% 算量 → HFU ≈ 52%。

判断"算法效率"看 MFU；判断"硬件忙不忙"看 HFU。

---

## 七、训练 vs 推理：到底该看哪个指标

| 场景 | 主要看 | 理由 |
|------|--------|------|
| 训练 | **MFU** | 大 GEMM，compute-bound，MFU 直接反映效率 |
| Prefill | MFU 为主 | 同训练 |
| **Decode** | **MBU** | memory-bound，MFU 低是物理决定的 |

> **别用 MFU 评价 decode，就像别用油耗评价电动车。**

---

## 八、走一遍完整例子：Llama-3 70B 在 H100 上 decode

任务：BF16，batch=1，每生成 1 个 token。

**算量**：$2N$ FLOPs ≈ 140 GFLOPs

**搬运量**：要把全部参数读一遍 $N \times 2$ bytes = 140 GB（先忽略 KV）

$$I = \frac{140 \times 10^9}{140 \times 10^9} = 1 \text{ FLOP/byte}$$

对比 H100 拐点 295 → **重度 memory-bound**。

预测时间（取两条路径里更慢的那条）：

- 算力路径：140 GFLOPs ÷ 989 TFLOPs/s ≈ **0.14 ms**（远没用上）
- 带宽路径：140 GB ÷ 3.35 TB/s ≈ **42 ms** ← 实际下限

实测 50 ms？

$$\text{MBU} = \frac{42}{50} \approx 84\%$$

非常接近理论上限，做得很好。

同时 MFU 大约只有：

$$\text{MFU} \approx \frac{140 \text{ GFLOPs}/50 \text{ ms}}{989 \text{ TFLOPs/s}} \approx 0.3\%$$

——但这**不是工程烂，是物理决定的**。所有人在这个硬件上 decode 单 token MFU 都只能这么低。

---

## 九、再走一遍：Llama-3 70B 训练时间反推

公式：
$$T = \frac{6ND}{\text{GPU 数} \times \text{peak FLOPs} \times \text{MFU}}$$

代入 $N = 70$B, $D = 15$T, 1024 张 H100，BF16 989 TFLOPs，MFU 假设 40%：

$$T = \frac{6.3 \times 10^{24}}{1024 \times 989 \times 10^{12} \times 0.4} \approx 1.5 \times 10^7 \text{ s} \approx 175 \text{ 天}$$

→ 想缩短到 70 天，要么 GPU 加到 2500 张，要么 MFU 拉到 50%+，要么换 FP8。

---

## 关键问答

**Q1**：为什么是 6ND，不是别的数？
- forward：每参数大约要做 1 乘 1 加 = 2 ops → 2N FLOPs/token
- backward：~2× forward = 4N FLOPs/token（链式法则要算两套梯度）
- 合计 6N FLOPs/token，乘 D tokens = 6ND
- attention 不严格满足（与 $L^2$ 相关），但 $N$ 主导

**Q2**：MFU 30% 算高还是低？
- 训练：30% 中等，>40% 好，>50% 顶级
- prefill：>30% 已不错
- decode：MFU 低不代表系统差，看 MBU

**Q3**：FP8 训练 MFU 反而下降，怎么解释？
- H100 BF16 989 TF → FP8 1979 TF，峰值算力翻倍
- 但带宽没变 → roofline 拐点从 295 翻到 590
- 更多任务掉进 memory-bound 区，MFU% 自然看起来变低
- 但 tokens/s 实际更高 —— "MFU 变低、速度变快"是 FP8 的常态

**Q4**：怎么提升训练 MFU？
- 把 GEMM 做大（增 micro-batch / seq len，让 $I$ 进 compute-bound 区）
- gradient accumulation
- 少用 recompute（多用显存换算力）
- 减通信比例（ZeRO-3 → ZeRO-1/2 + SP）
- 用 FlashAttention 系列

**Q5**：MFU 和训练成本的关系？
- 总成本 ∝ GPU 小时 = $\frac{6ND}{\text{MFU} \times \text{peak FLOPs}}$
- MFU 30% → 45% → 训练成本下降 1/3
- 这就是为什么 PaLM、Llama-3、DeepSeek 都在论文里强调 MFU

**Q6**：怎么算 decode MBU？
- 每 token 读 $\approx N$ bytes（参数）+ KV bytes
- Llama-3 70B BF16 单 token ≈ 140 GB
- H100 HBM 3.35 TB/s → 单 token 理论下限 42 ms
- 实测 50 ms → MBU = 42 / 50 = 84%（接近极限）

**Q7**：MFU 100% 可能吗？
- 不可能。即使理论 GEMM 满载，也有 LayerNorm、Softmax、reduce、通信等无法 100% 占用 tensor core 的部分
- 工业上 60% 已是天花板（小模型 + 简单结构）；LLM 训练 50% 是顶级

---

## 参考资料

- [Kaplan et al. 2020 — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Chowdhery et al. 2022 — PaLM (引入 MFU 定义)](https://arxiv.org/abs/2204.02311)
- [Hoffmann et al. 2022 — Chinchilla](https://arxiv.org/abs/2203.15556)
- [Korthikanti et al. 2022 — Reducing Activation Recomputation in LLMs (Megatron)](https://arxiv.org/abs/2205.05198)
- [Dao et al. 2022 — FlashAttention](https://arxiv.org/abs/2205.14135)
- [Llama-3 技术报告](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 技术报告 (FP8 训练)](https://arxiv.org/abs/2412.19437)
- [Horace He — Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [Eleuther AI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
