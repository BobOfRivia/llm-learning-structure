# 11.2 PEFT（LoRA / QLoRA / DoRA / RSLoRA / LongLoRA / 全家桶）

[← 返回框架](../../README.md) · [📎 materials.md → §11.2](../../materials.md)

---

## 〇、本节回答什么

> 不能做 full fine-tune（显存 / 卡数 / 数据不够），如何用 < 1% 参数得到 ~90% full FT 效果？

PEFT (Parameter-Efficient Fine-Tuning) 全家桶：
- **加法系**：Adapter、(IA)³
- **重参化系**：**LoRA / QLoRA / DoRA / AdaLoRA / VeRA / LoRA+ / RSLoRA / LongLoRA / LoftQ**
- **prompt 系**：Prefix-Tuning、P-Tuning v2、Prompt-Tuning
- 工业绝对主流：**LoRA + 衍生**（90%+ 场景）

本节核心要回答：

1. **LoRA 为什么有效？** —— "intrinsic rank" 假设的数学论证
2. **α / r / 初始化 / 加在哪些层** —— LoRA 调参的关键细节背后的数学
3. **DoRA 真的全面优于 LoRA 吗？** —— 方向-幅值分解的原理
4. **RSLoRA / LongLoRA / LoftQ 解决了什么** —— 2024 LoRA 家族新成员
5. **Multi-LoRA 在生产环境怎么 serve？** —— S-LoRA / Punica / LoRAX 工程细节
6. **PEFT 不能做什么** —— 何时该退回到 full FT 或 CPT

---

## 一、为什么需要 PEFT

### 1.1 Full Fine-Tune 显存账

70B 模型（参考 §10.3）：
- bf16 weight: 140 GB
- bf16 grad:   140 GB
- fp32 master + m + v: 280 + 280 + 280 = 840 GB
- 合计 ≈ **1260 GB**（不含 activation）
- 单卡 80GB H100 → 至少 16-32 卡

PEFT 把可训练参数压缩到 0.1-1%：

| 项 | full FT 70B | LoRA r=16 70B |
|----|-------------|--------------|
| trainable param | 70B | ~70M |
| grad (BF16) | 140 GB | 0.14 GB |
| optimizer state (FP32 m+v+master) | 840 GB | 0.84 GB |
| frozen base (BF16) | 140 GB | 140 GB (无需 grad/m/v) |
| **总额** | **~1260 GB** | **~141 GB** |
| 最小卡数 | 16-32 卡 | **2 卡足够**（H100 80GB） |

→ PEFT 让消费级 / 中小公司能 fine-tune 大模型。

### 1.2 PEFT 的其他工业收益

1. **训练速度 ↑ 2-5×**（grad / optimizer 算力小）
2. **多任务时**，可保留一份 base + N 份 small adapter（每份 0.1-1% 大小）
3. **推理可热加载 LoRA**（vLLM Multi-LoRA Serving，§7.2）
4. **存储 / 传输便宜**：一份 100MB adapter 远比 140GB 的 70B 模型友好

---

## 二、LoRA（绝对主角）

[**LoRA: Low-Rank Adaptation of LLMs** (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)

### 2.1 核心数学

冻结原权重 $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$，加一个低秩增量：

$$
W = W_0 + \Delta W = W_0 + B A, \quad B \in \mathbb{R}^{d_\text{out} \times r}, A \in \mathbb{R}^{r \times d_\text{in}}
$$

其中 $r \ll \min(d_\text{out}, d_\text{in})$（典型 r=8, 16, 32, 64）。

forward：

$$
y = W_0 x + \frac{\alpha}{r} B A x
$$

$\alpha$ 是 scaling，控制 LoRA 强度（典型 $\alpha = 2r$ 或 $\alpha = 16$）。

### 2.2 Intrinsic Rank 假设的数学论证

LoRA 假设的源头是 [Aghajanyan et al. 2020 — Intrinsic Dimensionality](https://arxiv.org/abs/2012.13255)：

> Fine-tuning 一个预训练大模型，**实际只在一个低维子空间内移动**。

实验方法：把 fine-tuning 限制在一个随机投影到 $d_\text{int}$ 维子空间里，看多大的 $d_\text{int}$ 能达到 90% 的 full FT 效果。

结果：
- RoBERTa-large（355M 参数）→ $d_\text{int} \approx 200$
- 比 full 参数空间小 6 个数量级
- 越大的 pretrained model，$d_\text{int}$ **越小**（→ "更好的 init 让 fine-tune 子空间更窄"）

**LoRA 的直觉来自这里**：既然 $\Delta W$ 的有效自由度很低，何不直接用低秩矩阵参数化 $\Delta W$？

### 2.3 初始化设计：为什么 A 用 Kaiming，B 用 0

```python
A ~ Kaiming_uniform(in_features=d_in, fan_in=d_in)
B = 0
```

理由：
- **训练开始时 $BA = 0$** → 不改变原模型行为，从 base 处平滑过渡
- 若两者都随机：开始时 $BA$ 是随机扰动，等价于打乱了 base 的参数 → 训练前几步可能崩
- 若 A=0, B 随机：对称，但梯度只能从 $\partial L / \partial A$ 流，B 不接收梯度（B 接收的是 $\partial L / \partial B = (\partial L / \partial y) \cdot A x = 0$）→ 训练初期 B 不动

为什么 A 用 **Kaiming uniform** 而不是 Normal？
- Kaiming uniform 范围 $[-\sqrt{6/d_\text{in}}, \sqrt{6/d_\text{in}}]$，方差与 Normal 一致
- 经验上对小 r（如 8）uniform 的有效样本覆盖更均匀（无极端值）
- 实际两者差距很小，HF PEFT 库默认 Kaiming uniform

### 2.4 直觉：低秩增量的几何意义

$BA$ 是秩 $\le r$ 的矩阵。等价于：

$$
\Delta W \cdot x = B (A x) = B z, \quad z \in \mathbb{R}^r
$$

即：**先把输入 $x$ 投影到 $r$ 维"特征方向"，再升回 $d_\text{out}$ 维**。

→ LoRA 学的不是"哪些权重要改"，而是 **r 个"任务相关方向"**。

实证：r=8 在多数下游任务上接近 full FT 质量。复杂任务（多语言、推理链）需要 r=32-128。

### 2.5 α/r scaling 的数学：rank-stabilized 问题

forward 公式里的 $\alpha / r$ 不是随便加的。考虑：

$$
\Delta W = \frac{\alpha}{r} B A
$$

若 $A$ 和 $B$ 都按 unit-variance 初始化，$BA$ 的元素方差 $\propto r$。

- 朴素 LoRA：$\Delta W \propto \alpha$（**与 r 无关**），所以 $\alpha = 16$ 是固定的，调 r 时不变
- 但这意味着 r=8 和 r=128 训练用同样 effective LR → r=128 时 $A$ 和 $B$ 的更新方差比 r=8 大很多（梯度算出来量级不一样）

**RSLoRA (Rank-Stabilized LoRA, Kalajdzievski 2024)**：把 scaling 改成 $\alpha / \sqrt{r}$：

$$
y = W_0 x + \frac{\alpha}{\sqrt{r}} B A x
$$

效果：
- 大 r（64, 128, 256）下不会崩
- 经验上 r=256 时质量比朴素 LoRA r=64 还好
- HF PEFT 已集成，2024 大数据/复杂任务推荐 RSLoRA

### 2.6 应用到哪些层？

经典：仅 Q, V 投影矩阵（[原 paper](https://arxiv.org/abs/2106.09685) 推荐）。

工业 2024 主流：**全部 4 个 attention 矩阵 + 3 个 FFN 矩阵都加 LoRA**（Q/K/V/O + W_up/W_gate/W_down）—— 简称 "all-linear"。

```
模型层  →  是否加 LoRA
─────────────────────────────────
Q,K,V,O    →  ✅ (all-linear 默认)
W_up       →  ✅ (gated MLP)
W_gate     →  ✅ (gated MLP)
W_down     →  ✅
embedding  →  ❌ (一般不加，参数太大且词表稀疏)
LM head    →  ❌ (一般不加；除非任务改 vocab 才考虑)
norm       →  ❌ (1D 参数，LoRA 没意义)
```

实证（[Sebastian Raschka 实验](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)）：
- 仅 QV: 1× 参数，baseline 质量
- 全 attention (QKVO): 2× 参数，+1-2% 质量
- all-linear: 4-7× 参数，+2-4% 质量

→ 工业 default = all-linear。

### 2.7 LoRA 参数量精算

每个 $d_\text{out} \times d_\text{in}$ 矩阵 → trainable = $d_\text{out} \cdot r + r \cdot d_\text{in} = r (d_\text{out} + d_\text{in})$。

例：Llama-3-70B，d=8192，r=16，全部 7 矩阵都加，80 层：
- 单矩阵: $r (d + d) = 16 \cdot 16384 = 262144 \approx 0.26 M$
- attention 4 个 (QKVO): $4 \times 0.26 = 1.05 M$
- FFN：W_up, W_gate 是 $d \times 4d$，W_down 是 $4d \times d$
  - W_up / W_gate: $16 \cdot (8192 + 28672) = 0.59 M$ each
  - W_down: $16 \cdot (28672 + 8192) = 0.59 M$
  - FFN 3 个: $3 \times 0.59 = 1.77 M$
- 单 layer 总: ~2.8 M
- 80 layer: **~224 M**
- vs 70B full → 仅 0.32%

### 2.8 LoRA + 推理合并（关键优势）

训练完可以把 LoRA "**合并回**" base weight：

$$
W_\text{merged} = W_0 + \frac{\alpha}{r} BA
$$

→ 推理时与原模型完全等价，**零额外开销**。这是 LoRA 与 Adapter 的关键差别。

但 **multi-LoRA serving**（一个 base + 多个 LoRA 动态切换）就不能合并，需要 §7.2 的 BGMV/SGMV kernel。

### 2.9 LoRA Hyperparameter 经验

| param | 默认 | 备注 |
|-------|------|------|
| r | 8 / 16 / 32 / 64 | 数据多 / 任务难 → 大 |
| α | 16 / 32 | α / r ≈ 1-2 (经典) 或 α/√r (RSLoRA) |
| dropout | 0.05-0.1 | 防过拟 |
| target | all-linear | 比仅 QV 好 |
| LR | 1e-4 ~ 5e-4 | 远高于 full FT 的 1e-5 |
| warmup | 50-100 step | LoRA 收敛快，warmup 短 |
| weight decay | 0.0 - 0.01 | 一般比 full FT 弱 |

### 2.10 LoRA vs Full FT：质量 / 遗忘 / 容量

| 维度 | LoRA | Full FT |
|------|------|---------|
| 同任务质量 (SFT) | 95-98% | 100% |
| **灾难遗忘** | **极小**（base 冻结） | 中等（除非加 KL 约束） |
| 容量上限 | 受 r 限制 | 全部 |
| 多任务切换 | 一份 base + N adapter | 每任务一份 70B model |
| 跨域迁移（如英→中） | 弱（< full FT 5-10%） | 强 |
| 训练成本 | 1× | 10-20× |

**经验规则**：
- 任务 ≤ "改风格 / 改格式 / 改 persona" → LoRA 完全够
- 任务 ≥ "学新知识 / 改语言分布" → 考虑 full FT 或 continued pretraining
- 多任务部署 → LoRA 几乎必选

---

## 三、QLoRA：4-bit base + LoRA

[**QLoRA** (Dettmers 2023)](https://arxiv.org/abs/2305.14314)

### 3.1 思想

- base model 用 **NF4** (4-bit Normal Float) 量化冻结
- LoRA 仍 BF16 训练
- 反向传播经过 dequant 算梯度

```
forward:    NF4 base  ──dequant──→ BF16 ──+ BA──→ output
backward:   gradient 仅流到 BA（base 冻结）
```

### 3.2 显存节省

70B 模型：
- Full BF16 base: 140 GB 
- NF4 base: 70B × 0.5B = **35 GB**（4-bit 即 0.5 字节/参数）+ scale ~3 GB = ~38 GB
- LoRA r=16 trainable: ~0.5 GB (BF16) + 1.5 GB (FP32 m+v)
- **单卡 48GB 即可微调 70B 模型**（消费级 RTX 6000 / A40 / L40）

→ QLoRA 直接打开开源社区微调大模型的大门。

### 3.3 NF4 的信息论推导

朴素 INT4 量化把 $[-1, 1]$ 均分 16 个 bin。但神经网络权重**服从近似零均值 Gaussian**：

$$
W \sim \mathcal{N}(0, \sigma^2)
$$

均分 bin 在分布密集区（接近 0）粒度太粗，稀疏区（尾部）粒度过细 → 信息浪费。

**NF4 (Normal Float 4-bit)**：把 bin 边界按标准 Gaussian 的 **分位数**划分：

$$
q_i = \Phi^{-1}\!\left(\frac{i + 0.5}{16}\right), \quad i = 0, 1, \dots, 15
$$

其中 $\Phi^{-1}$ 是标准正态分布的 quantile function。

效果：
- 每个 bin 在 weight 分布下的 **概率质量相等**（信息论最优）
- 相对均匀量化 PPL 降 ~5-10%（同 4-bit 容量）

### 3.4 Double Quantization

NF4 每个 block（典型 64 元素）需要一个 FP32 scale，scale 本身占 $32 / 64 = 0.5$ bit/param。

**Double Quantization**：把这些 scale 再用 8-bit 量化（用一个全局 FP32 scale）。

省下 0.5 - (8/64) ≈ 0.37 bit/param → 70B 模型省 ~3 GB。

### 3.5 Paged Optimizer

CUDA 11.4+ 提供 unified memory paging：optimizer state 在 GPU 满时自动 page 到 CPU。

- 防止 OOM spike（如 long sequence 突然出现）
- 速度比 ZeRO-Offload 快（按需 page）

### 3.6 QLoRA 的代价

- **速度比 LoRA 慢 ~30%**（dequant 开销，每次 forward 都要把 NF4 解回 BF16）
- 质量略低于 LoRA（量化误差），差距通常 < 1% PPL
- backward 时仍要算 base 的 dequant，显存峰值较高

### 3.7 NF4 之后的量化演化

| 方案 | 特点 | 适合 |
|------|------|------|
| NF4 (QLoRA) | 信息论最优 4-bit | 通用，HF 默认 |
| **HQQ** | half-quadratic 优化，速度比 NF4 快 2× | 推理 |
| **EETQ** | INT8 weight + INT16 GEMM | 速度优先 |
| **AWQ + QLoRA** | activation-aware 选哪些 channel 保高精度 | 质量优先 |
| **LoftQ** | 量化 + LoRA 初始化联合优化（解决 NF4 量化误差） | 学术最优 |

[**LoftQ (Li 2023)**](https://arxiv.org/abs/2310.08659)：传统 QLoRA 是先量化再 LoRA 微调，量化误差需要 LoRA 慢慢"补"；LoftQ 在初始化时就用 SVD 让 $W_\text{NF4} + BA \approx W_\text{原}$，收敛更快且质量更高。

---

## 四、DoRA：Weight-Decomposed LoRA

[**DoRA** (Liu et al. 2024)](https://arxiv.org/abs/2402.09353)

### 4.1 核心公式

把权重分解成**方向 + 幅值**：

$$
W = m \odot \frac{V}{\|V\|_c}, \quad V = W_0 + BA
$$

其中：
- $m \in \mathbb{R}^{d_\text{out}}$：可训练的**列向幅值**向量
- $V$：方向矩阵，用 LoRA 改
- $\|V\|_c$：按列取 L2 范数（保持每列单位方向）

### 4.2 初始化：保证与原 W 等价

训练开始时 $BA = 0$，所以 $V = W_0$。为了让初始的 DoRA 输出等于原 $W_0$，必须：

$$
m_\text{init} = \|W_0\|_c
$$

即把 $m$ 初始化为 $W_0$ 每列的 L2 范数。这样开始时：

$$
W = \|W_0\|_c \odot \frac{W_0}{\|W_0\|_c} = W_0 \ \ ✅
$$

### 4.3 直觉

LoRA 把"方向 + 幅值"耦合在一起调（$BA$ 同时改变 $W$ 的方向和模长）。DoRA 把它**解耦**：
- $m$ 独立调整每列的 magnitude
- $BA$ 专注调整方向

实证（[DoRA 论文](https://arxiv.org/abs/2402.09353) Fig. 2）：full FT 时 magnitude 和 direction 的更新是负相关的（解耦的）；LoRA 强行让它们正相关；DoRA 重新匹配了 full FT 的更新模式。

### 4.4 实测比 LoRA 提 1-2%

代价：
- 训练略慢 ~10-15%（多算每列 L2 norm 和归一化）
- 推理可以合并回 base（与 LoRA 一样无开销）
- 已被 [PEFT 库](https://github.com/huggingface/peft) 集成，**2024 Q3 起取代 LoRA 成新默认**

---

## 五、其他 LoRA 变体

### 5.1 [AdaLoRA (Zhang 2023)](https://arxiv.org/abs/2303.10512)

> 不是每层都给 r=8，**让重要的层 r 大、不重要的 r 小**。

把 $BA$ 用 SVD 形式 $P \Lambda Q^T$ 参数化，训练中根据奇异值大小动态修剪 rank。

质量略好（+1%），实现复杂，工业用得少。

### 5.2 [VeRA (Kopiczko 2024)](https://arxiv.org/abs/2310.11454)

> 所有层共享相同的随机 A 和 B，只训练 per-layer scaling 向量。

参数量再压 10× （单层 trainable 仅 $2d$ scaling，无 BA）。代价：稍弱于 LoRA。适合超多 adapter 的极端场景。

### 5.3 [LoRA+ (Hayou 2024)](https://arxiv.org/abs/2402.12354)

> 把 A 和 B 的 LR 设不同（B 比 A 大 ~16×）。

理论分析：在 $r \ll d$ 的 init 下，A 和 B 的最优更新速率不同（B 更新需要"放大"才匹配 A 的信息流）。

实验比标准 LoRA 提 1-2%，简单有效，HF PEFT 已支持。

### 5.4 [PiSSA (Meng 2024)](https://arxiv.org/abs/2404.02948)

LoRA 初始化用原 weight 的 SVD 主成分：

$$
W_0 = U \Sigma V^T \approx U_r \Sigma_r V_r^T + W_\text{residual}
$$

让 $A = \sqrt{\Sigma_r} V_r^T$, $B = U_r \sqrt{\Sigma_r}$，并把 base 改为 $W_\text{residual}$。

效果：LoRA 已经"在 W 的主要方向上"，收敛快 2-3×。

### 5.5 RSLoRA (Rank-Stabilized) — 见 §2.5

### 5.6 [LongLoRA (Chen 2024)](https://arxiv.org/abs/2309.12307)

> LoRA + **Shifted Sparse Attention (S²-Attn)**：让 LoRA 也能 efficient 训长 context。

S²-Attn 把长序列分组（如每 8K 一组），在 attention 内只算组内 + shifted 跨组。

应用：把 Llama-2-7B (4K context) 用 LoRA 扩到 32K / 100K 上下文，单卡 A100 就能跑。

### 5.7 [LoftQ (Li 2023)](https://arxiv.org/abs/2310.08659) — 见 §3.7

### 5.8 [X-LoRA (Buehler 2024)](https://arxiv.org/abs/2402.07148)

> Mixture-of-LoRA：多个 LoRA 通过 router 在推理时动态混合。

类似 MoE 思想用于 LoRA。适合"多专业领域同时支持"的场景，开源社区在尝试。

---

## 六、其他 PEFT 流派（已逐渐淘汰）

### 6.1 Adapter

[Houlsby 2019](https://arxiv.org/abs/1902.00751)：在每层加 bottleneck MLP。

```
x ── attention ── + adapter1 ── FFN ── + adapter2 ── out
```

缺点：**推理时无法合并**，引入 latency。已被 LoRA 取代。

### 6.2 (IA)³

[Liu 2022](https://arxiv.org/abs/2205.05638)：每层加一个 element-wise scale 向量。极少参数（< 0.01%），质量略弱。

### 6.3 Prefix-Tuning / P-Tuning v2

[Li 2021](https://arxiv.org/abs/2101.00190) / [Liu 2021](https://arxiv.org/abs/2110.07602)：

> 在每层 attention 的 K, V 前面插入可训练的 "soft tokens"。

```
attention:  Q · [P_K; K]^T   →  attend to prefix
```

适合 prompt-only 微调；推理时也无法合并（占 KV cache）。LLM 时代基本被 LoRA 淘汰。

### 6.4 BitFit

只 fine-tune bias 项。极简但质量明显弱，几乎不用。

---

## 七、PEFT 在 LLM 工业的真实场景

### 7.1 主流用途

```
SFT (supervised fine-tune):       LoRA / QLoRA  (绝大多数开源模型)
RLHF / DPO 微调:                  LoRA / DoRA   (节省显存)
领域适配 (domain adaptation):     LoRA + 中等量数据
多任务 multi-task:                每任务一份 LoRA + 共享 base
角色 / persona:                   LoRA (推理时 hot-swap)
长上下文扩展:                     LongLoRA
量化部署 + 微调:                  QLoRA / LoftQ
```

### 7.2 Multi-LoRA Serving 工程详解

**场景**：一个 base 模型 + 100 份 LoRA（如每个客户一份 LoRA），如何高效 batch 推理？

#### 7.2.1 朴素方案的问题

```
请求 batch: [req_1 (用 LoRA_A), req_2 (用 LoRA_B), req_3 (用 LoRA_A), ...]
朴素做法:   每个请求单独 forward
            → 完全没有 batch 收益
            → GPU 利用率 < 10%
```

如果 "把同 LoRA 的请求合并 batch"：
- 100 个 LoRA → 平均每个 batch 只能放 ~10 个请求
- 还是远低于 vLLM 的 256 batch size

#### 7.2.2 S-LoRA: Unified Memory Pool

[S-LoRA (Sheng 2023)](https://arxiv.org/abs/2311.03285)：

- **Unified Paging**：所有 LoRA 的 A, B 矩阵存到一个统一的 GPU memory pool，按页（page）管理
- **支持动态加载 / 卸载** LoRA（类似 KV cache 的分页管理）
- 单 GPU 可以承载 1000+ LoRA adapter

#### 7.2.3 Punica: BGMV / SGMV Kernel

[Punica (Chen 2023)](https://arxiv.org/abs/2310.18547) 提出两种 kernel：

**BGMV (Batched Gather Matrix-Vector multiplication)**：

$$
y_i = W_0 x_i + \frac{\alpha}{r} B_{\sigma(i)} A_{\sigma(i)} x_i
$$

其中 $\sigma(i)$ 是 batch 内第 $i$ 个请求使用的 LoRA index。

实现：用一个 fused CUDA kernel，把 "查表 + LoRA matmul" 合并，每个请求并行算自己的 LoRA。

**SGMV (Segmented Gather Matrix-Vector)**：对同 LoRA 的请求段内做标准 GEMM，跨段切换。

效果：
- 一个 batch 内多个不同 LoRA 几乎和单 LoRA 一样快
- vLLM、SGLang 都基于 Punica 的 kernel

#### 7.2.4 LoRAX

[LoRAX (Predibase 2024)](https://github.com/predibase/lorax)：商用化 multi-LoRA serving 框架，支持：
- 自动从 HuggingFace Hub 拉取 LoRA
- 动态 swap（请求级别选择 adapter）
- 内置 metrics / monitoring

#### 7.2.5 vLLM / SGLang 现状

| 框架 | Multi-LoRA 支持 | Kernel | 上限 |
|------|----------------|--------|------|
| **vLLM** | ✅ (v0.3+) | Punica BGMV | ~100 LoRA / GPU |
| **SGLang** | ✅ | 自研 + Punica | ~100 LoRA / GPU |
| **TensorRT-LLM** | ✅ (v0.9+) | NVIDIA 自研 | ~50 LoRA / GPU |
| **TGI** | ✅ | 调用 Punica | ~50 LoRA / GPU |

### 7.3 LoRA 不能做什么

- **大幅改变模型分布**：例如英文 base → 中文，LoRA 不够，需要 CPT
- **学非常长尾的新知识**：知识"塞不进"低秩增量
- **降低幻觉率**：本质不是参数容量问题
- **改 tokenizer**：必须重训 embedding
- **训出 reasoning 能力**（如 R1 级别）：需要 RL + reward 模型，PEFT 只能学已有能力

对这些场景，**full FT 或 continued pretraining 才是正确答案**。

### 7.4 RLHF / DPO + LoRA 实战

[**LoRA + DPO**](https://arxiv.org/abs/2305.18290) 已是开源对齐主流：
- base 模型加 LoRA → 用 DPO loss 微调
- 显存：与 SFT-LoRA 相当（DPO 需要 frozen ref model，但 ref = base，可共用）
- 质量：略低于 full FT DPO，但相差很小

例：[Zephyr-7B-β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) 用 QLoRA + DPO 训出来。

详见 §11 后训练。

### 7.5 PEFT 决策表

| 场景 | 数据量 | 显存 | 推荐 |
|------|-------|------|------|
| 单任务 SFT | < 10k | 任意 | LoRA r=8-16 |
| 单任务 SFT | 10k-100k | 任意 | LoRA r=32 + DoRA |
| 多语言扩展 | 100k+ | ≥ full | Full FT + KL 约束 |
| 领域 CPT | 1B+ tokens | ≥ full | Continued Pretraining (full) |
| 长 context 扩展 | < 100k | 单卡 | LongLoRA |
| 多角色 / persona | 任意 | 单卡 base | 每角色一份 LoRA + S-LoRA serving |
| RLHF / DPO | 50k+ | 中等 | QLoRA + DPO |
| Reasoning 训练 | RL | 多卡 | Full FT + RL，PEFT 不够 |

---

## 八、关键问答

**Q1**：LoRA r 怎么选？
- 简单任务 / 小数据：r=4-8
- 中等：r=16-32
- 复杂（多语言、长链推理）：r=64-128
- 大 r（>64）建议用 **RSLoRA**（α/√r）防崩
- 经验：先 r=16 跑通，看 loss 决定是否加大

**Q2**：α 和 r 是怎样的关系？
- 数学上 $\alpha/r$ 是有效缩放
- 朴素 LoRA：α = 2r 或 α = 16 固定
- RSLoRA：α/√r，大 r 时仍稳定
- 调 α 类似调 LR，调 r 类似调容量

**Q3**：LoRA 应该加在哪些层？
- 早期 paper：仅 Q, V
- 2024 主流：**all-linear**（QKVO + FFN 全部）
- 质量提升 1-3%，参数增加约 3× 但仍 < 1%

**Q4**：QLoRA vs LoRA 怎么选？
- 显存够（80GB+ 训 7-13B / 70B 多卡）→ LoRA
- 显存紧 / 消费级 / 70B 单卡 → QLoRA
- 质量差距：QLoRA < 1% PPL
- 想兼具：LoftQ（量化 + LoRA 联合初始化）

**Q5**：DoRA 真的全面优于 LoRA 吗？
- 大多数任务略好（~1-2% 提升）
- 训练略慢 ~10-15%（多算方向归一）
- 已成 HuggingFace PEFT 默认
- 推理可合并，无额外开销

**Q6**：LoRA 能不能用于 pretrain？
- 一般不行 —— from-scratch 时 base weight 没意义，应该 full
- 例外：**Sparse Upcycling** 用 LoRA 升 MoE（[ReLoRA (2023)](https://arxiv.org/abs/2307.05695) 探索过）
- 工业上 pretrain = full，PEFT = post-training

**Q7**：多 LoRA 同时 serve 有什么坑？
- 每条请求要选不同 LoRA → batch GEMM 不友好
- vLLM 的 **S-LoRA / Punica** 用 BGMV/SGMV kernel 解决
- 上限：单 GPU ~100 LoRA（受 unified memory pool 限制）
- 超过 100 → LoRA 在 CPU / 主机 RAM 缓存，按需 swap

**Q8**：LoRA 微调后会"灾难遗忘" base 能力吗？
- 远小于 full FT
- 因为 base weight 完全冻结，能力都还在 $W_0$
- 只在合并 $W_0 + BA$ 后 forward 行为可能略变
- 但若 $\Delta W$ 很大（r=128, 训很多 epoch），仍可能轻微遗忘

**Q9**：RSLoRA 和 LoRA+ 是同一回事吗？
- 不同。RSLoRA 改 **scaling** ($\alpha/r \to \alpha/\sqrt{r}$)，让大 r 不崩
- LoRA+ 改 **学习率** (A, B 分别用不同 LR)，让 forward 更对称
- 二者可以叠加

**Q10**：LongLoRA 怎么避免长 context 训练的显存爆？
- 用 **S²-Attn**（Shifted Sparse Attention）把 $O(S^2)$ 降到 $O(S)$
- 仅 LoRA 参数训练
- 单卡 A100 可把 7B 模型从 4K 扩到 32K-100K

**Q11**：QLoRA 训完的 LoRA adapter 可以加载到非量化的 base 上推理吗？
- 可以！LoRA adapter 是 BF16/FP32，与 base 的量化方式无关
- 实际工作流：QLoRA 训练（节省显存）→ adapter 加到 BF16 base 推理（无量化损失）

**Q12**：LoRA + DPO 训 reward 模型相对 full FT 损失多少？
- 经验：1-3% reward 准确率下降
- 但 LoRA 训得快、显存省，可以训更多种子 → 总体收益正
- Zephyr / Tulu 等开源对齐模型大量使用 QLoRA + DPO

---

## 九、本节与其他节关系

```
§11.2 PEFT（本节）
   ├─ 显存账       ←──  §10.3 显存四块（16N / Adam 12N）
   ├─ 量化         ←──  §10.2 混合精度（NF4, FP8 量化）
   ├─ SFT 配合     ←──  §11.1 SFT（PEFT 是 SFT 的轻量化分支）
   ├─ DPO + LoRA   ──→  §11.4 DPO 家族
   ├─ RLHF + LoRA  ──→  §11.3 RLHF/PPO
   ├─ Multi-LoRA serving → §12 推理优化（BGMV/SGMV kernel）
   ├─ LongLoRA     ──→  §8 长上下文
   └─ 蒸馏对照     ──→  §10.5 知识蒸馏
```

---

## 参考资料

- [Intrinsic Dimensionality (Aghajanyan 2020)](https://arxiv.org/abs/2012.13255) ⭐（LoRA 的理论基础）
- [LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685) ⭐⭐（开山）
- [QLoRA (Dettmers 2023)](https://arxiv.org/abs/2305.14314) ⭐⭐
- [DoRA (Liu 2024)](https://arxiv.org/abs/2402.09353) ⭐
- [RSLoRA (Kalajdzievski 2024)](https://arxiv.org/abs/2312.03732) ⭐
- [LoRA+ (Hayou 2024)](https://arxiv.org/abs/2402.12354)
- [PiSSA (Meng 2024)](https://arxiv.org/abs/2404.02948)
- [LoftQ (Li 2023)](https://arxiv.org/abs/2310.08659) ⭐
- [LongLoRA (Chen 2024)](https://arxiv.org/abs/2309.12307) ⭐
- [AdaLoRA (Zhang 2023)](https://arxiv.org/abs/2303.10512)
- [VeRA (2024)](https://arxiv.org/abs/2310.11454)
- [X-LoRA (Buehler 2024)](https://arxiv.org/abs/2402.07148)
- [Prefix-Tuning (Li 2021)](https://arxiv.org/abs/2101.00190)
- [P-Tuning v2 (Liu 2021)](https://arxiv.org/abs/2110.07602)
- [Adapter (Houlsby 2019)](https://arxiv.org/abs/1902.00751)
- [ReLoRA (2023)](https://arxiv.org/abs/2307.05695)
- [S-LoRA — 多 LoRA serving (Sheng 2023)](https://arxiv.org/abs/2311.03285) ⭐
- [Punica — BGMV kernel (Chen 2023)](https://arxiv.org/abs/2310.18547) ⭐
- [LoRAX (Predibase 2024)](https://github.com/predibase/lorax)
- [HuggingFace PEFT 库](https://github.com/huggingface/peft) ⭐⭐
- [Sebastian Raschka — Practical LoRA tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) ⭐⭐
- [Lightning AI — Why DoRA replaces LoRA (2024)](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)
- [Zephyr-7B (LoRA + DPO 案例)](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
