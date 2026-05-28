# 10.4 PEFT（LoRA / QLoRA / DoRA / AdaLoRA / Prefix / 全家桶）

[← 返回框架](../../README.md) · [📎 materials.md → §10.4](../../materials.md)

---

## 〇、本节回答什么

> 不能做 full fine-tune（显存 / 卡数 / 数据不够），如何用 < 1% 参数得到 ~90% full FT 效果？

PEFT (Parameter-Efficient Fine-Tuning) 全家桶：
- **加法系**：Adapter、(IA)³
- **重参化系**：**LoRA / QLoRA / DoRA / AdaLoRA / VeRA / LoRA+**
- **prompt 系**：Prefix-Tuning、P-Tuning v2、Prompt-Tuning
- 工业绝对主流：**LoRA + 衍生**（90%+ 场景）

---

## 一、为什么需要 PEFT

Full fine-tune 一个 70B 模型显存账（参考 §10.3）：
- weight 140 GB + grad 140 GB + optimizer state 560 GB = 840 GB
- 单卡 80GB H100 → 至少 16-32 卡

PEFT 把可训练参数压缩到 0.1-1%：
- 70B 模型 → trainable ~70M
- 显存：单卡即可（24-80 GB）
- 训练速度 ↑ 2-5×
- 多任务时，可保留一份 base + N 份 small adapter

---

## 二、LoRA（绝对主角）

[**LoRA: Low-Rank Adaptation of LLMs** (Hu et al. 2021)](https://arxiv.org/abs/2106.09685)

### 2.1 核心数学

冻结原权重 $W_0 \in \mathbb{R}^{d \times d}$，加一个低秩增量：

$$
W = W_0 + \Delta W = W_0 + B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

其中 $r \ll d$（典型 r=8, 16, 32, 64）。

forward：

$$
y = W_0 x + \frac{\alpha}{r} B A x
$$

$\alpha$ 是 scaling，控制 LoRA 强度（典型 $\alpha = 2r$ 或 $\alpha = 16$）。

初始化：
- $A \sim \mathcal{N}(0, \sigma^2)$
- $B = 0$
- → 训练开始时 $BA = 0$，不影响原模型行为

### 2.2 直觉：低秩增量假设

> Fine-tuning 改变的"语义方向"维度很低（intrinsic rank ≪ d）。

实证：r=8 在多数下游任务上接近 full FT 质量。

### 2.3 应用到哪些层？

经典：仅 Q, V 投影矩阵（[原 paper](https://arxiv.org/abs/2106.09685) 推荐）。

工业 2024 主流：**全部 4 个 attention 矩阵 + 3 个 FFN 矩阵都加 LoRA**（Q/K/V/O + W_up/W_gate/W_down）—— 简称 "all-linear"。

```
模型层  →  哪些加 LoRA
─────────────────────────────────
Q,K,V,O  →  ✅
W_up     →  ✅ (gated)
W_gate   →  ✅ (gated)
W_down   →  ✅
embedding →  ❌ (一般不加)
LM head  →  ❌ (一般不加)
norm     →  ❌
```

### 2.4 LoRA 参数量

每个 d×d 矩阵 → trainable = 2dr。

例：Llama-3-70B，d=8192，r=16，全部 7 层都加：
- 80 layers × 7 matrices × 2 × 8192 × 16 ≈ 147M
- vs 70B full → 仅 0.21%

### 2.5 LoRA + 推理合并

训练完可以把 LoRA "**合并回**" base weight：

$$
W_\text{merged} = W_0 + \frac{\alpha}{r} BA
$$

→ 推理时与原模型完全等价，**零额外开销**。这是 LoRA 与 Adapter 的关键差别。

### 2.6 LoRA Hyperparameter 经验

| param | 默认 | 备注 |
|-------|------|------|
| r | 8 / 16 / 32 | 数据多 / 任务难 → 大 |
| α | 16 / 32 | α / r ≈ 1-2 |
| dropout | 0.05-0.1 | 防过拟 |
| target | all-linear | 比仅 QV 好 |
| LR | 1e-4 ~ 5e-4 | 远高于 full FT 的 1e-5 |

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
- Full BF16: 140 GB weight + ... 
- NF4: 70B × 0.5B = **35 GB weight** + LoRA ~0.3 GB

**单卡 48GB 即可微调 70B 模型**（消费级 RTX 6000 / A40 / L40）→ 直接打开开源社区。

### 3.3 三个关键技术

1. **NF4**：理论上对正态分布最优的 4-bit 量化（信息论推导）
2. **Double Quantization**：连量化 scale 都再量化
3. **Paged Optimizer**：把 optimizer state 在 CPU/GPU 间分页（防 OOM spike）

### 3.4 QLoRA 的代价

- 速度比 LoRA 慢 ~30%（dequant 开销）
- 质量略低于 LoRA（量化误差），但差距很小

### 3.5 后续演化

- AWQ-Q (Activation-Aware QLoRA)：用 AWQ 量化代替 NF4
- HQQ / EETQ：更快的量化 backend
- 2024 工业：QLoRA 仍是消费级 / 中小公司主流

---

## 四、DoRA：Weight-Decomposed LoRA

[**DoRA** (Liu et al. 2024)](https://arxiv.org/abs/2402.09353)

把权重分解成**方向 + 幅值**：

$$
W = m \cdot \frac{V}{\|V\|}, \quad V = W_0 + BA
$$

- $m$：可训练的标量（每列一个）
- $V$：方向，用 LoRA 改

实测比 LoRA 提 1-2%，几乎免费。已被 [PEFT 库](https://github.com/huggingface/peft) 集成，**2024 Q3 起取代 LoRA 成新默认**。

---

## 五、AdaLoRA / VeRA / LoRA+

### 5.1 [AdaLoRA (Zhang 2023)](https://arxiv.org/abs/2303.10512)

> 不是每层都给 r=8，**让重要的层 r 大、不重要的 r 小**。

通过奇异值修剪动态调整 rank。质量略好但实现复杂，工业用得少。

### 5.2 [VeRA (Kopiczko 2024)](https://arxiv.org/abs/2310.11454)

> 所有层共享相同的随机 A 和 B，只训练 per-layer scaling 向量。

参数量再压 10×。代价：稍弱于 LoRA。

### 5.3 [LoRA+ (Hayou 2024)](https://arxiv.org/abs/2402.12354)

> 把 A 和 B 的 LR 设不同（B 比 A 大 ~16×）。

实验比标准 LoRA 提 1-2%，简单有效。

### 5.4 [PiSSA (2024)](https://arxiv.org/abs/2404.02948)

LoRA 初始化用原 weight 的 SVD 主成分，收敛更快。

---

## 六、其他 PEFT 流派

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
```

### 7.2 工业 LoRA 部署模式

- **静态合并**：训完 merge 回 base → 一份完整 model
- **动态 hot-swap**：base 常驻 + 多份 LoRA 在请求时加载（vLLM、SGLang 都支持 **Multi-LoRA Serving**）

### 7.3 LoRA 不能做什么

- **大幅改变模型分布**：例如英文 base → 中文 — LoRA 不够，需要 CPT
- **学非常长尾的新知识**：知识"塞不进"低秩增量
- **降低幻觉率**：本质不是参数容量问题
- 对这些场景，**full FT 或 continued pretraining 才是正确答案**

---

## 八、关键问答

**Q1**：LoRA r 怎么选？
- 简单任务 / 小数据：r=4-8
- 中等：r=16-32
- 复杂（多语言、长链推理）：r=64-128
- 经验：先 r=16 跑通，看 loss 决定是否加大

**Q2**：α 和 r 是怎样的关系？
- 数学上 $\alpha/r$ 是有效缩放
- 经验：α = 2r 或 α = 16 固定
- 调 α 类似调 LR，调 r 类似调容量

**Q3**：LoRA 应该加在哪些层？
- 早期 paper：仅 Q, V
- 2024 主流：**all-linear**（QKVO + FFN 全部）
- 质量提升 1-3%，参数增加约 3× 但仍 < 1%

**Q4**：QLoRA vs LoRA 怎么选？
- 显存够（80GB+ 训 7-13B / 70B 多卡）→ LoRA
- 显存紧 / 消费级 / 70B 单卡 → QLoRA
- 质量差距：QLoRA < 1% PPL

**Q5**：DoRA 真的全面优于 LoRA 吗？
- 大多数任务略好（~1-2% 提升）
- 训练略慢 ~10-15%（多算方向归一）
- 已成 HuggingFace PEFT 默认

**Q6**：LoRA 能不能用于 pretrain？
- 一般不行 —— from-scratch 时 base weight 没意义，应该 full
- 例外：**Sparse Upcycling** 用 LoRA 升 MoE（[ReLoRA (2023)](https://arxiv.org/abs/2307.05695) 探索过）
- 工业上 pretrain = full，PEFT = post-training

**Q7**：多 LoRA 同时 serve 有什么坑？
- 每条请求要选不同 LoRA → batch GEMM 不友好
- vLLM 的 **S-LoRA** / SGLang 用 segmented batched GEMM 解决
- 太多 LoRA（>100）会有 cache miss

---

## 九、本节与其他节关系

```
§10.4 PEFT（本节）
   ├─ 显存 / 量化  ←──  §10.2 混合精度
   ├─ SFT 用法     ──→  §11 后训练
   ├─ Multi-LoRA serving → §12 推理优化
   └─ 蒸馏对照     ──→  §10.5 知识蒸馏
```

---

## 参考资料

- [LoRA (Hu et al. 2021)](https://arxiv.org/abs/2106.09685) ⭐⭐（开山）
- [QLoRA (Dettmers 2023)](https://arxiv.org/abs/2305.14314) ⭐⭐
- [DoRA (Liu 2024)](https://arxiv.org/abs/2402.09353) ⭐
- [AdaLoRA (Zhang 2023)](https://arxiv.org/abs/2303.10512)
- [VeRA (2024)](https://arxiv.org/abs/2310.11454)
- [LoRA+ (Hayou 2024)](https://arxiv.org/abs/2402.12354)
- [PiSSA (2024)](https://arxiv.org/abs/2404.02948)
- [Prefix-Tuning (Li 2021)](https://arxiv.org/abs/2101.00190)
- [P-Tuning v2 (Liu 2021)](https://arxiv.org/abs/2110.07602)
- [Adapter (Houlsby 2019)](https://arxiv.org/abs/1902.00751)
- [ReLoRA (2023)](https://arxiv.org/abs/2307.05695)
- [HuggingFace PEFT 库](https://github.com/huggingface/peft) ⭐⭐
- [S-LoRA — 多 LoRA serving (Sheng 2023)](https://arxiv.org/abs/2311.03285) ⭐
- [Sebastian Raschka — Practical LoRA tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) ⭐⭐
- [Lightning AI — Why DoRA replaces LoRA (2024)](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch)
