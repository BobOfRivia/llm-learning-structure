# 2.4 FFN（SwiGLU / GeGLU）

[← 返回框架](../../README.md) · [📎 materials.md → §2.4](../../materials.md)

---

## 一、FFN 在 Transformer 中的角色

每个 Transformer block：
```
x → Attn → +x → Norm
       ↓
      FFN → +x → Norm → next layer
```

**Attention 做"信息交互"，FFN 做"信息加工"。**

- Attention 让 token 之间交换信息（混合 sequence 维）
- FFN 是逐 token 的两层 MLP（混合 feature 维）
- **FFN 占模型大约 2/3 的参数与 FLOPs**（在 MHA + 4× FFN 配置下）

---

## 二、原版 FFN（Vaswani 2017）

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

- $W_1: d \to 4d$（升维）
- $W_2: 4d \to d$（降维）
- 中间宽度 $d_{ff} = 4d$ 是经验值

**直觉**：升维到一个更大的空间做 ReLU 选择性激活，再投影回去 → 等价于一个查表式的"知识存储"。

> Geva et al. 2021 [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) 论证了 FFN 起的就是 "associative memory" 作用。

---

## 三、激活函数的演化

| 激活 | 公式 | 优点 | 主流采用 |
|------|------|------|----------|
| ReLU | $\max(0, x)$ | 简单 | 原版 Transformer |
| GELU | $x \cdot \Phi(x)$ | 平滑、负区间有梯度 | BERT / GPT-2 / GPT-3 |
| Swish (SiLU) | $x \cdot \sigma(x)$ | 类似 GELU 但更便宜 | EfficientNet |
| **SwiGLU** | gated Swish | 经验上最好 | **Llama / Qwen / DeepSeek / 主流 LLM** |
| GeGLU | gated GELU | 类似 SwiGLU | T5-v1.1 / PaLM |
| ReGLU | gated ReLU | 简单门控 | — |

---

## 四、GLU 家族（核心创新）

**GLU = Gated Linear Unit**（Dauphin 2016），把 FFN 拆成"主路径 × 门控路径"。

### 4.1 标准 GLU 公式

$$\text{GLU}(x) = (W_1 x) \odot \sigma(W_2 x)$$
- $W_1$ 主路径（线性）
- $W_2$ 门控路径（sigmoid 决定开关）
- $\odot$ 是逐元素乘（Hadamard 积）

**关键思想**：不再用一个"固定形状"的激活函数逐点作用，而是让网络自己学一条**动态门**——每个隐藏单元的开关由输入决定。数学上这是把**加法非线性**（$\phi(Wx)$）换成**乘法非线性**（$Wx \odot \phi(Vx)$），后者能表达更多的高阶交互项。

### 4.2 Swish / SiLU 激活函数本身

SwiGLU 里的门用的是 **Swish**（Google, Ramachandran 2017），当 $\beta = 1$ 时又叫 **SiLU**（Sigmoid Linear Unit）：

$$\text{Swish}_\beta(x) = x \cdot \sigma(\beta x), \quad \text{SiLU}(x) = x \cdot \sigma(x)$$

性质速览（对理解 SwiGLU 为什么好很关键）：

| 性质 | ReLU | GELU | SiLU / Swish |
|------|------|------|--------------|
| 光滑（$C^\infty$） | ✗（0 点不可导） | ✓ | ✓ |
| 负区间有梯度 | ✗（死神经元） | ✓（极小） | ✓（明显） |
| 单调 | ✓ | ✗（有小凹陷） | ✗（有小凹陷） |
| 上界 | +∞ | +∞ | +∞ |
| 下界 | 0 | $\approx -0.17$ | $\approx -0.278$ |
| 计算成本 | 1× | ~3×（含 tanh 近似） | ~2×（含 sigmoid） |
| 导数（关键点） | 阶跃 | 涉及 $\text{erf}$ / tanh | $\sigma(x) + x\sigma(x)(1-\sigma(x))$ |

导数（PyTorch 直接可算）：
$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x) \cdot (1 + x(1 - \sigma(x)))$$

在 $x \to -\infty$ 时导数 $\to 0$ 但不真等于 0；在 $x \approx -1.28$ 处取最小值 $\approx -0.278$（这个"小凹陷"提供了**自门控**的负反馈）。

### 4.3 SwiGLU 完整定义（Llama 系）

$$\boxed{\text{SwiGLU}(x) = \text{SiLU}(W_{gate}\,x) \odot (W_{up}\,x)}$$
$$\text{FFN}_{\text{SwiGLU}}(x) = W_{down}\,\bigl(\text{SiLU}(W_{gate}\,x) \odot W_{up}\,x\bigr)$$

由 [Noam Shazeer 2020 — GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) 系统评测提出。原文名句：

> "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

Shazeer 在 T5-base（220M）预训练里比较了 9 种 FFN，SwiGLU 与 GeGLU 在 log-perplexity 上明显最优，SwiGLU 对 GELU 相对损失下降约 **1.37%**（在同 FLOPs 预算下），在 GLUE / SuperGLUE 下游平均提升约 0.5–1.0 分。

之后 Llama-1（2023）在 65B 规模上大规模验证，其后 **Llama-2/3/4、Mistral、Qwen-1/2/3、DeepSeek-V2/V3、Gemma、Yi、Baichuan、Kimi K2** 等主流开源 LLM 几乎全数采用。

### 4.4 完整结构与 PyTorch 参考实现

```
        x  (batch, seq, d_model)
         │
  ┌──────┴──────┐
  ▼             ▼
 W_gate        W_up          (各自 d_model → d_ff, 无 bias)
  │             │
 SiLU           │
  │             │
  └──── × ──────┘            (element-wise Hadamard)
         │
       W_down                (d_ff → d_model, 无 bias)
         │
        out (batch, seq, d_model)
```

Llama / HuggingFace 命名约定：`gate_proj / up_proj / down_proj`。

```python
import torch.nn as nn
import torch.nn.functional as F

class LlamaMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 三个矩阵，Llama 传统上全部 bias=False
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # SwiGLU: SiLU(gate) * up  → down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**注意事项**：
1. 相比原 FFN（2 个矩阵），SwiGLU 有 **3 个矩阵**——这是后面维度换算的根因。
2. Llama/Mistral/Qwen/DeepSeek 都是 **bias-free**（bias 对 SwiGLU 增益极小、去掉后 shard/量化更干净）。Qwen2 曾保留 QKV bias，Qwen3 也把它移除并改用 QK-Norm。
3. 顺序是先 SiLU 再乘，不能反（激活作用在门支路上是定义本身）。

### 4.5 维度调整：$d_{ff} = \tfrac{8}{3}d$ 规则完整推导

**参数量持平**是设计目标——想让 SwiGLU FFN 与原始 4d FFN 参数相当，从而可直接和历史配方对齐。

| 结构 | 矩阵数 | 参数量 |
|------|--------|--------|
| 原 FFN（$d \to 4d \to d$） | 2 | $d \cdot 4d + 4d \cdot d = 8d^2$ |
| SwiGLU（$d \to d_{ff}, d \to d_{ff}, d_{ff} \to d$） | 3 | $3 \cdot d \cdot d_{ff}$ |

令 $3 d \cdot d_{ff} = 8d^2$，解得：

$$d_{ff} = \frac{8}{3}\,d \approx 2.6667\,d$$

**工程习惯**：为了 GPU kernel 对齐（tensor core 通常要 8/16/128 的倍数，TP shard 又需要能被 world_size 整除），Llama 引入 `multiple_of` 参数向上取整：

```python
d_ff = int(8 * d_model / 3)
d_ff = ((d_ff + multiple_of - 1) // multiple_of) * multiple_of   # 通常 multiple_of=256
```

**真实模型的实际值**：

| 模型 | $d_{model}$ | $d_{ff}$ | 比值 | 是否严格 8/3 |
|------|-------------|----------|------|--------------|
| Llama-1 7B | 4096 | 11008 | 2.6875 | 接近 8/3 |
| Llama-2 7B | 4096 | 11008 | 2.6875 | 接近 8/3 |
| Llama-2 70B | 8192 | 28672 | 3.5 | 更宽（补偿 GQA 减少的参数）|
| Llama-3 8B | 4096 | 14336 | 3.5 | 更宽 |
| Llama-3 70B | 8192 | 28672 | 3.5 | 更宽 |
| Mistral 7B | 4096 | 14336 | 3.5 | 更宽 |
| Qwen2.5 7B | 3584 | 18944 | 5.29 | 显著更宽 |
| DeepSeek-V3（共享专家 FFN） | 7168 | 18432 | 2.57 | 接近 8/3 |
| DeepSeek-V3（路由专家 FFN） | 7168 | 2048 | 0.286 | 细粒度小 FFN |

**规律**：Llama-3 及后续开始偏离 8/3（放宽到 3.5–5×），因为 GQA 大幅减少了 attention 参数，把预算多分给 FFN 反而提升更明显。**"$d_{ff} = 8d/3$ 只是参数持平的起点，不是天花板**"。

### 4.6 为什么 SwiGLU 更好——五个层面的解释

**(1) 乘法非线性 > 加法非线性**
$\phi(Wx)$ 只是对线性投影做一次弯曲；$\phi(Wx)\odot(Vx)$ 每个输出单元由两条路径**相乘**——多项式展开中包含 $W_i V_j$ 二阶交互项。表达能力严格更强（Bilinear layer 就是 GLU 去激活的退化，实验也强于纯 FFN）。

**(2) 门控 = 动态特征路由**
$\text{SiLU}(W_{gate}x)$ 可视作一个"软掩码"，每个 token 根据自身内容动态决定哪些隐藏单元被打开、以什么强度打开。这与 MoE 的路由思想同源，只不过 MoE 的门是 top-k 稀疏、SwiGLU 是稠密软门。

**(3) Swish 的自门控性质避免了梯度病态**
- ReLU：负区间梯度=0 → 死神经元；SwiGLU 的门支路走 Swish → 负区间仍有小梯度。
- Sigmoid：饱和快 → 大 $|x|$ 时梯度消失；Swish 在正区间近似 $x$ → 大值梯度稳定。
- 实证：SwiGLU 训练比 GELU-FFN 更稳定，尤其在 fp16/bf16 下。

**(4) 条件数视角（Wang et al. 2024）**
[The Devil is in the Condition Numbers](https://arxiv.org/pdf/2605.20749) 从 Hessian 条件数分析：GLU 结构比非 GLU 结构的 layer-wise Jacobian 更良态，收敛速度可用条件数比值直接量化。这是目前对"SwiGLU 为什么好"最数学的解释之一。

**(5) 实证收益（Shazeer 2020）**

| 变体 | log-PPL (C4) | GLUE 平均 | SuperGLUE |
|------|:---:|:---:|:---:|
| ReLU FFN（基线） | 1.997 | 83.80 | 71.66 |
| GELU FFN | 1.983 | 84.20 | 72.98 |
| Swish FFN | 1.994 | 84.36 | 73.34 |
| Bilinear（无激活门） | 1.960 | 84.79 | 73.35 |
| ReGLU | 1.953 | 84.67 | 73.09 |
| GeGLU | **1.942** | **84.12** | **74.20** |
| **SwiGLU** | **1.944** | **84.36** | **74.56** |

SwiGLU vs GELU：log-PPL 降低约 **0.039（相对 ~1.37%）**，SuperGLUE +1.58 分。Bilinear（无激活的纯门控）也很强，说明**门控本身是主贡献，激活函数选型是二阶**——但 Swish 由于工程可维护性（无 erf 近似歧义）胜出。

### 4.7 GLU 变体全景对比

| 名称 | 门激活 $\phi$ | 公式 | 采用者 |
|------|--------------|------|--------|
| GLU | $\sigma$ | $\sigma(W_g x)\odot(W_u x)$ | Dauphin 原始 |
| Bilinear | 无（恒等） | $(W_g x)\odot(W_u x)$ | 消融实验 |
| ReGLU | ReLU | $\text{ReLU}(W_g x)\odot(W_u x)$ | 少见 |
| GeGLU | GELU | $\text{GELU}(W_g x)\odot(W_u x)$ | T5-v1.1 / PaLM / Gemma-1 |
| **SwiGLU** | Swish/SiLU | $\text{SiLU}(W_g x)\odot(W_u x)$ | **Llama / Mistral / Qwen / DeepSeek** |

**为什么 SwiGLU 胜出 GeGLU**：性能几乎打平（SwiGLU 在某些下游甚至略优），但 SiLU 计算比 GELU 便宜（无需 erf/tanh 近似），且导数形式简单，量化/编译器友好。

### 4.8 工程加速：gate/up 融合

`gate_proj` 和 `up_proj` 输入相同、形状相同，可以**合并成一个大矩阵**做单次 GEMM，然后切两半：

```python
# 融合前（两次 GEMM）
gate = self.gate_proj(x)   # [B, S, d_ff]
up   = self.up_proj(x)     # [B, S, d_ff]

# 融合后（一次 GEMM）
class FusedLlamaMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj   = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
```

**收益**：
- 减少一次 kernel launch 开销（对小 batch/短序列尤其明显）
- 更大的 GEMM 更容易吃满 GPU tensor core（利用率 ↑）
- vLLM / TensorRT-LLM / SGLang / Megatron 默认都做这个融合

**进一步**：`SiLU * mul` 也可以写成一个 fused kernel（Triton / CUDA），把中间张量留在寄存器里、免走 HBM：

- vLLM 有 `silu_and_mul` fused kernel
- Flash-Attn 项目里的 `fused_dense_lib` 提供 `SwiGLUFusedFunction`
- TransformerEngine 提供 fp8 版本的 SwiGLU

### 4.9 现代 LLM 中的 SwiGLU 配置速查

| 模型（2024–2026） | Attention | FFN 结构 | $d_{ff}/d$ |
|-------------------|-----------|----------|-----------|
| Llama-3.1 405B | GQA (8/128) | SwiGLU bias-free | 3.5 |
| Mistral 7B / Mixtral 8×7B | GQA | SwiGLU（每专家）| 3.5 |
| Qwen3-32B | GQA + QK-Norm | SwiGLU bias-free | ~4 |
| DeepSeek-V3 671B | MLA | SwiGLU（256 路由专家 + 1 共享）| 见 4.5 表 |
| Gemma 2 27B | GQA + logits softcap | GeGLU（Google 传统）| 4 |
| Kimi K2 | MLA | SwiGLU | ~3 |

结论：**SwiGLU 已经是 2024 年后开源 LLM 的事实标准，Google 系仍保留 GeGLU 习惯，二者性能上没有决定性差异**。

---

## 五、FFN 的工程细节

### 5.1 参数 / FLOPs 占比

对 dense LLM（MHA 配置）：
- Attention 参数（Q/K/V/O 投影）≈ $4d^2$
- FFN 参数 ≈ $8d^2$（标准）或 $\frac{8 \cdot 3}{2} d^2 = 12d^2$（SwiGLU 持平时）
- **FFN ≈ 2/3 总参数**

每 token FLOPs 同分布。这就是为什么：
- 量化 / 稀疏化优化首要瞄准 FFN
- MoE 把 FFN 改成专家集合（§9）能省 4-8 倍激活参数

### 5.2 显存

FFN 中间激活 $h = d_{ff} \cdot L$，比 attention 的 $L^2$ 在短序列下更耗显存。
- 这是 activation checkpointing 必砍的对象
- FlashAttention 解决的是 attention 的 $L^2$ 显存，**对 FFN 无效**

### 5.3 与量化的关系

- AWQ / GPTQ 对 FFN 的 W_gate / W_up / W_down 都量化
- W_down 的输入是激活，输出是残差路径 → 量化敏感
- SmoothQuant 通过把激活 outlier 转移到权重缓解

---

## 六、MoE 视角的 FFN

把单个 FFN 换成"N 个 FFN + 路由"：
$$\text{MoE-FFN}(x) = \sum_{i \in \text{Top-K}} g_i(x) \cdot \text{FFN}_i(x)$$

- 训练参数 N×（如 256 个专家）
- 推理只激活 K 个（如 8 个）→ 激活参数 = K × FFN 参数
- DeepSeek-V3 的 256 专家 / 激活 8 即此模式（§9 详）

---

## 关键问答

**Q1**：FFN 的中间维度为什么是 4d？
- 经验值，Vaswani 2017 直接用的
- 一些研究表明 2d - 8d 都可工作，4d 是精度/计算的折中
- SwiGLU 因为 3 矩阵，常取 2.67d 持平参数

**Q2**：SwiGLU 为什么效果更好（直观解释）？
- 门控提供 multiplicative interaction（乘法非线性比加法非线性表达更强）
- Swish 在负区间不像 ReLU 死，比 GELU 略便宜
- 实证：在 reasoning / math benchmark 上提升明显

**Q3**：FFN 占总 FLOPs 多少？
- dense MHA: FFN ≈ 2/3 总参数和 FLOPs
- GQA: FFN 占比更高（因为 attention 投影变少）
- MoE: 激活 FFN 减少 → attention 占比反而上升

**Q4**：FFN 与"知识存储"的关系？
- Geva et al.（2021）解释 FFN 是 key-value memory：W_1 是 key，W_2 是 value
- 编辑 FFN 权重可以"修改"模型已知的事实（ROME / MEMIT 方法）
- 这是为什么 MoE 用专家 FFN 能水平扩展知识

**Q5**：为什么 SwiGLU 有 3 个矩阵而原 FFN 只有 2 个？
- GLU 需要"主线性路径"和"门控路径"两条独立投影
- 加上输出投影一共 3 个
- 为持平参数，d_ff 缩小到 8d/3

**Q6**：能不能只压 FFN 中间维度？
- 可以，"narrow FFN" 模型（如 GPTQ 等量化方案、低秩 FFN）
- DeepSeek-V3 用细粒度专家把"小 FFN × 多个"实现 sparse compression
- 极端情况下 FFN 可以用低秩分解（SVD-LoRA）替代

**Q7**：SwiGLU 里三个矩阵能不能共享参数？
- `gate_proj` 和 `up_proj` 输入相同、形状相同 → 可以**融合成一个 GEMM**（工程加速），但**不能共享权重**：一旦共享就退化成 $\phi(Wx)\odot(Wx)$，等价于给 Wx 做逐点非线性，门控信息量退化，性能显著下降。
- `down_proj` 与前两者形状不同（转置也不匹配），无共享空间。

**Q8**：训练时 SwiGLU 会不会因为门相乘导致梯度爆炸/消失？
- 门 $\text{SiLU}(W_g x)$ 的值域是 $[-0.278, +\infty)$，乘上 $W_u x$ 后极端值可能放大方差。
- 解决方案：**Pre-Norm（RMSNorm）+ 恰当的初始化**（Llama 用 $\sigma = \sqrt{2/(5d)}$ 的 truncated normal）能让训练稳定到 400B 参数规模。
- 观测：SwiGLU 相比 GELU-FFN 反山更少见 loss spike，因为 SiLU 的负区间"回拉"起到软限流作用。

**Q9**：SwiGLU 和 MoE 的关系？
- MoE 就是"多套 SwiGLU FFN + 稀疏路由"——每个专家本身还是 SwiGLU 结构（Mixtral、DeepSeek、Qwen3-MoE 全部如此）。
- 换个角度：**SwiGLU 内部的门控是稠密软 MoE 的极限（专家数=$d_{ff}$、每个专家 1D）**，MoE 是把门做成 top-k 稀疏 + 每个专家高维。二者是同一思想的连续谱。

**Q10**：为什么现代 SwiGLU 一律 bias-free？
- 参数收益极小（$\ll 0.1\%$），但 bias 会破坏 tensor parallel 的等价性（bias 只在 rank 0 生效）、复杂化 fp8/int4 量化、还需要单独广播。
- Llama-1 起就设为 `bias=False`，后续所有开源模型跟随；Qwen2 是少数保留 QKV bias 的例外，Qwen3 也已删除。

---

## 参考资料

- [Vaswani et al. 2017 — Attention Is All You Need (原版 FFN)](https://arxiv.org/abs/1706.03762)
- [Dauphin et al. 2016 — Language Modeling with Gated Convolutional Networks (GLU)](https://arxiv.org/abs/1612.08083)
- [Shazeer 2020 — GLU Variants Improve Transformer (SwiGLU / GeGLU)](https://arxiv.org/abs/2002.05202)
- [Hendrycks & Gimpel 2016 — Gaussian Error Linear Units (GELU)](https://arxiv.org/abs/1606.08415)
- [Ramachandran et al. 2017 — Searching for Activation Functions (Swish)](https://arxiv.org/abs/1710.05941)
- [Geva et al. 2021 — Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [Meng et al. 2022 — Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262)
- [Llama 2 paper (SwiGLU 工程实践)](https://arxiv.org/abs/2307.09288)
- [Mistral 7B paper](https://arxiv.org/abs/2310.06825)
- [Jianlin Su 博客 — Gated Linear Units 解读](https://kexue.fm/archives/9812)
- [Llama 3 Herd of Models (2024) — SwiGLU 大规模工程实践](https://arxiv.org/abs/2407.21783)
- [Qwen3 Technical Report (2025) — bias-free SwiGLU + QK-Norm](https://arxiv.org/abs/2505.09388)
- [DeepSeek-V3 Technical Report (2024) — SwiGLU + MLA + 细粒度 MoE](https://arxiv.org/abs/2412.19437)
- [Wang et al. 2024 — The Devil is in the Condition Numbers（GLU 为何胜出的数学解释）](https://arxiv.org/abs/2605.20749)
- [vLLM `silu_and_mul` fused kernel（工程实现参考）](https://github.com/vllm-project/vllm)
