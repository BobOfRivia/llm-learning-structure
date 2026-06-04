# 6.2 DeepSeek 稀疏路线（NSA / DSA / CSA + HCA）

[← 返回框架](../../README.md) · [📎 materials.md → §6.2](../../materials.md)

---

## 〇、为什么单列 DeepSeek 一节

2024-2025 年，DeepSeek 在稀疏 attention 上连续放出几篇有工程影响力的工作：

- **NSA**（Native Sparse Attention，2025-02）—— **训练时就稀疏**，可端到端学习
- **DSA**（DeepSeek Sparse Attention，DeepSeek-V3.2 / V3.2-Exp）—— 在 MLA 上叠 sparse
- **CSA / HCA**（Hierarchical / Compressed-Selection）—— 工业部署组件

它们共同把"sparse attention"从"推理后处理 trick"推回**"训练原生组件"**。这是 2025 长 context 路线的核心转折。

```
2020-2023:  Longformer / BigBird (训练时稀疏，没火)
2023-2024:  Streaming-LLM / H2O / Quest (推理稀疏，无需重训)
2025+:      NSA / MoBA / DSA (训练原生稀疏，sub-quadratic 长 context)
2026+:      CSA + HCA (DeepSeek V4 1.6T, 分层压缩+层间交替, 1M context 已成本可控)
```

---

## 一、NSA — Native Sparse Attention（[DeepSeek 2025](https://arxiv.org/abs/2502.11089)）

### 1.1 核心思想

每个 query 同时走**三条并行的稀疏路径**，结果加权求和：

```
                   ┌─→  ① Compression branch（粗粒度全局）
query Q  ──→ Gate ─┼─→  ② Selection branch（细粒度 top-k 块）
                   └─→  ③ Sliding window branch（最近的局部）

       gating 输出三路权重，softmax 归一化 + 求和
```

### 1.2 三个分支详细

**① Compression（压缩分支）**：
- 把 KV 序列按 block（如 32 token）平均池化或线性映射成 1 个"代表"
- 然后 Q 对所有"代表"做 full attention
- 复杂度：$O(L \cdot L / B_{compress})$
- 作用：粗粒度看全局

**② Selection（选择分支）**：
- 基于 compression 分支算出的 attention score，**选 top-k 个 block**
- 对这 k 个 block 内的原始 token 做 full attention
- 复杂度：$O(L \cdot k \cdot B_{sel})$
- 作用：把全局粗信号细化成"我具体要看哪些块"

**③ Sliding Window（滑窗分支）**：
- 类似 §6.1 sliding window，看附近 w 个 token
- 复杂度：$O(L \cdot w)$
- 作用：补全局部细节

### 1.3 三路合并

每个 token 有个**门控 gate**（小 MLP，输入是 query），输出 3 个权重：
$$\text{out} = g_{cmp} \cdot O_{cmp} + g_{sel} \cdot O_{sel} + g_{win} \cdot O_{win}$$

→ 模型自己学"现在该看粗的、细的、还是近的"。

### 1.4 NSA 的关键工程点

- **训练原生稀疏**：从 pretrain step 0 就是这样训出来的（不是后改）
- **Block-aligned**：所有稀疏粒度都对齐 block（如 32 或 64），FA-friendly
- **硬件友好**：3 个分支都能用 Block-sparse FA kernel（GPU SM 占满）
- DeepSeek 自己实测：64k context 下，**端到端速度 ≈ FA full 的 3-9×**

### 1.5 数字

| Context | Full Attention | NSA | 加速比 |
|---------|---------------|-----|--------|
| 8k | 1× | ~1× | ~1×（短时无优势） |
| 32k | 1× | ~3× | 3× |
| 64k | 1× | ~6× | 6× |
| 128k | 1× | ~9× | **9×** ⭐ |

质量与 dense attention 接近，长上下文 retrieval 任务略好。

---

## 二、DSA — DeepSeek Sparse Attention（V3.2 / V3.2-Exp）

DSA 是 DeepSeek-V3.2 系列引入的"在 MLA 上叠 sparse"的方案。

### 2.1 与 NSA 的关系（高层对比）

| 维度 | NSA | DSA |
|------|-----|-----|
| 提出 | 学术论文（2025-02） | V3.2 模型（2025-09） |
| 基础 | MHA / GQA | **MLA（必须走 MQA 模式）** ⭐ |
| 选择粒度 | block-level（32/64 token） | **token-level** |
| 路由结构 | 三分支并联（cmp+sel+win）+ gating | **两段**（lightning indexer → top-k → 标准 attention） |
| Top-k 单位 | k 个 block | k=2048 个 token |
| 是否端到端训练 | 是 | 是（两阶段蒸馏） |

→ 不是"NSA 移植到 MLA"，而是**针对 MLA 重新设计**的稀疏方案。详见 §2.7。

### 2.2 Lightning Indexer：DSA 的"打分器" ⭐ 核心组件

#### 2.2.1 它要解决什么问题

DSA 的最终目标：每个 query 只对**最相关的 k=2048 个 prev token** 算主 attention，把 $O(L^2)$ 降到 $O(L \cdot k)$。

但马上有个**鸡生蛋问题**：

> "最相关"怎么定义？理论上是"主 attention 算出来 softmax 分数最高的那 k 个"。可是为了知道分数，得先算主 attention —— 那就回到 $O(L^2)$ 了，**省不下来**。

朴素方案的死胡同：
- 用主 attention 的 Q·K 选 → 计算量没省
- 滑窗 / 块对齐选（NSA 类）→ 选择规则是写死的，不是"真该看的"
- 随机选 → 质量崩

DSA 的答案：**训一个"便宜的代理打分器"**，它的目标不是算 attention，而是**预测"主 attention 会把高分给谁"**。这个代理就是 lightning indexer。

> 类比：在 1M token 的文档里检索 → 不可能每个 token 都细读（=主 attention）；先用"摘要/索引"快速扫一遍（=indexer），挑出 2048 个最可疑的位置，再对这些位置细读。indexer 是被训练好的"索引卡片"，它的分数应该与"细读后会觉得多重要"高度相关。

#### 2.2.2 公式与每个符号的意思

V3.2 paper Eq. 1：

$$I_{t,s} = \sum_{j=1}^{H_I} w_{t,j}^I \cdot \text{ReLU}\!\left(q_{t,j}^I \cdot k_s^I\right)$$

| 符号 | 含义 |
|------|------|
| $I_{t,s}$ | 标量分数："query $t$ 应该多关注 prev token $s$" |
| $H_I$ | indexer head 数 —— **远小于**主 attention 的 head 数 |
| $q_{t,j}^I \in \mathbb{R}^{d_I}$ | 由当前 hidden $h_t$ 投影出的 indexer query（**和主 attention 的 Q 是两套独立投影**） |
| $k_s^I \in \mathbb{R}^{d_I}$ | 由 prev hidden $h_s$ 投影出的 indexer key |
| $w_{t,j}^I$ | query 端额外学出的 per-head 权重（模型自己挑哪个 indexer head 在当前 query 下更可信） |
| 激活 | **ReLU**（不是 softmax，不是 GELU —— 纯 throughput 考虑） |
| 精度 | **FP8**（q、k 全程 FP8） |

读法：每个 indexer head 算一个 $(q \cdot k)$，过 ReLU 把负相关砍掉，再用学出来的 $w$ 加权求和成一个标量。**比 softmax-attention 简单很多**。

#### 2.2.3 一步 decode 的完整数据流

设当前 decode 到第 $t$ 步，prev 已经有 $L = t-1$ 个 token，cache 里存着所有 $k_s^I$（$s=0..t-1$，每个是 $d_I$ 维 FP8 向量，**非常小**）：

```
  hidden h_t
      │
      ├──→ 主 attention 路径：投影成 Q, KV-latent c_t（MLA）
      │
      └──→ indexer 路径：
            ① 投影 h_t → {q^I_{t,1}, ..., q^I_{t,H_I}} 和权重 {w^I_{t,j}}
            ② 对每个 prev s ∈ [0, t-1]：
                  取出 cached k^I_s
                  算 I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
            ③ 在 L 个分数里取 argtopk(k=2048) → 索引集 S_t
            ④ 把这一步新算的 k^I_t 写入 indexer cache

  主 attention：只对 S_t 里的 2048 个 MLA latent c_s 做标准 attention
       → 输出 u_t → 下一层
```

注意：indexer **必须有自己的 KV cache**（缓存所有历史 $k_s^I$），否则每步都要从 hidden 重新算 → 退化回 $O(L)$ per step。好在 $d_I$ 小 + FP8，这个 cache 几乎可以忽略不计（相对 MLA latent cache 来说）。

#### 2.2.4 它怎么学会"猜得准"——蒸馏目标

indexer 的分数想"代理"主 attention 的注意力分布，所以训练目标就是直接蒸馏。

**target distribution** $p_{t,:}$ 的构造（V3.2 paper）：

1. 用真实的主 attention 算一遍 attention scores（$H_{main}$ 个 head 各一份）
2. **跨 head 求和**：把所有 head 的注意力分数加起来 → 一个长度为 $L$ 的向量
3. **L1 归一化**沿序列维度 → 得到概率分布 $p_{t,:}$（"主 attention 整体上把注意力分给了谁"）

indexer 的输出经 softmax 得到 $\hat p_{t,:}$，损失是：

$$\mathcal{L}_{indexer} = \text{KL}(p_{t,:} \,\|\, \hat p_{t,:})$$

→ 训练完后：**indexer 的 top-k argmax ≈ 主 attention 跨 head 整体注意力的 top-k**。这就是它能"用小代理代替大计算"的根本。

#### 2.2.5 为什么叫 lightning（性能账）

| 维度 | 主 attention | Lightning indexer |
|------|-------------|-------------------|
| Head 数 | 数十 ~ 128 | 少（论文称"small number"） |
| 每 head 维度 | $d_{head}$（如 128） | $d_I$（更小） |
| 激活 | softmax（跨序列归一化） | **ReLU + 加权和**（无归一化） |
| 精度 | BF16 | **FP8** |
| KV cache | MLA latent（已经压缩过） | 仅 $k^I_s$（$d_I$ 维 FP8，更小） |
| 复杂度 | 之前 $O(L^2)$，DSA 后 $O(L \cdot k)$ | $O(L^2)$ 但常数小 1-2 数量级 |

净效果：indexer 多花的远小于主 attention 省下的，L 越大优势越明显。

#### 2.2.6 一句话总结 + 两个常见误区

**正确理解**：

> Lightning indexer 是一对**小型 QK 投影 + ReLU + 学出来的加权和**；通过 **forward KL 蒸馏**，让它输出的分布逼近"主 attention 跨 head 聚合后的真实注意力分布"。训练完后用它的标量分数直接做 top-k，把**主 attention** 的部分从 $O(L^2)$ 降到 $O(L \cdot k)$；indexer 本身仍是 $O(L^2)$ 但常数极小，所以净下来 L 越大越赚。

**常见误区 1**：indexer 是不是就是个"小 attention"（标准 $QK^\top$ → softmax）？

→ **不是**。两处关键差异：

| 标准 attention | Lightning indexer |
|---|---|
| 每 head 算 $q \cdot k$ | ✓ 一样 |
| 跨序列 **softmax** 归一化 | ❌ 改成 **ReLU**（按 token 维度，不归一化） |
| 跨 head 拼接送入 V | 改成 **学出来的权重 $w_{t,j}^I$ 加权求和**，输出**标量** |

- ReLU 比 softmax 便宜得多（没有 exp、没有跨序列 reduce）
- 加权和让 indexer 学会"当前 query 下哪个 indexer head 更可信"
- 输出标量后**直接 top-k**，根本不需要 softmax（softmax 单调，不改变 argtopk）
- **只在训练时**，把 L 个 $I_{t,s}$ 一起 softmax，再和主 attention 目标分布算 KL

→ indexer = **简化版 attention 打分器**，不是标准 QK-softmax。

**常见误区 2**：总复杂度是不是从 $O(L^2)$ 直接降到 $O(L \cdot k)$？

→ **不是**。精确说：

```
                  indexer 成本      +    主 attention 成本
dense:            0                 +    O(L²) · 大常数
                                         (H_main · d_head · BF16 · softmax)

DSA:              O(L²) · 小常数    +    O(L · k) · 大常数
                  (H_I · d_I · FP8        (k=2048 常数 → 线性)
                   · ReLU)
```

| 项目 | 复杂度 | 备注 |
|------|--------|------|
| **主 attention** | $O(L^2) \to O(L \cdot k)$ ⭐ | "降到 $O(L \cdot k)$"指的是这块 |
| **Indexer** | 仍是 $O(L^2)$ | 但常数比主 attention 小 1-2 个数量级 |
| **总复杂度** | 形式上仍 $O(L^2)$ | 实际成本主要是 indexer 的小常数项 |

→ 严格说不是"$O(L^2) \to O(L \cdot k)$"，而是：

> **昂贵的 $O(L^2)$ 大常数项** → 换成 **廉价的 $O(L^2)$ 小常数项 + 线性的主 attention $O(L \cdot k)$**

L 越大优势越明显（128k 下 prefill 省 46%、decode 省 78%），因为指数项的"大常数"被换掉了，只剩 indexer 那个几乎免费的小常数。

> 💡 关于"为什么 KL 散度能直接当 loss 用 / forward vs reverse KL 怎么选"——见 [补充 1：KL 散度作为损失函数](../00-补充知识/01-kl散度作为损失函数.md)

### 2.3 Top-k 选择 + 必须走 MLA 的 MQA 模式

拿到所有 $I_{t,s}$ 之后：

1. **Top-k**：对每个 query $t$，按 $I_{t,s}$ 取分数最高的 k 个 prev token。**默认 k = 2048**
2. **取 MLA latent**：拿这 k 个 token 对应的 MLA latent $c_s$（不是原始 KV）
3. **走 MQA 模式**：选出的 k 个 $c_s$ **被所有 query head 共享**
4. **主 attention**：query 对这 k 个 latent 做标准 attention

复杂度对照：

| 模块 | 复杂度 | 实际开销 |
|------|--------|---------|
| Indexer | $O(L^2)$ | 很小（小 head + FP8 + ReLU） |
| 主 attention | $O(L \cdot k)$，k=2048 常数 | **等价线性** ⭐ |
| 总 | dominated by indexer 的 $L^2$ 常数项 | L 越大，DSA 优势越明显 |

**为什么必须走 MQA 模式**：MLA 的 MHA 模式下每个 head 还是有自己的投影路径；如果让每个 head 各选一份 top-k token，开销会乘 H 倍。走 MQA → 所有 head 共享同一份 latent → top-k **选一次就够**。这是 DSA 工程上能跑起来的硬约束。

### 2.4 两阶段训练：dense warmup → sparse continued

DSA 不能从随机初始化裸训 —— indexer 一开始随机选 token，主模型会被噪声打崩。V3.2 用**两阶段蒸馏**：

**Stage 1 — Dense Warmup（冻主模型，只训 indexer）**

| 项目 | 值 |
|------|---|
| 学习率 | $10^{-3}$ |
| 步数 | 1,000 步 |
| Batch | 16 序列 × 128k token |
| 总 tokens | **2.1 B** |
| 损失 | **KL(indexer score ‖ 主 attention 分布)** |

→ indexer 的训练目标是 **"模仿主 attention 真实的注意力分布"**（蒸馏）。主模型完全冻结。

**Stage 2 — Sparse Continued Training（全模型解冻）**

| 项目 | 值 |
|------|---|
| 学习率 | $7.3 \times 10^{-6}$ |
| 步数 | 15,000 步 |
| Batch | 480 序列 |
| 总 tokens | **943.7 B** ⭐ |
| 主模型损失 | LM loss（标准 next-token） |
| Indexer 损失 | 继续 KL 蒸馏 |
| **关键 trick** | **indexer 输入 detach 出 compute graph** |

为什么 detach：如果不 detach，indexer 选出的 top-k 会通过 LM loss 反传梯度，但 top-k 本身离散不可导 → 梯度乱跑。Detach 后：

- indexer 走独立 KL 蒸馏路径（监督信号是"真实 attention 该在哪")
- 主模型走"在 sparse 路径上的标准 LM loss"
- 两者解耦，训练稳

### 2.5 数字：和 V3.1-Terminus 的对照

**推理成本（128k context, 公开 API 价）**：

| | V3.1-Terminus（dense） | V3.2-Exp（DSA） | 降幅 |
|---|---|---|---|
| Prefill / 1M tokens | ~$0.65 | ~$0.35 | **-46%** |
| Decode / 1M tokens | ~$2.10 | ~$0.45 | **-78%** ⭐ |

→ decode 降得比 prefill 多：decode 是 memory-bound，每步要扫整段 KV cache；DSA 每步只扫 k=2048。

**质量（关键 benchmark）**：

| Benchmark | 变化 |
|---|---|
| MMLU-Pro | 持平（85.0） |
| Codeforces | +75 分 |
| AA-LCR（长 context retrieval, reasoning 模式）| **+4 分** ⭐ |
| HMMT 2025 | -2.5 |
| Humanity's Last Exam | -1.9 |

→ 长 context retrieval 明显更好；hard reasoning 掉 1-2 分。整体"基本持平"，但成本砍掉一半以上。

### 2.6 工程意义

- **MLA + DSA 是 DeepSeek-V3.2 长上下文的核心**：MLA 压 KV 体积，DSA 压 attention 算量
- 推理引擎已开放：DeepSeek 的 SGLang、FlashMLA 都支持 DSA 的 paged KV + lightning indexer kernel
- 开源 kernel 栈：**TileLang** 写的 indexer kernel + **DeepGEMM** 处理 FP8 + FlashMLA 处理主 attention

### 2.7 与其他长上下文方案的关系

```
长 context 的三条独立优化轴：
  ① KV 体积：     MHA → GQA → MLA → DSA
  ② attention 算量：full → sparse (NSA, DSA, MoBA)
  ③ 位置编码：    RoPE → YaRN → NTK → ...
```

DSA = 在 MLA 这个 KV 压缩之上，再做 attention 算量稀疏。两者正交。

### 2.8 为什么 NSA 没落地，DSA 才上车（V3.2-Exp 的过渡定位）

NSA 在 2025-02 放出（ACL 2025 Best Paper），但**此后所有 DeepSeek 在线模型都没上 NSA**，直到 2025-09 的 V3.2-Exp 才用一个**新设计的 DSA** 取而代之。原因不是 NSA 训不出来，而是它跟 V3 的工程栈不合：

| 障碍 | NSA 的预设 | DeepSeek 主线（V2/V3）的现实 |
|------|-----------|----------------------------|
| KV 形态 | 每 head 一份 KV（MHA/GQA） | **MLA**：所有 head 共享一个低秩 latent $c$ |
| 选择粒度 | block-level（32/64 token 一块） | 想要更细的 token 级，避免块边界丢信息 |
| 路由结构 | 三分支并联 + gating + load balance | 太重，训练/推理 kernel 都要写 3 套 |
| 选择函数 | 基于 compression branch 的 score | 想要独立、可量化（FP8）的轻量打分器 |

→ 把 NSA 直接搬到 MLA 上会很别扭：MLA 的 latent 是跨 head 共享的，NSA 那种"每 head 各自选 block"的三分支结构失去意义；硬塞回去等于把 MLA 拆掉。

**DSA 的重新设计**（针对 MLA 重写，而不是 NSA 的"MLA 版"）：

- **Lightning indexer**：少量 head + FP8 算一个轻量打分器，对每个 query 给历史 token 打分（O(L²) 但常数极小）
- **Top-k token 选择**（默认 k=2048）：**token-level**，不再是 block-level
- **走 MLA 的 MQA 模式**：选出的 k 个 latent 被所有 query head 共享，与 MLA 的"一份 latent 走天下"天然吻合
- 砍掉 NSA 的三分支 + gating；只剩 "indexer → 选 token → 标准 attention" 两段

公开数字（V3.2-Exp）：128k 推理成本 ↓60%+、速度 ↑3.5×、显存 ↓70%；评测分数与 V3.1-Terminus 持平。

**为什么称 V3.2-Exp 为"过渡"**：DeepSeek 自己把 V3.2-Exp 定位成 **intermediate / experimental release**，目标是把 DSA 在真实流量下跑稳，**为下一代（V4）打地基**——也就是说 DSA 当前的形态（lightning indexer + top-k）大概率还会继续演化，但"NSA 那套三分支 block routing"基本上已经在 DeepSeek 主线被放弃了。

> NSA 的贡献是 **证明了 native trainable sparse 可行**（学术奠基），DSA 是**在 MLA 工程栈里把它重新做对**（工程落地）。两者不是 v1→v2，而是同一个目标的两条独立实现，DeepSeek 主线选择了 DSA。

---

## 三、CSA / HCA — DeepSeek V4 的混合稀疏架构（2026-04）

> 上面 §2 讲的 DSA 是 V3.2 的方案，**单层 sparse**。到 V4（2026-04-24 放出 V4-Pro 1.6T 和 V4-Flash 284B），DeepSeek 把"sparse attention"升级成**层间分工的两种压缩 attention 交替**：CSA + HCA。这一节讲清楚它们各自怎么工作、为什么要做成两种。

### 3.1 一句话区分

| | CSA（Compressed Sparse Attention） | HCA（Heavily Compressed Attention） |
|---|---|---|
| 压缩率 | **m = 4**（4 个 token 压成 1 个 entry） | **m' = 128**（128 个 token 压成 1 个 entry） |
| 选择方式 | **Lightning indexer + top-k 选 1024 个** | **dense**（对所有压缩 entry 做标准 attention） |
| Sliding window | 128 个最近原始 token | 128 个最近原始 token |
| 1M context 视角 | 250k 个压缩 entry → 选 top-1024 | 7,800 个压缩 entry → 全看 |
| 角色 | 局部 / 中距 细看（low compression + sparse） | 远距 全局 概览（heavy compression + dense） |

**关键设计直觉**：

> 与其用单一压缩率，不如**分两层**：
> - "近的、可能重要的"用轻压缩（CSA, m=4），保留细节，但用 sparse top-k 压住成本
> - "远的、概要就行"的用重压缩（HCA, m'=128），细节少了无所谓，反正只要全局轮廓 —— 既然压完总长才 7.8k，那干脆 **dense attention** 一次性扫光

这是一个"分辨率 × 选择率"的权衡：CSA 高分辨率 + 低选择率；HCA 低分辨率 + 全选。

### 3.2 共同基础：怎么把 m 个 token 压成 1 个 entry

CSA 和 HCA 用同一个压缩机制，差别只在 m。压缩器是**学出来的 soft pooling**（不是简单 mean pooling）：

1. 把 hidden 序列 $H$ 通过两个学出来的矩阵 $W_a^{KV}$、$W_b^{KV}$ 投影出两路 latent
2. 在 m 个 token 的窗口内做 **Hadamard 乘积 + softmax 加权**（learned soft pooling）
3. 重叠窗口的部分按权重求和，得到一个压缩 KV entry

→ 这是个**带学习参数的池化层**，比 mean pooling 强（能学到"该重点压谁"），比标准 attention 便宜（只在 m 个 token 内算）。

### 3.3 CSA 详解：保细节 + 稀疏选

```
                    1M tokens 输入
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   压缩 (m=4)       Lightning        Sliding window
   → 250k entries   Indexer 打分      最近 128 个原始 token
        │               │               │
        │         argtopk(k=1024)       │
        │               │               │
        └──→ 选出 1024 个 entry ←───────┘
                        │
                合并：1024 + 128 = 1152 个 KV
                        │
                  query 做标准 MQA attention
```

每一步：
1. **压缩**：1M token → 250k 个 m=4 的压缩 entry
2. **打分**：lightning indexer（同 §2.2 的机制）对 250k 个 entry 打分
3. **Top-k**：选分数最高的 1024 个压缩 entry
4. **拼接**：1024 个被选 entry + 128 个最近的原始 token（sliding window）
5. **MQA attention**：query 对这 1024+128 个 KV 做标准 MQA

→ **CSA = MLA-MQA + 轻压缩 + DSA 选择 + 滑窗**。本质是 DSA 的强化版，多了一步轻压缩让 indexer 扫的"基本单元"从单 token 变成 4-token chunk。

### 3.4 HCA 详解：重压缩 + 全看

```
                    1M tokens 输入
                        │
        ┌───────────────┴───────────────┐
        │                               │
   压缩 (m'=128)                  Sliding window
   → 7,800 entries                最近 128 个原始 token
        │                               │
        └──→ 全部 7,800 + 128 拼起来 ←──┘
                        │
                 query 做 dense MQA attention
                 （**没有 top-k 选择**）
```

关键差异：
- **没有 lightning indexer，没有 top-k**
- 压完只剩 7.8k entry，对这个长度做 dense attention 已经很便宜了，**写一个 sparse 选择反而是 over-engineering**
- 简单 = 快、稳

→ **HCA = MLA-MQA + 重压缩 + 全 attention + 滑窗**。专门负责"远距离上下文的全局视角"。

### 3.5 层间编排：CSA 和 HCA 怎么混

V4 的 transformer 不是每层都一样，**不同层用不同的 attention**：

**V4-Pro（1.6T）**：
```
Layer 0:  HCA
Layer 1:  HCA            ← 前两层全 HCA，先建立全局视野
Layer 2:  CSA
Layer 3:  HCA
Layer 4:  CSA
Layer 5:  HCA            ← 之后 CSA / HCA 交替
   ...
```

**V4-Flash（284B）**：
```
Layer 0:  Sliding Window only
Layer 1:  Sliding Window only   ← 前两层只看局部
Layer 2:  CSA
Layer 3:  HCA
Layer 4:  CSA
Layer 5:  HCA                   ← 之后交替
   ...
```

为什么这么编排：
- **前两层定基调**：Pro 用 HCA（全局先验），Flash 用 SW（先抓局部，省更多算力）
- **CSA / HCA 交替**：每两层一组 —— CSA 给"局部+中距细节"，HCA 给"远距全局"，叠加后每个 token 同时拿到两个尺度的上下文
- 类比："读论文先看 abstract（HCA）再扫某些段落（CSA）"，两种粒度交错使用

### 3.6 数字：V4-Pro vs V3.2（1M context）

| | DeepSeek-V3.2（DSA） | DeepSeek-V4-Pro（CSA+HCA） | 降幅 |
|---|---|---|---|
| 单 token 推理 FLOPs | 1× | **0.27×** | **-73%** ⭐ |
| KV cache 大小 | 1× | **0.10×** | **-90%** ⭐⭐ |

→ KV cache 砍掉 90%：因为 CSA/HCA 实际缓存的是**压缩后的 entry**，不是原始 KV。比如 1M token 在 HCA 层只存 7.8k entry。

### 3.7 为什么是"两层"而不是单层 / 三层

| 方案 | 问题 |
|------|------|
| 只有 CSA（轻压缩 + sparse） | 远距离信息要靠 indexer 选 1024 个出来，命中率随 L 下降；全局视野不稳 |
| 只有 HCA（重压缩 + dense） | m'=128 太粗，**局部细节**（如代码、数学公式）会被池化掉 |
| 三层（细 / 中 / 粗） | 工程复杂度大涨，调度 + kernel + 训练稳定都难度 ×3；论文/财报里没人这么做 |
| **CSA + HCA 两层** | 一近一远、一密一疏，**最简的能覆盖全频段的组合** ⭐ |

→ "两层混合"是 quality-cost-complexity 三角的当前最优点，V4 是第一个把这套架构落到 1T+ 规模的产品。

### 3.8 与 DSA 的关系

| | DSA（V3.2） | CSA+HCA（V4） |
|---|---|---|
| 基本单元 | 单 token | **压缩 entry**（4 或 128 token） |
| 选择 | indexer + top-k | CSA 用 indexer+top-k；HCA 不选 |
| 层间 | **每层都 DSA** | **CSA / HCA 交替** |
| 1M FLOPs | 1× | 0.27× |
| 1M KV cache | 1× | 0.10× |

→ CSA 可以看作"DSA 在压缩 entry 上的复用"；HCA 是新增的"远距全局通道"。整个 V4 attention = **DSA 思想 × 压缩 × 层间分工**。

---

## 四、和 MoBA 等同期方案对比

| 方案 | 来源 | 时间 | 稀疏 pattern | 训练原生 | 与 MLA 兼容 |
|------|------|------|--------------|---------|-------------|
| **NSA** | DeepSeek | 2025-02 | 三分支（cmp+sel+win） | ✓ | 不直接 |
| **MoBA** | Moonshot | 2025-02 | MoE-style block routing | ✓ | 不直接 |
| **DSA** | DeepSeek | 2025-09 | indexer + top-k token | ✓ | ✓ |
| **CSA+HCA** | DeepSeek V4 | 2026-04 | 双层压缩 + 层间交替（CSA sparse / HCA dense） | ✓ | ✓ |
| H2O / Quest | 学术 | 2023-2024 | 推理时 KV eviction | ✗ | 可 |

→ 2025 的趋势：**训练原生稀疏**是三家共识；但"选择粒度"有分歧——NSA / MoBA 选 block-level，**DSA 选 token-level**（详见 §5）。

---

## 五、token-level vs block-level：选择粒度的演化

稀疏 attention 的"每个 query 选谁"有几种粒度选择：

| 稀疏粒度 | 选择精度 | FA kernel 友好（朴素实现） | 典型代表 |
|----------|---------|----------------|---------|
| **token-level**（每 token 独立选） | 高 | 历史上 ✗，**DSA 之后 ✓**（见 §5.2） | DSA |
| **block-level**（32/64 token 一块） | 中（块内不可分） | ✓ | NSA / MoBA |
| layer-level（整层换 attention 类型） | 低 | ✓ | Streaming-LLM、推理 trick |

### 5.1 为什么 2025 早期共识是 block-level

NSA、MoBA 2025-02 同期工作都选了 block-level，原因清楚：

- block 尺寸（32/64）和 FA 的 tile 天然对齐
- 选 block 后，块内 token 还是**连续内存** → HBM 带宽利用高
- gradient 回传容易：block 内是 dense matmul，FA backward 直接跑
- 反观 token-level：每个 query 选出的 token 集合都不同 → 索引散乱、kernel 难写、HBM 利用差

→ 所以早期的工程哲学是"想稀疏，就 block-sparse"。

### 5.2 DSA 打破共识：token-level 也能高效（2025-09）

DSA 在 V3.2 用 token-level top-k（k=2048）跑通了。它绕开"token-level 不友好"的方式是：

1. **Lightning indexer 本身就是 dense $O(L^2)$ matmul**（不是 block-sparse） → 天然 FA-friendly
2. **选出的 k=2048 个 token 在算主 attention 前 gather 成一块连续 buffer** → 主 attention 在"逻辑上的 2048-token 段"上做 dense FA，**仍然连续**
3. **MLA 让每个 token 只有 1 个 latent**（不是 H 个独立 KV head）→ "选 token" 和 "选 latent" 一一对应，没有 head 间分歧
4. **训练用蒸馏 + detach** 绕开 top-k 不可导（见 §2.4）

→ DSA 同时拿到了 **token-level 的精度** + **block-level 的 FA 友好性**：
- 选择阶段：token-level，灵活
- 计算阶段：在 gather 后的"逻辑 block"上 dense，连续

这是 2025-09 的关键突破——**"token-level 不能做高效 kernel"这条经验法则被推翻**。

### 5.3 V4 的折中：在压缩 entry 上做 token-level

V4 的 CSA 又是另一种思路：

- 先把 4 个原始 token 压成 1 个 entry（learned compressor）
- 再在 entry 层面做 token-level top-k 1024
- 选出的 entry gather 后做 dense attention

→ "entry-level top-k" 介于 NSA 的 64-token block 和 DSA 的 1-token 之间。压完之后总数变少（1M → 250k），indexer 打分更便宜，命中也更稳。

### 5.4 结论

> ✗ "block-level 是工程主流"（旧说法，2025-02 时正确）  
> ✓ "block-level 在 FA 朴素实现下友好；DSA 用 gather + dense 在 token-level 也做到了同样的友好性"（2025-09 后的正确认识）

按时间线看：

```
2025-02   NSA / MoBA          block-level（32/64 token / block）
2025-09   DSA                 token-level（1 token，k=2048）
2026-04   V4 CSA              entry-level（4 token 压成 1 entry，再 top-k 1024）
                              + V4 HCA dense over heavy-compressed entries
```

粒度从 block → token → 压缩 entry，每一代都在"精度 vs 工程开销"的权衡曲线上往前挪一格。

---

## 六、训练时稀疏的工程难点

NSA paper 强调"native training"，背后的工程问题：

1. **三分支训练稳定**：gate 一开始随机，怎么不塌缩到单分支？
   - NSA: 初始化时 3 路均匀 + auxiliary load balancing loss（类似 MoE）

2. **稀疏 selection 不可导**：top-k 选 block 是离散的
   - NSA: 让 selection 走 straight-through estimator；compression 分支始终 dense（梯度走那里）

3. **kernel 必须支持 block-sparse forward + backward**
   - DeepSeek 开源了 NSA kernel（基于 Triton）
   - 这是 NSA 落地的关键

---

## 关键问答

**Q1**：NSA 和 MoBA 谁更主流？
- 都是 2025-02 同期工作，思路相似（block-level + trained sparse）
- MoBA 走 MoE 风格（每个 query 选 top-k block，按 expert 路由）
- NSA 走三分支并联（compression + selection + window）
- 现在两条线在并行发展。DeepSeek 押 NSA/DSA；Kimi K2 押 MoBA

**Q2**：DSA 和 MLA 是替代还是叠加？
- **叠加**。MLA 压 KV bytes（一个 token 多少 GB），DSA 压 attention FLOPs（一次算多少）
- DeepSeek-V3.2 = MLA + DSA + MoE，三者正交叠加

**Q3**：为什么 DeepSeek 在 V3.2 才上 DSA？
- V2、V3 已经用 MLA 把 KV 压到 1/40，128k context 显存够
- V3.2 目标 1M+ context，光压 KV 不够，必须再压算量
- → DSA 是顺着这个 roadmap 来的

**Q3.5**：那 NSA 呢？为什么 NSA 从来没在 DeepSeek 在线模型里出现？见 §2.8。一句话：NSA 是给 MHA/GQA 设计的 block-level 三分支，跟 MLA 的"跨 head 共享 latent"不合槽；DSA 是为 MLA 重写的 token-level 两段式（lightning indexer + top-k）。NSA 留作学术奠基，主线走 DSA。

**Q3.6**：DSA 的 indexer 既然也是 O(L²)，为什么省得动？
- indexer 的 head 数和维度都比主 attention 小一两个数量级，又跑 FP8 + ReLU，没有 softmax
- 主 attention 从 $O(L^2)$ 降到 $O(L \cdot 2048)$，省下来的远大于 indexer 多花的
- L 越大优势越明显：1k 上几乎打平，128k 上 prefill -46% / decode -78%

**Q3.7**：top-k 离散不可导，DSA 怎么训？
- 关键 trick：**两阶段 + detach**（见 §2.4）
- Stage 1 冻主模型，indexer 只学"模仿真实 attention 分布"（KL 蒸馏）
- Stage 2 把 indexer 输入从计算图 detach 掉，indexer 继续 KL 学，主模型走标准 LM loss
- → 完全绕开"top-k 不可导"问题，indexer 的梯度从蒸馏目标来，不从 LM loss 来

**Q4**：CSA 和 HCA 为什么要做成两种？只用一种不行吗？
- 单 CSA（轻压缩 + sparse）：远距离信息全靠 indexer 选 1024 → 长 context 命中率掉
- 单 HCA（重压缩 + dense）：m'=128 太粗，**局部细节**（代码、数学）被池化掉
- 双层 = "近的看清楚 / 远的看轮廓"互补，**这是 V4 在 1M context 的核心架构选择**（见 §3.7）

**Q5**：HCA 为什么用 dense 而不是像 CSA 那样 sparse？
- 因为 m'=128 已经把 1M token 压成 7.8k 个 entry —— 这个长度做 dense MQA attention 已经很便宜
- 再叠一层 sparse 选择反而是 over-engineering：增加 kernel 复杂度、不可导问题、训练不稳
- 工程哲学：**能 dense 就别 sparse**，sparse 只在 dense 真的扛不住时才上

**Q6**：CSA 和 DSA 是不是几乎一样？
- 区别：DSA 的基本单元是**单 token**；CSA 的基本单元是**压缩 entry（4 个 token 一组）**
- CSA = "DSA 跑在压缩过的 KV 上" → indexer 要打分的对象从 1M 变成 250k，**top-k 选择本身更便宜，命中也更稳**（每个 entry 信息更密集）
- 你可以把 CSA 理解成"加了一层压缩前置 + 沿用 DSA 的 indexer+top-k 范式"

**Q4**：NSA 训练比 dense 难吗？
- 训练 wall time 与 dense 接近（block-sparse kernel 已优化）
- 收敛速度与 dense 同步（gating 能学到合理路由）
- 主要难点在 kernel + 训练稳定性，不在数学

**Q5**：NSA 的 compression 分支会不会丢信息？
- compression block size 通常 32（vs full 1）→ 信息密度降 32×
- 但 compression 只是"粗筛"，最终细信息靠 selection 分支补回
- 类似 attention 的 hierarchical 设计（先看大图，再 zoom-in）

**Q6**：sparse attention 会不会被 SSM/Mamba 取代？
- SSM 是 fundamental 的另一条路（线性时间、recurrent state）
- sparse attention 是"在 attention 内部省"
- 当前混合趋势：**Mamba + sparse attention** 一起用，互补（如 Jamba、Mamba-Hybrid）
- 见 §7

---

## 参考资料

- [DeepSeek NSA paper (2025) — Native Sparse Attention](https://arxiv.org/abs/2502.11089) ⭐⭐
- [DeepSeek-V3.2 技术报告 (arXiv 2512.02556)](https://arxiv.org/abs/2512.02556) ⭐ — DSA 完整描述
- [DeepSeek-V3.2-Exp 技术报告 / 模型卡](https://github.com/deepseek-ai/DeepSeek-V3)
- [Shawn Ding — DSA 在 SGLang 中的实现](https://shawnding.medium.com/deepseek-sparse-attention-and-its-implementation-in-sglang-b0bb907c375a) — lightning indexer + top-k 选择
- [Sebastian Raschka — Technical tour of DeepSeek V3 → V3.2](https://magazine.sebastianraschka.com/p/technical-deepseek) — NSA / DSA 同源、不同实现的讨论
- [DeepSeek-V2 技术报告 (MLA)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [Tri Dao — Block-sparse FlashAttention 讨论](https://github.com/Dao-AILab/flash-attention/discussions)
- [MoBA paper (Moonshot, 2025-02)](https://arxiv.org/abs/2502.13189)
- [SGLang — Sparse attention support 文档](https://docs.sglang.ai/)
- [DeepSeek-V4-Pro Hugging Face 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) — V4 CSA/HCA 架构
- [James Koh — DeepSeek-V4 Beyond Basics: mHC, CSA, HCA, Muon](https://medium.com/mitb-for-all/deepseek-v4-beyond-basics-a-practical-guide-to-mhc-csa-hca-and-muon-bf40c9863ef8) ⭐ — eq.9-14 压缩器公式、m=4/m'=128 细节
- [Into AI — DeepSeek-V4 Attention: From MHA to CSA and HCA](https://www.intoai.pub/p/what-makes-deekseek-v4-so-good)
- [Outcome School — Decoding DeepSeek-V4](https://outcomeschool.com/blog/decoding-deepseek-v4) — V4-Pro/Flash 层间编排
