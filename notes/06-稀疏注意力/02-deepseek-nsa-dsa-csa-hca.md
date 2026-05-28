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

### 2.1 与 NSA 的关系

| 维度 | NSA | DSA |
|------|-----|-----|
| 提出 | 学术论文（2025-02） | V3.2 模型 |
| 基础 | MHA / GQA | **MLA** ⭐ |
| 三分支 | 是 | 简化版 |
| 是否端到端训练 | 是 | 是 |

→ DSA 把 NSA 的思想 **mountain 到 MLA 之上**：KV 已经是低秩 latent $c$，sparse selection 直接在 $c$ 上选。

### 2.2 工程意义

- **MLA + DSA 是 DeepSeek-V3.2 长上下文的核心**：MLA 压 KV 体积，DSA 压 attention 算量
- 推理引擎已开放：DeepSeek 的 SGLang、FlashMLA 都支持
- 公开评测：相同硬件下，1M context 训练 / 推理 都比 dense 快数倍

### 2.3 与其他长上下文方案的关系

```
长 context 的三条独立优化轴：
  ① KV 体积：     MHA → GQA → MLA → DSA
  ② attention 算量：full → sparse (NSA, DSA, MoBA)
  ③ 位置编码：    RoPE → YaRN → NTK → ...
```

DSA = 在 MLA 这个 KV 压缩之上，再做 attention 算量稀疏。两者正交。

---

## 三、CSA / HCA（Compressed/Hierarchical-Compressed Attention）

这一组是 DeepSeek 在工业部署中用到的"过渡"组件，没单独成 paper，但在工程文档里出现：

- **CSA (Compressed Selection Attention)**：用更 aggressive 的 compression（如 1/64 池化）做粗粒度路径
- **HCA (Hierarchical Compressed Attention)**：分多级压缩（L0 全局、L1 段级、L2 块级），层层 zoom-in
- 它们都是 NSA/DSA 思想的"工程子集"，根据具体长度切换

工业上的意义：
- 1M context 时单级 compression 仍太密 → 加 hierarchical 层级
- 部署时根据 batch 平均长度动态切换

---

## 四、和 MoBA 等同期方案对比

| 方案 | 来源 | 时间 | 稀疏 pattern | 训练原生 | 与 MLA 兼容 |
|------|------|------|--------------|---------|-------------|
| **NSA** | DeepSeek | 2025-02 | 三分支（cmp+sel+win） | ✓ | 不直接 |
| **MoBA** | Moonshot | 2025-02 | MoE-style block routing | ✓ | 不直接 |
| **DSA** | DeepSeek | 2025-09 | NSA 思想 + MLA 集成 | ✓ | ✓ |
| H2O / Quest | 学术 | 2023-2024 | 推理时 KV eviction | ✗ | 可 |

→ 2025 的趋势：**block-level 选择 + 训练原生**，这是 DSA、NSA、MoBA 三家共识。

---

## 五、为什么 block-level 选择是工程主流

回到 §6.1 提的"稀疏的工程难点"：

| 稀疏粒度 | 工程难度 | FA kernel 友好 |
|----------|---------|----------------|
| token-level | 很难（每 token 一个不规则集合） | ✗ |
| **block-level** | **可行** | **✓** ⭐ |
| layer-level | 简单（整层换 attention 类型） | ✓（但粒度太粗） |

block-level 的好处：
- block 尺寸（32/64）和 FA 的 tile 完美对齐
- 内存读取仍然连续 → HBM 带宽利用高
- gradient 容易回传（block 内是 dense matmul）

NSA、MoBA、DSA 都选 block-level，**不是偶然**。

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
- [DeepSeek-V3.2-Exp 技术报告 / 模型卡](https://github.com/deepseek-ai/DeepSeek-V3)
- [DeepSeek-V2 技术报告 (MLA)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [Tri Dao — Block-sparse FlashAttention 讨论](https://github.com/Dao-AILab/flash-attention/discussions)
- [MoBA paper (Moonshot, 2025-02)](https://arxiv.org/abs/2502.13189)
- [SGLang — Sparse attention support 文档](https://docs.sglang.ai/)
