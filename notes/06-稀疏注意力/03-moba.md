# 6.3 MoBA — Mixture of Block Attention

[← 返回框架](../../README.md) · [📎 materials.md → §6.3](../../materials.md)

---

## 〇、一句话

MoBA = **把 MoE 的"top-k 路由"思想搬到 attention 上**：每个 query 选 top-k 个 KV block 去 attend，其他 block 完全跳过。

由 Moonshot（月之暗面）于 [2025-02 提出](https://arxiv.org/abs/2502.13189)，是 Kimi 长 context（1M+）的核心组件之一。

---

## 一、和 NSA / 经典稀疏的关系

回到 §6.2 的结论：**2025 年长 context 稀疏 attention 的共识 = block-level + 训练原生**。
MoBA 和 NSA 同属这一波，但路径不同：

| 维度 | MoBA | NSA |
|------|------|-----|
| 设计灵感 | **MoE 路由** | 三分支并联 |
| 稀疏结构 | 每 query top-k block | compression + selection + window |
| 是否带 dense fallback | 否（纯 sparse） | 有（compression 分支始终全连） |
| Decoder 友好 | ✓（causal block routing） | ✓ |
| 来源 | Moonshot (Kimi) | DeepSeek |
| 开源 | ✓ | ✓ |

---

## 二、核心机制

### 2.1 把 KV 序列切 block

```
KV 序列 (L=128k):
[ block 0 | block 1 | block 2 | ... | block N ]
   每 block 大小 B (e.g., 512)
```

### 2.2 每 block 算一个"代表向量" (gating key)

对每个 block，把内部所有 K 平均（或线性投影）成一个向量 $\bar{k}_b$。

### 2.3 Query 选 top-k block

```
score_b = Q · k̄_b        # query 跟每个 block 代表的相似度
top_k = argtopk(score_b)  # 取分数最高的 k 个 block
```

→ 类似 MoE 里"每 token 选 top-k expert"，但 expert 换成了 **KV block**。

### 2.4 在选中的 block 内做 full attention

```
对选中的 k 个 block，把它们的 K, V 拼起来做标准 attention
其余 block：跳过（既不算分，也不读 KV）
```

复杂度：
- 选 block：$O(L \cdot L/B)$（每个 token 算 L/B 个分数，但 L/B 通常 ~256，开销小）
- 实际 attention：$O(L \cdot k \cdot B)$
- 当 $k \cdot B \ll L$ 时显著 sub-quadratic

---

## 三、Causal 处理

decoder-only 模型必须 causal，MoBA 的 trick：

```
token 位置 t 所在 block 编号 = floor(t / B) = b_cur

可选 block：
  - block 0 ... block_{b_cur - 1}    全部历史 block
  - block_{b_cur}                    自己所在 block（强制必选）
```

再在历史 block 中选 top-k。

→ 当前 block **强制选中**（保证 local context 不丢）
→ 历史 block 走 top-k routing（远距离按需取）

实测对长上下文 retrieval、long needle-in-a-haystack 等任务，与 dense attention 接近甚至更好。

---

## 四、Block size 和 top-k 的取舍

| 参数 | 大 | 小 |
|------|----|----|
| **Block size B** | 选 block 时粒度粗、kernel 友好；但 block 内可能混杂 | 粒度细，但 score 噪声大 |
| **Top-k** | 接近 dense，速度收益小 | 速度快，质量风险高 |

Kimi 的典型配置：
- $B$ = 512
- $k$ ≈ 16-32（对 128k context，约访问 8k-16k token，类似 8-16× 加速）

---

## 五、训练相关

MoBA 是**端到端训练**的稀疏 attention，但实现上有几个细节：

1. **从 dense 起步、渐进切到 MoBA**
   - 训练前期，gate score 接近随机 → 全开 dense
   - 用 schedule 逐步把 dense 比例降到 0
   - 类似 MoE 的 warm-up

2. **不可导的 top-k**
   - top-k selection 本身离散
   - 采用 straight-through estimator（前向硬选、反向走 soft）
   - 或在 block 代表向量上用 sigmoid soft routing

3. **辅助 load-balancing loss**
   - 类似 MoE，防止部分 block 永远不被选
   - 在 long context 上尤其重要（不然中间 block 容易冷启动失败）

---

## 六、MoBA 在 Kimi 中的角色

Kimi K1.5/K2 的长 context 部署组合：
```
MLA (KV 压缩) + MoBA (attention 稀疏) + RoPE + chunked prefill
```

效果：
- 1M context 推理 / 训练 都成立
- 长 context 评测（RULER、LongBench）成绩在开源 SOTA 之列
- 对 retrieval-heavy 任务保持精度（路由能学到"找哪些 block"）

---

## 七、与 MoE 的对比和借鉴

| 维度 | MoE | MoBA |
|------|-----|------|
| 路由对象 | FFN expert | KV block |
| 每 token 选几个 | 通常 1-2 | 通常 16-32 |
| Capacity 限制 | expert overflow drop | 没有（无 token drop） |
| Load balancing loss | ✓ | ✓ |
| 训练问题 | gate 塌缩、expert 死掉 | block 偏好、长尾 block 不被选 |

→ MoE 让 FFN 稀疏，MoBA 让 attention 稀疏。两者**正交**且能一起用：Kimi K2 就是 MoE + MoBA + MLA 全开。

---

## 关键问答

**Q1**：MoBA 和 H2O 都做 block / KV 选择，区别是？
- H2O：**推理时的 KV eviction**，看历史 attention 累积分数 → 永久 drop 一些 KV
- MoBA：**训练原生的 query-side routing**，每步 query 重选 → 不 drop KV，只是这次不读
- 实测 MoBA 质量更稳定（KV 没丢，召回回来很容易）

**Q2**：MoBA 在短 context 上有用吗？
- $L \le 8k$ 时：dense + FA 已经够快，MoBA 没显著加速
- $L \ge 32k$：开始有收益
- $L \ge 128k$：必选；这是为长 context 设计的方案

**Q3**：top-k 选错了怎么办？
- 训练时模型会学到"选错代价高"→ gate 倾向保守
- 推理时确实可能漏 retrieval，但实测漏召率 < 1%（对 NIAH 评测）
- 实在担心可以加 fallback：top-k + 全局少量随机 block

**Q4**：MoBA 推理时的 KV-cache 量是多少？
- KV 全部要保存（query 选 block 是动态的）
- 所以 MoBA **不省 KV bytes**，只省 attention 算量
- 需要省 KV 体积要叠 MLA / GQA / 量化

**Q5**：MoBA 和 NSA 哪个对长 context 更优？
- 没有定论，都是 2025-02 同期工作
- NSA 的 compression 分支保证"全局总能看到"（即使被 selection miss）
- MoBA 更简洁，但只靠 routing 学到的覆盖能力
- 工业上 Kimi 押 MoBA，DeepSeek 押 NSA/DSA，两条路并行验证中

**Q6**：MoBA + MLA 能合并 kernel 吗？
- 理论上可以（block selection + 低秩 K 展开）
- 工程上仍在演进，目前两者更多是"先后两段"
- 这是 2025-2026 引擎层的优化方向之一

---

## 参考资料

- [MoBA paper — Mixture of Block Attention for Long-Context LLMs (Moonshot, 2025-02)](https://arxiv.org/abs/2502.13189) ⭐⭐
- [MoBA GitHub (Moonshot)](https://github.com/MoonshotAI/MoBA)
- [Kimi K1.5 技术报告](https://arxiv.org/abs/2501.12599)
- [DeepSeek NSA paper（同期对比）](https://arxiv.org/abs/2502.11089)
- [MoE 路由原始论文 — Switch Transformer (Fedus et al. 2021)](https://arxiv.org/abs/2101.03961)（路由思想来源）
- [Anthropic — Long Context 综述博客](https://www.anthropic.com/news/100k-context-windows)
- [HuggingFace — Long-context Methods 综述](https://huggingface.co/blog/long-context)
- [SGLang Sparse Attention 文档](https://docs.sglang.ai/)
