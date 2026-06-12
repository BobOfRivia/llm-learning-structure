# 1.3 Scaling Laws

[← 返回框架](../../README.md) · [📎 materials.md → §1.3](../../materials.md)

---

## 一、问题定义

给定**固定计算预算** C（FLOPs），如何选择参数量 N 和训练 token 数 D，使 test loss L 最小？

**核心公式**（业内"6ND 法则"）：
$$C \approx 6 \cdot N \cdot D$$

> **推导**：每个 token 前向 FLOPs ≈ 2N（每参数 1 mul + 1 add），反向 ≈ 2× 前向 ≈ 4N。
> 故每 token 总 FLOPs ≈ 6N，全程 ≈ 6ND。

---

## 二、Kaplan 2020（OpenAI）

第一篇系统化的 Scaling Law。

**经验形式**：
$$L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

**结论**：
- N 和 D 都对 loss 有幂律影响
- **建议**：算力增加时，**多给 N**（参数），少给 D（数据）
- 计算最优比例：N ∝ C^0.73，D ∝ C^0.27

**后果**：直接驱动了 GPT-3（175B 但只训 300B tokens，~1.7 tokens/param）等"大而欠训"的模型。

---

## 三、Chinchilla 2022（DeepMind）⭐ 范式转折

Hoffmann et al. 用 **3 种独立方法**（IsoFLOPs、参数化拟合、loss 曲面）得出与 Kaplan 截然不同的结论：

**核心结论**：
> **计算最优时，N 和 D 应当 ≈ 等比例 scale。**
> 约 **20 tokens per parameter** 是 compute-optimal。

**经验损失**：
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 E ≈ 1.69（不可约 entropy floor），α ≈ 0.34，β ≈ 0.28。

### 3.1 推导：约束最优化

在 C = 6ND 约束下最小化 L：
- 拉格朗日法解得：
  $$N_{opt} \propto C^{a}, \quad D_{opt} \propto C^{b}, \quad a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha+\beta}$$
- Chinchilla 实测 a ≈ b ≈ 0.5 → **N 和 D 同步增长**

### 3.2 Chinchilla 本身的验证
- 训了 70B 模型 + 1.4T tokens（比 Gopher 280B 小 4×，token 多 4×）
- 在几乎所有 benchmark 上击败 Gopher

### 3.3 含义
- GPT-3 / Gopher / Megatron-Turing NLG 等大模型都**严重欠训**
- 同样算力下"小一点多训练"更优

---

## 四、Kaplan vs Chinchilla 差异溯源

| 项 | Kaplan | Chinchilla |
|----|--------|-----------|
| **结论** | 优先放大 N | N、D 同步放大 |
| **学习率调度** | 固定 cosine 长度 | 与 D 匹配（关键差异） |
| **小模型实验** | LR 调度未对齐 | 重新调度 |
| **影响** | GPT-3 路线 | Llama 路线 |

> 学界共识：**Kaplan 的 LR 调度让小模型 underperform**，导致高估 N 的回报。

---

## 五、后 Chinchilla 时代：推理感知的 over-training

Chinchilla 只考虑**训练成本**，忽略**部署/推理成本**。

### 5.1 实际场景
- 一个模型训练一次，但要服务**万亿次**推理请求
- 总成本 = 训练 FLOPs + 推理 FLOPs × 请求数
- 推理 FLOPs ∝ N（不依赖 D），训练 FLOPs ∝ ND

### 5.2 结论
**部署成本驱使现代模型严重 over-train**：模型小一点（推理便宜）、tokens 多很多。

| 模型 | N | D (tokens) | D/N | vs Chinchilla |
|------|---|-----------|-----|---------------|
| Chinchilla | 70B | 1.4T | 20 | optimal |
| Llama-2 7B | 7B | 2T | 286 | ~14× over |
| Llama-3 8B | 8B | 15T | 1875 | ~94× over |
| Llama-3 70B | 70B | 15T | 214 | ~11× over |
| Qwen-2.5 7B | 7B | 18T | 2570 | ~129× over |
| DeepSeek-V3 | 671B (37B active) | 14.8T | 22 (基于全参) | ~Chinchilla |

> 注：MoE 的"实际 N"取激活参数还是全参数仍有争议。

### 5.3 论文
- Sardana et al. 2023 ("Beyond Chinchilla-Optimal")：当推理成本占比高时，最优解远离 Chinchilla
- 形式化推理最优 token / 参数比

---

## 六、MoE 与多模态的 Scaling Law

### 6.1 MoE Scaling
- "Scaling Laws for Fine-Grained MoE" (Krajewski et al. 2024)
- 关键变量增加：**激活专家数**、**专家粒度**
- 经验上 MoE 在固定计算预算下优于 dense

### 6.2 多模态
- "Scaling Laws for Multimodal" (Aghajanyan et al. 2023)
- 各模态有不同的 scaling exponent

---

## 七、Scaling Law 的局限

1. **数据质量假设恒定**：FineWeb-Edu 已经打破——更高质数据上同样 C 出更低 loss
2. **不预测涌现能力**：MMLU、GSM8K 等的飞跃不是 loss 的平滑外推(详见 §3.4 ICL 与涌现)
3. **不覆盖后训练**：SFT/RLHF/RLVR 的 scaling 是另一个体系
4. **不覆盖 test-time compute**：o1/R1 的"推理时 scaling"是新维度
5. **数据墙**：高质量人类语料可能在 10²-10³ T tokens 量级见顶

---

## 关键问答

**Q1**：C = 6ND 怎么来的？
- 前向：每参数 1 个 mac = 2 FLOPs → 每 token 2N
- 反向：≈ 2× 前向 → 每 token 4N
- 合计：6N FLOPs / token，全程 6ND
- 不含 attention 的 O(L²) 项（在 L 较小时可忽略）

**Q2**：Chinchilla 和 Kaplan 的本质差异？
- Kaplan 学习率调度跟参数挂钩而不跟 token 数挂钩，导致小模型"未训完"
- Chinchilla 重新跑了对齐的实验，得出 N、D 同步增长结论
- 形式上：L = (N_c/N)^α + (D_c/D)^β（Kaplan，无 floor） vs L = E + A/N^α + B/D^β（Chinchilla，有 entropy floor）

**Q3**：为什么 Llama-3 把 8B 模型训到 15T tokens，远超 Chinchilla？
- 推理成本主导总成本
- 小模型 over-train 的边际 loss 改善仍是正的（虽然递减）
- 长尾能力（罕见知识、多语言）更需要数据量

**Q4**：MoE 的 N 怎么算？
- 训练成本由**激活参数**决定（FLOPs ∝ active params × D）
- 推理 KV-cache 与全参数无关，权重 IO 与激活参数相关
- 显存占用由**全参数**决定
- 故"scaling laws for MoE"通常分别建模

**Q5**：Scaling Law 还有用吗（在 2025-2026 已经反复被"打破"的语境下）？
- 仍是**指导预算分配**的基本工具
- 但需要：考虑数据质量、考虑推理、考虑 test-time compute
- 涌现 / 后训练 / reasoning 需要新的 scaling 框架

---

## 参考资料

- [Kaplan et al. 2020 — Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. 2022 — Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Sardana et al. 2023 — Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)
- [Krajewski et al. 2024 — Scaling Laws for Fine-Grained MoE](https://arxiv.org/abs/2402.07871)
- [Aghajanyan et al. 2023 — Scaling Laws for Generative Mixed-Modal LMs](https://arxiv.org/abs/2301.03728)
- [Chinchilla's Wild Implications (LessWrong)](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)
- [Llama-3 技术报告](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Inverse Scaling Prize](https://github.com/inverse-scaling/prize)
