# 12.6 Test-time Scaling（CoT / Self-Consistency / Best-of-N / Tree Search / o1·R1 范式）

[← 返回框架](../../README.md) · [📎 materials.md → §12.6](../../materials.md)

---

## 〇、本节回答什么

> 2024-09 OpenAI 发布 o1 之后，"用推理算力换质量"成为继 pretraining scaling、post-training scaling 之后的**第三条 scaling 曲线**。本节梳理：CoT / Self-Consistency / Best-of-N / Tree Search (ToT / MCTS / rStar) → o1 / R1 范式 → 2025 反直觉发现（"更长 ≠ 更好"、Short-m@k）。它既是推理优化的一种特殊"反向"——主动消耗更多算力，又是后训练（§11.4）的延伸——靠推理时搜索逼近 RLVR 的能力。

定位：本节关注 **inference-time techniques**，§11.4 关注 **training-time RL**。两者互补。

---

## 一、什么是 Test-Time Scaling

### 1.1 三条 scaling 曲线

```
Pre-training scaling   = 训练 data + params + compute
Post-training scaling  = SFT/RLHF/RLVR 算力（§11）
Test-time scaling      = 推理时多采样、长 CoT、tree search

→ 任务 difficulty 越高，越后段曲线越陡
   AIME / 竞赛数学 / 困难代码：test-time scaling 收益最显著
```

### 1.2 直觉

任务难度越大，模型一次 forward 给出最优解的概率越低；但**让它"想"得长一点 / 多试几次 / 验证一下**，正确率系统性提升。

OpenAI o1 工程化的核心：在 RL 训练里**鼓励模型自己延长 CoT**，推理时把这条长 CoT 充分跑出来。

---

## 二、最朴素的 test-time scaling：Chain-of-Thought (CoT)

### 2.1 zero-shot CoT

> "Let's think step by step." — Kojima et al., 2022

模型显式展开中间步骤再给答案。GSM8K 上 zero-shot 从 ~17% → ~78%。

### 2.2 few-shot CoT

> Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

在 prompt 里给 few-shot 例子（每个含中间步骤）。当时 GSM8K 上 PaLM-540B 从 17% → 57%。

### 2.3 为什么 CoT 有效

- **计算扩展**：每多一个 reasoning token，模型多走 N 层 transformer，等价获得额外计算预算
- **错误分解**：把一步推理拆成多步，每步错误率降低
- **格式诱导**：让模型把推理过程外化为可验证的中间状态

---

## 三、Self-Consistency / Best-of-N（采样 N 次再选）

### 3.1 Self-Consistency (Wang et al. 2022)

```
1. 用同一个 prompt 采样 N 条 CoT （temperature 0.7）
2. 提取每条的最终答案
3. 多数投票（majority vote）选答案
```

GSM8K 上 N=40 时再涨 10–20 个点。**直觉**：错误答案是分散的，正确答案是 mode。

### 3.2 Best-of-N (BoN) — 用 reward model 选

```
1. 采样 N 条 CoT
2. 用一个 reward model 给每条打分（PRM/ORM）
3. 取最高分
```

BoN 比 majority vote 更强（reward model 比"投票"信息量更多）。

### 3.3 收益曲线（empirical）

```
GSM8K / MATH 上：
  N=1   ─ baseline
  N=4   ─ +5 pt
  N=16  ─ +10 pt
  N=64  ─ +15 pt
  N>64  ─ 收益递减（饱和到 pass@N 上限）
```

Test-time compute 增加越多，BoN 收益越大——这就是 **inference scaling curve**。

### 3.4 BoN 的两类 verifier

| | Outcome Reward Model (ORM) | Process Reward Model (PRM) |
|---|---|---|
| 监督信号 | 只看最终答案对错 | 给每步推理打分 |
| 数据成本 | 低（自动验证） | 高（每步标注） |
| BoN 表现 | 中 | 强（但贵） |
| 流派代表 | R1 用 ORM 路线 | OpenAI / DeepMind PRM800K |

> R1 论文明确说：**PRM 难训、容易 reward hacking，ORM + RLVR 是更鲁棒路线**（详见 §11.4）。

---

## 四、Tree Search：把推理变成搜索

### 4.1 Tree of Thoughts (ToT, Yao et al. 2023)

```
                    root
                  /  |  \
                ...  ...  ...    ← 每节点是 partial thought
              /  |  \
            ...  ...  ...
```

每节点是一段推理，扩展时用 LLM 提出 candidates；选节点用 LLM/heuristic value function 评估。常配 BFS / DFS / beam。

应用：24-game、creative writing、crossword 等需要回溯的任务。

### 4.2 MCTS (Monte Carlo Tree Search)

把 AlphaGo 那一套搬到 LLM：

```
四步循环：
  1. Selection：从 root 按 UCT 走到叶
  2. Expansion：扩展叶节点（用 LLM 提议下一步推理）
  3. Simulation：随机或 short-rollout 估终值
  4. Backup：把 reward 往上 propagate

UCT = Q(s,a) + c · sqrt(ln N(s) / N(s,a))
```

代表：**rStar**（Microsoft, 2024）、**rStar-Math**（2025）—— 7B 模型 + MCTS 在 AIME 上接近 GPT-4o。

### 4.3 ToT vs MCTS 对比

| | ToT | MCTS |
|---|---|---|
| 节点选择 | beam / heuristic | UCT |
| 探索-利用 | 弱 | 显式 |
| 适合 | 创意 / 短推理 | 长推理 + reward 可估 |
| 工程复杂度 | 中 | 高（需 value model） |

### 4.4 树搜索的代价

- 每个节点都是一次 LLM forward，开销巨大（一道题几百次 forward）
- 在线推理通常不可行；适合**离线生成训练数据**或 offline batch 任务
- 是 rStar / R1 数据合成 / OpenAI o1 训练 pipeline 的一环

---

## 五、o1 / R1 范式：把推理 scaling 内化进模型

### 5.1 范式革命

之前：CoT / SC / BoN / Tree Search 都是 **prompting + 外部搜索**——模型是被动的。

o1 (OpenAI, 2024-09) 与 R1 (DeepSeek, 2025-01) 的范式革命：
> **让模型自己生成长 CoT（含 self-reflection、self-correction、回溯）。RL 训练把"想得长 → 答得对"的策略内化。**

```
推理时：
  user: 给我 AIME 第 15 题
  model: <think>
           Let me re-read the problem... ← 自反思
           Try approach A:
             ... derive ...
             Hmm wait, this doesn't work because... ← 自纠错
           Let me try approach B:
             ...
           Verify: ...
         </think>
         <answer>42</answer>
```

### 5.2 训练 → 推理的 scaling 双侧

o1 / R1 同时挑战两条曲线：

| 轴 | 资源 | 收益 |
|----|------|------|
| training compute（RL） | 训练算力 | 把长 CoT 策略训进权重 |
| test compute（max thinking） | 推理算力（thinking token） | 同一模型给更多 thinking budget → 更准 |

OpenAI 公开图：**两条 log-linear 曲线同时存在**。

### 5.3 工程上的"thinking budget"

API 已支持 `max_thinking_tokens` 控制：
- 简单任务 200 token 思考 → 快
- 难题 8K+ thinking → 慢但准
- 极端：o3 on ARC-AGI 单题 1M+ thinking token

→ test-time scaling 成为 **可配置的推理预算**，与 §12.2 serving 调度直接相关。

---

## 六、2025 反直觉发现：长 ≠ 好

### 6.1 长 CoT 的边际收益递减

实测：thinking tokens 从 1K → 8K 涨明显，8K → 64K 涨缓慢甚至下降。原因：
- 推理误差累积（错一步坏整条）
- "rambling"：模型陷入无效自言自语
- 上下文超过训练长度 → 质量下降

### 6.2 Short-m@k（2025）

新发现：与其让一个模型推理 64K token，不如**采样 m 条短 CoT（如各 8K）再 BoN/SC**——质量更高、成本更低、可并行。

```
长策略：1 × 64K token thinking → 1 个答案
短策略：m × 8K token thinking → BoN/SC → 1 个答案

固定总 token 预算下，短策略往往胜出
```

### 6.3 inverse scaling 现象（part）

某些任务长 CoT 反而错——典型是简单算术、明显题。模型会"过度思考"把对的搞错。
对策：**自适应 thinking budget**——简单题不思考、难题深入想。已成为 2025 推理 API 设计方向。

---

## 七、Test-Time Scaling 与系统的关系

```
Test-time scaling 引出新的系统挑战：

1. 长 CoT → 长 KV → §12.5 长上下文推理优化
2. BoN/SC → 并行 N 路 → §12.2 serving 调度 + prefix cache
3. Tree search → 大量 forward → §12.3 speculative decoding 收益大
4. thinking budget 控制 → 调度按 SLO + budget 双维度
5. PRM/Verifier → 额外模型推理 → 多模型 serving
```

**重要工程模式**：BoN 时多条 CoT 共享同一个 prompt → prefix cache 命中率 100% → 极适合 SGLang RadixAttention。

---

## 八、不同任务上的 test-time scaling 配方

| 任务 | 推荐策略 |
|------|----------|
| 简单 chat / 翻译 | 不需要 test-time scaling，单次采样足够 |
| GSM8K（小学数学） | CoT + SC (N=8) 足够 |
| MATH / AIME | R1 / o1 类长 CoT 模型 + BoN |
| 困难编程（SWE-Bench） | Agentic 多步 + MCTS / 多候选验证 |
| 创意写作 | ToT + 多候选 + 人/AI 选 |
| 科研推理（GPQA） | o1 / R1 长 CoT + verifier |

---

## 九、关键问答

**Q1：CoT 为什么有效——只是 prompt 技巧吗？**
A：不只。CoT 提供了**额外的算力预算**——每个 reasoning token 让模型多过一遍 transformer，等价加深网络。同时**外化中间状态**让错误可分解、可验证。理论分析（Feng et al., NeurIPS 2023）指出某些问题（动态规划、组合）只有当允许 polynomial 长度 CoT 时才能被 transformer 解。

**Q2：Self-Consistency 与 Best-of-N 哪个强？**
A：BoN 在有好 reward model 时强；SC 在无 reward model 时是最便宜的 BoN approximation。SC 隐含假设"正确答案是 mode"——对开放生成任务不成立；对数学/代码这种有"标准答案"的任务很有效。R1 论文里实际用的就是 **majority vote @ N**（特殊形式的 SC）来评估 pass@1 的稳定性。

**Q3：MCTS 与 ToT 选哪个？**
A：(1) reward 可估（有 verifier 或可执行测试）→ MCTS；(2) reward 难估、靠人工/启发式 → ToT/beam；(3) 任务深度浅 → 简单 BFS+rerank 就够；(4) 任务深度长（长代码、长 proof）→ MCTS + value model。rStar 用 MCTS 是因为数学题有自动 verifier。

**Q4：o1/R1 内化的长 CoT 与外部 tree search 是替代关系吗？**
A：很大程度上是。o1/R1 用 RL 把"分支-回溯-验证"训练进单一 forward 路径，省去显式 tree。但**仍可叠加外部 BoN / 自我一致性**——R1 + majority vote 通常还有 3–5 pt 提升。所以工程实践是：**内化 CoT + 外部 BoN** 组合最强。

**Q5："越想越好"在什么时候失效？**
A：(1) 简单题——长 CoT 容易 over-think 把对的搞错；(2) 模型训练上下文小于 thinking budget——超出训练长度后 self-attention 漂移；(3) 任务无客观答案（创意写作）——thinking 反而僵化；(4) 错误累积型任务（长链推理）——一步错全链错，长比短危险。Short-m@k 等工作正在系统化这些发现。

**Q6：PRM 真的没用吗，为什么 R1 选 ORM？**
A：PRM 在评测阶段（BoN）依然有效，OpenAI PRM800K 在 MATH 上确实赢 ORM。但**作为 RL 训练 reward** 时 PRM 容易被 hack（模型学会"看起来每步对但答案错"），且训练数据贵（每步要标）。R1 选择 RLVR + ORM 是出于**可验证、不可 hack** 的考虑。不矛盾：**ORM 做 RL，PRM 做推理 BoN** 可叠加。

**Q7：test-time scaling 与 serving 系统的关系？**
A：四个关键点：(1) thinking 长 → KV 长 → 长上下文优化（§12.5）必须；(2) BoN 多采样 → prefix 共享 → SGLang Radix 收益大；(3) 推测解码（§12.3）在 BoN 多分支验证上收益翻倍；(4) thinking budget 成为新的调度维度——SLO 不再只是 TTFT/TPOT，还包括"答对率 vs 总 budget"。

---

## 参考资料

### 必读论文 ⭐⭐
1. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Wei et al., NeurIPS 2022 ⭐⭐
2. [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) — Wang et al. 2022 ⭐⭐
3. [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Yao et al. 2023 ⭐
4. [Let's Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050) — OpenAI 2023 ⭐⭐
5. [Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) — DeepMind 2024 ⭐⭐

### o1 / R1 范式
6. [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) ⭐⭐
7. [OpenAI o1 System Card](https://openai.com/o1/) ⭐
8. [Lilian Weng — Why Reasoning Models](https://lilianweng.github.io/) ⭐⭐
9. [Open-R1 (HuggingFace)](https://github.com/huggingface/open-r1) ⭐

### Tree Search / MCTS for LLM
10. [rStar-Math: Small LLMs Can Master Math Reasoning](https://arxiv.org/abs/2501.04519) — Microsoft 2025 ⭐
11. [AlphaMath: Process-Supervised Math Reasoning via MCTS](https://arxiv.org/abs/2405.03553)
12. [Reasoning with Reinforced Functional Token Tuning](https://arxiv.org/abs/2502.13389)

### Short-m@k / 反直觉发现
13. [Don't Overthink it: A Survey of Efficient R1-Style Large Reasoning Models](https://arxiv.org/abs/2503.16419)
14. [Short-m@k: Why Multiple Short Traces Beat One Long Trace](https://arxiv.org/abs/2505.00007) ⭐
15. [The Pitfalls of Reasoning for Code Generation](https://arxiv.org/abs/2502.14444)

### 综述 / 工程
16. [A Survey on Inference-Time Compute Scaling for LLMs](https://arxiv.org/abs/2410.18116) ⭐
17. [Test-Time Compute Scaling Survey (HuggingFace blog)](https://huggingface.co/blog/) ⭐⭐
18. [vLLM / SGLang 的推理时采样并行实现](https://docs.vllm.ai/) — 工程参考

---

### 第 12 章小结

```
§12.1 量化            → 压 bandwidth + 显存，FP8/NVFP4 时代
§12.2 Serving 系统     → 调度 + 内存管理（vLLM/SGLang）
§12.3 推测解码         → 摊薄 decode 串行成本（EAGLE 主流）
§12.4 PD 分离          → 集群尺度解 prefill-decode 冲突（Mooncake）
§12.5 长上下文推理     → KV 量化/驱逐 + chunked prefill + 位置外推
§12.6 Test-time Scaling → 第三条 scaling 曲线（o1/R1 范式）
```

→ 下一章 [§13 评测体系](../13-评测体系/)：MMLU / GSM8K / SWE-Bench / 长上下文 / Agent / Arena / 污染问题。
