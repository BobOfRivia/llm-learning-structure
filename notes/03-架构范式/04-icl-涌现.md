# 3.4 In-Context Learning 与涌现

[← 返回框架](../../README.md) · [📎 materials.md → §3.4](../../materials.md)

---

## 一、In-Context Learning（ICL）是什么

由 GPT-3（Brown et al. 2020）定义：

> **模型在不更新任何参数**的情况下，通过在 prompt 中提供若干输入-输出示例（few-shot），就能学会新任务。

```
prompt:
  Translate English to French.
  sea otter => loutre de mer
  peppermint => menthe poivrée
  cheese =>
output:
  fromage
```

- **Zero-shot**：只有任务描述，无示例
- **One-shot**：1 个示例
- **Few-shot**：几到几十个示例
- **Many-shot**（2024）：100+ 示例（长上下文模型新场景）

> ICL 不是"学习"（参数不变），而是在**推理时根据 context 改变行为**。

---

## 二、为什么 ICL 这么关键

**ICL 让 decoder-only 范式赢得了一切**：
- 一个预训练模型就能做所有下游任务（无需 fine-tune）
- 用 prompt 切换任务（翻译 / 问答 / 摘要 / 代码 / agent）
- 没有 ICL，LLM 只是一个"文本续写器"，难以工程化部署
- ICL = LLM 能成为"通用 API"的核心理由

---

## 三、ICL 在结构上需要什么

### 3.1 必要条件
- **自回归 + causal attention**：让 context 能逐 token 影响后续生成
- **长 context**：示例需要塞得下
- **足够 scale**：小模型几乎没有 ICL 能力

### 3.2 Encoder-Decoder 为什么 ICL 弱
- examples 先被 encoder 压缩成定长 hidden states → 信息瓶颈
- decoder 通过 cross-attn 间接访问，无法逐 token 精细对齐
- 改造（如 Flan-T5）可以缓解，但天花板低于 decoder-only

### 3.3 Encoder-only 为什么没有 ICL
- 没有"生成"，自然没有 in-context "调用"
- BERT 系本质上是 supervised 工具

---

## 四、ICL 的机制（学术解释）

### 4.1 Induction Heads（Anthropic 2022）⭐

[Olsson et al. 2022](https://arxiv.org/abs/2209.11895) 提出：
- 训练过程中模型形成 **induction heads** —— 一类专门做模式补全的 attention head
- 模式：`...A B...A → B`（在 context 早期看到 A 后跟 B，下次看到 A 时复制 B）
- ICL 能力的出现**与 induction heads 形成时间高度对齐**

### 4.2 Implicit Gradient Descent 假说

[Akyürek et al. 2022](https://arxiv.org/abs/2211.15661)、[von Oswald et al. 2022](https://arxiv.org/abs/2212.07677)：
- Transformer 的 forward pass 可以模拟"在 context 中做隐式梯度下降"
- 在线性回归任务上得到了数学证明：足够大的 Transformer 一次 forward = 一步 GD
- 这给"为什么 few-shot 能学到新模式"提供了理论支撑

### 4.3 Bayesian 解释

[Xie et al. 2021](https://arxiv.org/abs/2111.02080)：
- 把 ICL 看作 Bayesian 推断：context 中的示例帮助模型推断潜在任务变量
- 示例越多 → 后验越尖锐 → 输出更确定

### 4.4 复合解释

主流共识：ICL 不是单一机制，而是多个电路（induction head、引发模式匹配、知识检索）的协同。

---

## 五、Chain-of-Thought（CoT）—— ICL 的延伸

[Wei et al. 2022](https://arxiv.org/abs/2201.11903)：在 few-shot 示例里加入"推理过程"显著提升 reasoning 能力。

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls. How many does he have now?
A: Roger started with 5 balls. 2 cans of 3 balls = 6 balls. 5+6=11. The answer is 11.
```

变体：
- **Zero-shot CoT**：在 prompt 后加 "Let's think step by step"（[Kojima et al. 2022](https://arxiv.org/abs/2205.11916)）
- **Self-Consistency**：采样多条推理 → 投票（[Wang et al. 2022](https://arxiv.org/abs/2203.11171)）
- **Tree of Thoughts**：搜索式 reasoning（[Yao et al. 2023](https://arxiv.org/abs/2305.10601)）
- **ReAct**：thought-action-observation 循环，agent 雏形（[Yao et al. 2022](https://arxiv.org/abs/2210.03629)）

CoT 推动了 **test-time compute** 这条新维度（§12.6）。

---

## 六、Emergent Abilities（涌现能力）

[Wei et al. 2022](https://arxiv.org/abs/2206.07682) 定义：

> 某些能力在小模型上几乎为零，**到达某个规模阈值后才出现**，且不是平滑外推。

经典涌现案例（按 scale 大致顺序）：
- arithmetic（3-digit add 在 ~10B 后涌现）
- word unscrambling
- multi-step reasoning
- instruction following
- ICL 本身

### 6.1 涌现的争议

[Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004) "Are Emergent Abilities a Mirage?"：
- 涌现可能是**指标选择**的产物
- 用 exact-match 评测 → 阶跃式
- 用 token-level log-prob → 平滑曲线
- 即"能力是连续提升的，只是离散指标遮蔽了它"

### 6.2 共识

- 涌现现象在某些 task 上确实存在（不仅是指标问题）
- 但很多"涌现"在更细的指标下是平滑的
- **Scaling 不能保证涌现**：很多能力需要数据 / 后训练设计

---

## 七、ICL 的实际工程要点

### 7.1 示例的选择
- **多样性 > 数量**：覆盖任务的边角
- **顺序敏感**：示例顺序影响输出（特别是 small model）
- **格式一致**：input/output 分隔符固定

### 7.2 失败模式
- **Reasoning 复杂任务**：few-shot CoT 必须，否则正确率骤降
- **超长 context**："lost in the middle" 现象（[Liu et al. 2023](https://arxiv.org/abs/2307.03172)；成因机制见 [§8.4](../08-长上下文/04-lost-in-the-middle.md)）
- **prompt 注入**：context 中的恶意示例可能改变行为

### 7.3 Many-shot（2024 长上下文场景）
- Gemini / Claude 长上下文（>100k）支持 100-1000 个示例
- 在某些任务上接近 fine-tune 效果
- 推理成本随 context 长度二次增长（除非用 prefix caching）

---

## 八、Test-Time Scaling —— 第三种 scaling 维度

2024 OpenAI o1 / 2025 DeepSeek R1 揭示：

> **训练时 scaling**（增大 N、D）+ **推理时 scaling**（增大思考 token 数）是**两条独立的轴**。

```
Total compute = train_FLOPs + serve_FLOPs × num_requests × thinking_tokens
```

- 推理时让模型生成长 CoT（数千-数万 token）
- 通过 RL（RLVR）训练这种"思考"能力
- 任务越难，思考时间收益越大

ICL（few-shot prompt）是 test-time 的低阶形态；
CoT / self-consistency / search 是中阶；
RL 训练出的 long-CoT 是高阶（§11.5 详）。

---

## 关键问答

**Q1**：ICL 是真"学习"吗？
- 参数不变 → 严格意义上不是学习
- 但 forward pass 内可以 emulate gradient descent / Bayesian update
- "推理时改变行为"是工程角度的关键

**Q2**：为什么 induction heads 解释 ICL？
- 模式：`...[A][B]...[A] → [B]`，最基本的 in-context 复用
- 训练中突然出现的 phase transition 与 ICL 能力出现一致
- 这只是 ICL 电路的最简形式，更复杂的"semantic induction"也存在

**Q3**：涌现是真的还是评测问题？
- 既有真涌现（某些 task 在阈值前真的 0%）
- 也有指标错觉（exact-match 的不连续）
- 实际工程中"以为模型不能做的任务，scaling 后能做"仍是常见现象

**Q4**：ICL 与 fine-tune 怎么选？
- Few examples（< 50）+ 通用任务：ICL
- Many examples（> 1k）+ 重复任务 + 延迟敏感：fine-tune / LoRA
- Many-shot ICL（100+ examples）+ 长上下文：折中点
- 注意 **prefix caching**：固定示例可缓存，长 ICL 推理成本下降

**Q5**：CoT 为什么有效？
- 把"一步答出"变成"分步推导"，每一步是简单子问题
- attention 在中间 token 上可"读"前面推理
- 是 test-time compute 的早期形式

**Q6**：long context 模型是否让 fine-tune 过时？
- 不会，但显著缩小了 fine-tune 的优势区
- Many-shot ICL 在 ~100 examples 上能匹配 LoRA 微调
- 推理成本仍高，所以 latency-sensitive 场景 fine-tune 仍胜

**Q7**：ICL 的"位置无关性"是怎么回事？
- 一些 task：示例顺序影响小（典型分类）
- 另一些：顺序敏感（recency bias，最后一个示例影响最大）
- 小模型对顺序更敏感

---

## 参考资料

- [Brown et al. 2020 — GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Wei et al. 2022 — Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)
- [Schaeffer et al. 2023 — Are Emergent Abilities of LLMs a Mirage?](https://arxiv.org/abs/2304.15004)
- [Olsson et al. 2022 — In-context Learning and Induction Heads (Anthropic)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Akyürek et al. 2022 — What Learning Algorithm is In-context Learning?](https://arxiv.org/abs/2211.15661)
- [von Oswald et al. 2022 — Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677)
- [Xie et al. 2021 — An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080)
- [Wei et al. 2022 — Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Kojima et al. 2022 — Zero-shot CoT (Let's think step by step)](https://arxiv.org/abs/2205.11916)
- [Wang et al. 2022 — Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171)
- [Liu et al. 2023 — Lost in the Middle: How LLMs Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Agarwal et al. 2024 — Many-Shot In-Context Learning (Google)](https://arxiv.org/abs/2404.11018)
- [OpenAI — Learning to reason with LLMs (o1 blog)](https://openai.com/index/learning-to-reason-with-llms/)
- [DeepSeek-R1 技术报告](https://arxiv.org/abs/2501.12948)
