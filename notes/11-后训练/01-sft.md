# 11.1 SFT（Supervised Fine-Tuning）

[← 返回框架](../../README.md) · [📎 materials.md → §11.1](../../materials.md)

---

## 〇、本节回答什么

> 怎么把 pretrained base model 调成"能听指令的 chat model"？SFT 的数学目标到底是什么、和 pretrain 的 next-token-prediction 有什么区别？loss masking、packing、多轮对话怎么处理才严谨？数据量、配比、epoch 的工业经验是什么？SFT 失败的几种典型形态，机制是什么？

SFT 是后训练（Post-training）的第一步。一句话定位：

> **SFT = 激发 base model 已有能力 + 把行为对齐到 instruction-following 输出格式。**

它**不是**用来灌新知识的（灌新知识 → §10.5 continued pretraining，或 RAG），**也不是**用来"让模型变聪明"的（变聪明 → §11.4 reasoning RL）。这条边界 2023+ 已成共识，本节会给出形式化的解释。

```
base model (next-token predictor)
        ↓  SFT (instruction tuning)
chat model (能听指令、按格式回复)
        ↓  RLHF / DPO / RLVR
aligned / reasoning model
```

本节自下而上构建 SFT 的完整图景：

1. **数学起点**：SFT 损失从 conditional MLE 推出，与 pretrain loss 是同一个公式，差别只在 **mask 和数据分布**。
2. **能力激发**：为什么 "1k 高质量样本 ≈ 1M 普通样本"（LIMA 现象）？信息论与表示学习角度。
3. **工程细节**：loss mask 的边界处理、packing 的 block-diagonal attention、multi-turn 的 token layout，每一处都解释 *为什么这样做* 而不只是 *怎么做*。
4. **数据策略**：来源、配比、量级、清洗的工业经验，背后的多任务学习理论。
5. **失败模式**：灾难性遗忘的机制（Fisher information）、过拟合到样式、SFT 灌知识引发 hallucination 的根因。

---

## 一、数学起点：SFT 的损失到底是什么

### 1.1 next-token prediction 的统一视角

Pretrain 和 SFT 用的**是同一个损失公式**——条件似然的负对数：

$$
\mathcal{L}(\theta) = - \mathbb{E}_{u \sim p_{\text{data}}} \Big[ \sum_{t=1}^{|u|} \log \pi_\theta(u_t \mid u_{<t}) \Big]
$$

差别只在两件事：

1. **数据分布 $p_{\text{data}}$**：pretrain 是 web/书/代码混合；SFT 是 `(prompt, response)` 对组成的 chat 模板。
2. **哪些 token 算 loss**（loss mask）：pretrain 全算；SFT 只算 response token。

把 SFT 写清楚：每条样本 $u = (x, y)$（prompt + response），损失为

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \Big[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \Big]
$$

注意几个细节：

- 求和**只对 response 部分**（不算 prompt token 的 log-prob）。
- $\pi_\theta(y_t | x, y_{<t})$ 仍然是 next-token 分布，等价于把 $(x, y_{<t})$ 整段喂进模型、读 $y_t$ 位置的 logits。
- 这等价于最小化条件 KL：$\mathbb{E}_x [\text{KL}(p_{\text{data}}(\cdot|x) \,\|\, \pi_\theta(\cdot|x))]$。

> SFT 不是"新算法"，它就是**条件 MLE**——监督学习里最经典的目标。所有 chat 模型表面的"指令跟随"能力，本质上是这条简单 loss 在精挑细选的数据上跑出来的。

### 1.2 为什么 mask 掉 prompt token（不是工程节省，而是统计必要）

很多新手会问："prompt 上也算 loss 不行吗？多了不少梯度信号。" 答案是**会让模型变差**。原因有三层：

**(1) 任务目标错配**。我们想学的条件分布是 $p(y|x)$，而不是 joint $p(x, y) = p(x) p(y|x)$。如果 prompt token 也算 loss，等价于让模型同时最大化 $\log p_\theta(x)$。但 prompt 来自任意用户分布——既不是 base model 的 pretrain 分布、也不代表我们关心的语言分布。**模型会去拟合 prompt 的统计形态**（"以 'Please' 开头"、"用户问 XX 类问题的频率"），泛化变差。

**(2) 灾难性遗忘的加速器**。Prompt token 在 SFT 数据里高度模板化（system prompt、客套话）。如果让模型背 prompt，它会在这些位置上做大量梯度更新，把 pretrain 学到的通用语言分布迅速覆盖掉。

**(3) 数据效率的浪费**。Loss 对 prompt token 的贡献是"额外的、与 alignment 无关的"——梯度被这部分稀释，真正想学的 response 行为反而学得慢。

→ Mask 掉 prompt 是**统计上正确的事**，不是"省 FLOPs 的小聪明"。

工程上的实现：

```
input_ids:   [SYSTEM] [USER PROMPT] [ASSISTANT RESPONSE] [EOS]
labels:      [-100  ] [-100        ] [response tokens   ] [EOS]
                ↑ ignore             ↑ compute CE
```

PyTorch `CrossEntropyLoss(ignore_index=-100)` 会跳过 labels 为 -100 的位置，**既不算 loss、也不算分母**（loss 用有效 token 数归一化）。

### 1.3 与 pretrain 的关系：同公式、异数据、异 mask

把 pretrain 和 SFT 并排放：

| 维度 | Pretrain | SFT |
|---|---|---|
| Loss 公式 | $-\sum_t \log \pi_\theta(u_t \| u_{<t})$ | 同上 |
| 数据 $u$ | 任意自然文本 | `(prompt, response)` 模板 |
| Mask | 全部计算 | 只算 response（含 EOS） |
| 数据量 | $10^{12}$+ tokens | $10^6$–$10^9$ tokens |
| LR | $2 \times 10^{-4}$ 量级 | $10^{-5}$ 量级（小 1-2 个量级） |
| Epochs | 1（数据量大不需多过） | 1-3 |
| 目的 | 学语言/世界模型 | 学输出格式 + 激发已有能力 |

→ SFT 在数学上**就是 pretrain 的特殊形态**，没有任何"魔法"。从这个视角看，本节剩下的内容都在回答："既然只是同一个 loss，为什么效果如此剧烈？"——答案在数据与初始化。

---

## 二、能力激发假说：SFT 不增加知识，只重定向行为

### 2.1 Superficial Alignment Hypothesis

[LIMA (Zhou 2023)](https://arxiv.org/abs/2305.11206) 用 **1000 条** 精挑的 SFT 数据微调 LLaMA-65B，得到与 InstructGPT 相近的 chat 表现。基于此，作者提出：

> **Superficial Alignment Hypothesis**：模型在 pretrain 阶段已经习得**几乎全部知识与推理能力**，alignment 只是教它"用什么子分布回复用户"。

把它形式化一点：base model 学到的是一个超大的条件分布族 $\{p_\theta(\cdot | x)\}_x$，里面已经包含"正确回答"这条模式（在 pretrain 的某些上下文里出现过），但**它和'继续胡说'、'问反问题'、'输出代码块' 等模式混在一起**。SFT 干的事情是把"产生 alignment 风格回复"这条模式的概率**抬到 dominate**。

为什么 1000 条够？因为：

- 你不是在教模型"什么是因式分解"（pretrain 已经会了）；
- 你只在教模型"看到数学问题 → 用解题格式回答"——这是一个**很窄的行为切换**，需要的信号量极少。

### 2.2 实证：SFT 灌新知识反而让模型变差

[Gekhman 2024 — Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904) 把 SFT 数据按"模型是否在 pretrain 见过这个事实"分桶，然后跟踪 fine-tune 过程：

- **见过的事实**：fine-tune 时 loss 快速下降，泛化 OK；
- **没见过的新事实**：loss 下降慢得多，最终模型在这些事实上准确率**反而下降**，同时 **hallucination 率显著上升**——模型学会的不是事实，而是"在这个 prompt 形态下，自信地编一个回答"。

这背后的机制：base model 内部没有这些事实的"表示"。强行 SFT 等于让模型把别的位置的参数挪过来拟合这些样本，副作用是**破坏了原本正确的表示**，让模型在所有相关事实上都变差。

→ **工程含义**：SFT 数据的 response 必须是 **pretrain 中已存在的知识形态**。想要加新知识，要么 continued pretraining（让模型有时间反复见到、形成稳定表示），要么 RAG（推理时检索）。

### 2.3 这条假说的边界

Superficial Alignment Hypothesis **不是说 SFT 完全没用**，也不是说"SFT 数据越少越好"。它的真实含义：

- SFT 的**主要贡献**是"行为重定向"——所以质量比数量重要得多。
- 一旦把行为对齐到，**再加更多同分布数据收益递减**（边际接近 0）。
- 它没有否定 RLHF 阶段的价值：偏好对齐、reasoning RL 仍是 SFT **无法替代**的（详见 §11.2-4）。

实证边界：LIMA 1k 样本能跑出"差不多能用"的 chat，但和 Llama-3 用 100w 精挑数据训出的模型仍有可见差距（IFEval/MT-Bench 差 5-10 个点）。所以工业上**不会**真用 1k 条 SFT 出货，但这个数字告诉你"**多到一定量后纯量上没用了**"。

---

## 三、对话模板（Chat Template）：special token 不是装饰

### 3.1 为什么必须有 chat template

base model 不知道 "system / user / assistant" 这种角色概念——它学的是连续文本流。要让它在生成时"看到 user 说话就切换到 assistant 模式回话"，必须**用 token 序列把角色边界编码进上下文**。

主流模板：

```text
ChatML（OpenAI / Qwen / Yi）：
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>

Llama-3：
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

...<|eot_id|><|start_header_id|>user<|end_header_id|>

...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

...<|eot_id|>

Gemma-2：
<start_of_turn>user
...<end_of_turn>
<start_of_turn>model
...<end_of_turn>
```

这些 `<|im_start|>` / `<|eot_id|>` / `<start_of_turn>` 是**专门的 single-token 特殊 token**（tokenizer 的 added_tokens），不会被切碎。它们在 SFT 阶段被赋予"角色切换"的语义，模型学会"看到 `<|im_start|>assistant` 后就用 assistant 子分布生成"。

### 3.2 train/inference 模板必须严格一致

工业上最常见的 SFT bug 是：训练用 ChatML，推理时不知道为什么用了 Llama 风格 prompt，结果模型"看上去能用但效果差 5-10 个点"。原因：

- 训练时模型把 `<|im_start|>assistant` 学成了"开始回话的触发器"；
- 推理时给它一个 `<|start_header_id|>assistant<|end_header_id|>`——这是**完全不同的 token id 序列**，模型在 pretrain/SFT 中都没在这个 context 下做过"开始回话"，行为退化到 base model 的弱 chat 表现。

→ **铁律**：训练和推理用同一个 `tokenizer.apply_chat_template()`，并用脚本对几条样本反序列化校验（"训练时这条样本喂进去的 token 序列，等不等于推理时 prompt 的 token 序列"）。

### 3.3 几个隐蔽细节

- **末尾 EOS / `<|eot_id|>` 是否计入 loss**：要算。否则模型不学"什么时候该停"，推理时会一直生成。
- **system prompt 不算 loss**：放在 mask 区，避免模型背诵 system prompt 内容。
- **assistant 起始 token 算不算 loss**：`<|im_start|>assistant\n` 这部分通常 mask 掉（它由 template 自动给出，不是 generation target），从 `\n` 之后开始算 loss。**TRL 的 `DataCollatorForCompletionOnlyLM` 默认就是这个边界**。

边界一旦切错（多 mask 一个 token / 少 mask 一个 token），有时不容易察觉但会让模型行为发生奇怪偏移。建议用单元测试：合成 10 条样本 → 检查 mask 后的有效 token 序列是不是恰好等于"原始 response + EOS"。

---

## 四、Loss masking 的工程实现与多轮处理

### 4.1 单轮：最简单的 mask

```python
input_ids = tokenize(system_prompt + user_prompt + assistant_response)
labels    = [-100] * len(prompt_part) + list(response_ids) + [eos_id]
# labels 与 input_ids 等长，长度 = T
# CrossEntropy 内部会做 shift：logits[:-1] 预测 labels[1:]
```

**注意 shift**：HuggingFace 的实现里 loss 计算前会把 logits 截掉最后一个位置、labels 截掉第一个位置——也就是说 `logits[t]` 预测 `labels[t+1]`。所以你写 labels 的时候**不需要**手动 shift，把它和 input_ids 对齐就行。

### 4.2 多轮对话：每个 assistant turn 都算 loss

错误做法："只在最后一轮 assistant 上算 loss，前面几轮当成上下文"——这浪费了一大半数据信号，多轮训练效率减半。

正确做法：把整段拼成一个序列，每个 assistant turn 的 token 都算 loss：

```
input:  [sys] [u1] [a1] [u2] [a2] [u3] [a3] [eos]
labels: [-1 ] [-1] [a1] [-1] [a2] [-1] [a3] [eos]
                    ↑       ↑       ↑
              三段都算 loss，每段算自己的 next-token CE
```

为什么这样在数学上是对的？因为每轮 `(u_i, a_i)` 在给定前文条件下是一个**独立的 conditional MLE 子目标**，把它们求和等价于对整段 chat 的条件似然——和 §1.1 的公式一致。

实现上有两种方式：

- **A. 一个 example 一条 sequence**：用一个长 mask 标出所有 assistant token。简单直接。
- **B. 把多轮拆成多条样本**（如 turn1, turn1+turn2, turn1+turn2+turn3 三条）：每条只算最后一轮的 loss。**这种方式数据膨胀严重且 attention 重复计算**，工业上极少用。

→ 一律用方式 A。

### 4.3 Packing：把多个短样本塞进一条长序列

SFT 数据的长度分布通常严重右偏：90% 样本短（< 1k token），少数样本长（4k+）。固定 padding 到 max_len 会让 90% 的算力浪费在 padding 上。

**Packing**：把多条样本头尾相接拼成一条接近 max_len 的长序列：

```
pack:    [seq1 ... <eos>] [seq2 ... <eos>] [seq3 ... <eos>] [pad ...]
labels:  [..response1..] [..response2..] [..response3..] [-100 ..]
```

但这里有一个**必须解决**的正确性问题：默认 causal attention 会让 `seq2` 的 token 看到 `seq1` 的所有 token（位置在它前面）。这种**跨样本污染**会让模型学到错误的关联（"看左侧别人的话来生成自己")，工业里观察到的退化包括：

- 模型生成时会重复前一条样本的话题；
- packing 数据 vs 非 packing 数据评测差 1-3 个点。

正确做法：**block-diagonal attention mask**（每条样本只能看自己）：

$$
M_{ij} = \begin{cases} 1 & \text{if } i, j \text{ 在同一个子样本内且 } i \ge j \\ 0 & \text{otherwise} \end{cases}
$$

这等价于把序列切成几个独立的"小三角形" attention 模式。

**工程实现**：

- **稠密实现**：构造 `T × T` 的 mask 矩阵传给 attention。简单但 O(T²) 内存，不实用。
- **FlashAttention varlen**：`flash_attn_varlen_func(qkv, cu_seqlens, max_seqlen)`——传入每条子样本的 `cumulative sequence lengths`，FlashAttention 内部自动按 block 切。**这是工业标准**。
- **Position id 重置**：每条子样本的 position id 要从 0 重新开始（不能让 seq2 拿到 position id 1500），否则模型用错 RoPE 频率。

`transformers` 的 `DataCollatorWithFlattening` + flash-attn 是当前主流栈，可以一行配好。

工程效果：packing 带来 **2-4× 训练吞吐**，几乎无质量损失（前提是 mask 与 position id 都做对）。

---

## 五、数据策略：SFT 数据的来源、配比、量级

### 5.1 数据来源谱系

| 类别 | 代表 | 优势 | 风险 |
|---|---|---|---|
| 人工标注 | OpenAI / Anthropic 内部 | 质量最高，标注一致性 | 慢、贵（$5-10 / pair） |
| Self-Instruct | Alpaca / WizardLM | 规模化便宜 | 模板化、错误传播 |
| 教师蒸馏 | OpenHermes、Magpie | 用 GPT-4/Claude 当老师，质量稳 | 法律/ToS 风险，蒸馏偏差 |
| 任务合成 | OpenMathInstruct / NuminaMath | 学科覆盖好 | 分布偏窄 |
| 真实对话日志 | ShareGPT、WildChat | 分布最真实 | 隐私 / 噪声 |
| Rejection sampling | Llama-3 内部 | 利用自己的策略产 data | 自我强化偏见 |

工业实战 SFT 数据通常是 **多源混合**：人工 + 蒸馏 + 真实日志 + rejection sampling。完全单源（如纯 Alpaca）已经过时。

### 5.2 高质量公开数据集（2024-2025）

- **Tülu-3 SFT mix** (AllenAI, 2024)：100w+ 条，覆盖 instruction、math、code、reasoning、safety。完整 pipeline 开源。
- **Magpie** (2024)：通过 prompt Llama-3-Instruct 自合成 1M+ 条 (prompt, response) 对，质量惊人地高。
- **OpenHermes-2.5**：GPT-4 蒸馏混合数据集，工业基线常用。
- **WildChat-1M**：真实 ChatGPT 用户对话日志，分布真实但噪声大。
- **OpenMathInstruct-2** / **NuminaMath**：数学专项。
- **Code-Feedback** / **OpenCodeInterpreter-SFT**：代码专项。

### 5.3 数据配比：多任务学习的负迁移

一个最容易踩的坑：以为"什么数据都加点对模型有好处"。多任务学习的现实是：**任务之间会互相伤害**（negative transfer），尤其当 task A 的 response 风格与 task B 完全不同时（如简短 chat vs 长 CoT）。

经验配比（Llama-3 / Tülu-3 / Qwen-2.5 报告综合）：

```
通用对话 / 日常 chat:       40-50%   (helpful 维度)
数学 + 代码（硬技能锚定）:   25-35%   (防止"软化"，保数学/代码能力)
推理 / CoT（think 风格）:    10-15%   (教 think 输出格式)
安全 / 拒答 / refusal:      5-10%    (harmless 维度)
长上下文（16K+ 文本）:       ~5%      (防长上下文退化)
工具调用 / function call:    ~5%      (可选)
```

为什么这个配比？背后的直觉：

- **硬技能数据要锚定**（数学/代码 25-35%）：否则会被通用对话"软化"——模型变得更礼貌但 GSM8K / HumanEval 掉点。Llama-3 团队在报告里明确提到要"持续重新加入" math/code 数据来防退化。
- **短回复 vs 长回复要平衡**：纯短回复数据训出的模型答数学题不写 CoT；纯长 CoT 数据训出的模型回答 "你好" 都写一段话。
- **安全数据不能太多**：超过 10% 会让模型过度谨慎、拒答率飙升（"over-refusal"），实用度变差。

实操：先按上面配比起步，跑评测后**找弱项加权**——比如数学掉点就把 math 比例提到 35-40%。

### 5.4 数据量经验

| 数据量 | 效果 | 谁在用 |
|---|---|---|
| 1k 高质 | LIMA 证明可行，"差不多能用"，但脆弱 | 学术 demo / 快速原型 |
| 10k-50k | 实用 chat 起点 | 中小团队、专业垂域微调 |
| 100k-500k | 当前工业主流 | Tülu-3、Qwen-2.5、Mistral |
| 1M 精挑 | 边际收益快速衰减 | Llama-3（明确强调"质量 > 量"） |
| > 5M | 几乎全是噪声，过拟合风险高 | 极少（除非数据极脏） |

**Llama-3 报告**特别强调：他们对 SFT 数据做了多轮过滤（safety / 长度 / 重复 / 模型自评），最终留下约 1M 高质条目，**比刚开始收集的体量小 10×**。"宁缺勿滥"是工业共识。

### 5.5 数据质量怎么衡量

不是看条数，而是看：

1. **去重率**：n-gram dedup 后剩多少（< 80% 留存说明源数据冗余严重）；
2. **长度分布**：是不是过于集中在某个窗口（说明合成数据模板化）；
3. **指令多样性**：用 embedding 聚类看类别数（< 50 类的数据集风险高）；
4. **教师质量**：蒸馏数据的教师模型是不是足够强（用 GPT-4o 蒸的 Tülu-3 比用 GPT-3.5 蒸的 Alpaca 高一个档次）。

工业常用工具：[lilac](https://github.com/lilacai/lilac) / [datatrove](https://github.com/huggingface/datatrove) 做交互式数据探索。

---

## 六、训练超参与流水

### 6.1 推荐超参（综合 Llama-3、Tülu-3、Qwen2.5 报告）

| 超参 | 推荐区间 | 说明 |
|---|---|---|
| **Learning rate** | 1e-5 ~ 5e-5 | 比 pretrain 小 1-2 量级；大模型偏小，小模型偏大 |
| **Batch size**（effective） | 64-256 sequences | 太大反而过拟合（小数据集尤其） |
| **Epochs** | 1-3 | 多了直接掉点（过拟合到样式） |
| **Warmup** | 3-5% | 前期梯度方向重要 |
| **Schedule** | Cosine / linear decay | Cosine 是工业默认 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, ε=1e-8) | 与 pretrain 同设 |
| **Weight decay** | 0.1 | 防过拟合 |
| **Gradient clip** | 1.0 | SFT 比 pretrain 更安全，少见 spike |
| **Sequence length** | 4K-16K | 视数据；超过 32K 用 long-context SFT |
| **Mixed precision** | BF16 | FP16 在 SFT 阶段也偶有 NaN，BF16 更稳 |
| **DropOut** | 0 | SFT 不需要 dropout（数据量够） |

### 6.2 Full SFT vs LoRA SFT

| 维度 | Full SFT | LoRA / QLoRA |
|---|---|---|
| 显存（70B） | ~700 GB（master + grad + Adam state） | ~80 GB（QLoRA） |
| 训练速度 | 1× | ~1× 训练步耗时近似（forward 一样） |
| 效果天花板 | 高 | 损失 1-3 点（IFEval / MT-Bench） |
| 灾难性遗忘 | 显著 | 几乎无（base 不动） |
| 多任务部署 | 一个 checkpoint 一个用途 | 多个 LoRA 适配同一 base |
| 适用 | 工业大厂 / 主力模型 | 中小团队 / 垂域 / 实验快迭代 |

经验：能 full 就 full；显存不够再 LoRA；多任务同 base 时 LoRA 不可替代。详见 §10.4 PEFT。

### 6.3 训练曲线监控

健康的 SFT loss 曲线应该：

```
loss
 |
 |\
 | \____________
 |              ─────────────────
 +─────────────────────────────────> step
   warmup    fast drop    平台期
```

- **Warmup 末（~3% step）**：loss 还在高位，正常。
- **Fast drop（3% → 30% step）**：loss 快速从 ~3 降到 ~1.5（GPT 风格 chat 数据）。如果你看到 loss 几乎不动，说明 LR 太小或 mask 错了。
- **平台期（30% step+）**：loss 缓慢下降，最终稳在 0.7-1.2 之间。

异常信号：

- **Loss 反弹**：通常是 LR 太大或 batch 太小，把 LR 减半重试。
- **Loss 突降到 < 0.3**：模型在记忆数据（过拟合或重复样本），检查 dedup。
- **Loss 跳跃**：脏数据（出现极端样本），检查 grad clipping 与数据。

---

## 七、SFT 的失败模式（机制与解药）

### 7.1 灾难性遗忘（Catastrophic Forgetting）

**现象**：SFT 后 base model 的某些能力下降——典型是长上下文检索能力、数学能力、罕见语种能力。

**机制**：连续学习里的经典问题。SFT 的梯度更新会改变 pretrain 学到的参数；如果 SFT 数据**不覆盖**某些 pretrain 学到的能力，这些能力对应的参数子空间会被无目的地"洗掉"。形式化：

- 设 pretrain 损失对参数 $\theta$ 在 $\theta^*$ 附近的二阶展开 Hessian 为 $H_{\text{pre}}$；
- SFT 优化朝梯度 $-\nabla \mathcal{L}_{\text{SFT}}$ 走，每步 $\theta^* \to \theta^* - \eta \nabla \mathcal{L}_{\text{SFT}}$；
- Pretrain 损失变化约为 $\Delta \mathcal{L}_{\text{pre}} \approx \frac{1}{2} \eta^2 \nabla \mathcal{L}_{\text{SFT}}^\top H_{\text{pre}} \nabla \mathcal{L}_{\text{SFT}}$；
- 当 $\nabla \mathcal{L}_{\text{SFT}}$ 落在 $H_{\text{pre}}$ 的**大特征值方向**（即 pretrain 的 critical directions）时，pretrain 能力损失最大。

**EWC（Elastic Weight Consolidation）**的思想就是用 Fisher information 矩阵近似 $H_{\text{pre}}$，惩罚这些方向上的移动。但 LLM 体量下 EWC 不实用，工业上用更粗的代理：

**解药 1（最常用）：Pretrain replay**。在 SFT 数据里混入 **5-10% 的 pretrain 数据**（或 Tülu-3 的 "FLAN replay"），相当于让梯度同时拉向 pretrain loss 的最小值。

**解药 2：LR 减小**。LR 越小、每步移动越小、pretrain loss 增长越慢（公式里的 $\eta^2$ 项）。

**解药 3：LoRA**。base 参数完全不动，灾难性遗忘几乎消失（但效果天花板低 1-3 点）。

**解药 4：early stopping on pretrain proxy**。每 N 步在 MMLU / 长上下文 RULER 上跑一下，发现退步立刻停。

### 7.2 过拟合到样式（Stylistic Overfitting）

**现象**：3+ epoch 后模型生成开头总是"Sure, I'd be happy to help you with that!"，结尾总是"Let me know if you have any other questions!"——明显模板化。

**机制**：SFT 数据如果在某些 surface form 上高度重复，模型会迅速学到这些"廉价的高概率模式"。CE loss 在重复模式上下降快，模型偷懒。

**解药**：

- **1-2 epoch 即停**（绝大多数 SFT 数据集都不该跑 3 epoch 以上）。
- **数据多样化**：换不同 system prompt、不同语气示例。
- **DPO 阶段抑制**：DPO/RLHF 可以惩罚这些模板化开头（chosen 是不带模板的、rejected 是带模板的）。

### 7.3 SFT 灌新知识引发 hallucination

详见 §2.2。机制总结：

- 新事实对应的"正确"表示在 base model 里不存在；
- SFT 让模型在该 prompt 下输出某个特定字符串，但因为缺少底层支撑表示，模型实际学到的是"在此类 prompt 下，自信地编一个合理的字符串"；
- 副作用：相邻事实（共享部分参数）也被波及，准确率全面下降，hallucination 率上升。

**解药**：

- SFT 数据**只**用 pretrain 已会的知识形态；
- 想加新知识 → continued pretraining 或 RAG；
- 如果必须 SFT 进新知识，至少要先做小规模 continued pretrain "灌输"，再 SFT 调样式。

### 7.4 多轮污染（Packing 没 mask 好）

跨样本 attention 没断开 → 模型学到"看左侧别人的话生成自己"。典型表现：模型在某些 prompt 上突然蹦出与上下文无关的话题（来自训练时被错误 attend 到的相邻样本）。

**解药**：FlashAttention varlen + 正确 cu_seqlens（§4.3）。验证：随机抽 100 条 packed batch，反查 attention pattern 是不是 block-diagonal。

### 7.5 EOS 不被学会

**现象**：模型生成时一直说不停（直到 max_new_tokens）。

**机制**：SFT 数据里 EOS / `<|eot_id|>` 没被算 loss，或者被错误 mask 掉了。

**解药**：明确把 EOS 列入 labels（不 mask）。这是 chat 模型最基础的 sanity check。

---

## 八、SFT 评测：别只看 loss

SFT 阶段的 loss 是 "在 SFT 数据上的拟合度"，**不直接反映 chat 质量**。需要外部评测覆盖几个正交维度：

| 维度 | 评测 | 检查什么 |
|---|---|---|
| **Instruction following** | IFEval, MT-Bench | 是否照格式回话 |
| **知识保留**（防回退） | MMLU, CMMLU | 数学/历史/法律等知识不掉 |
| **数学**（防回退 + 提升） | GSM8K, MATH | 是否数学能力被软化 |
| **代码** | HumanEval, MBPP | 同上 |
| **对话偏好** | AlpacaEval-2, Arena-Hard | LLM judge 主观偏好 |
| **安全** | ToxicChat, XSTest | 是否过度拒答 / 不当拒答 |
| **长上下文**（防退化） | RULER, NIAH | 长上下文检索/推理能力 |

**回退检查**比"刷上限"更重要——SFT 让你拿到一个 chat 模型，但如果它在 GSM8K 上从 80 掉到 65，那这次 SFT 就是失败的（即使 MT-Bench 上去了）。

详见 §13 评测章节。

---

## 九、SFT 在后训练流水中的位置

```
pretrain（§10）
    ↓
SFT（§11.1，本节）           ← 教格式、激发能力
    ↓
DPO / PPO（§11.2-3）        ← 偏好对齐
    ↓
RLVR / GRPO（§11.4）        ← 推理能力强化
    ↓
[可选] Agentic RL（§11.5）   ← 工具调用与多轮
    ↓
deploy
```

工业上常见的变体：

- **Llama-3 路径**：SFT → DPO → SFT(rejection sampling 自合成) → DPO → ...（迭代 ~6 轮）。
- **DeepSeek-R1 路径**：base → R1-zero（**跳过 SFT 直接 RL**，让 reasoning 自发涌现）→ 用 R1-zero 输出做 cold-start SFT → 再 RL → R1 → distill 到小模型。
- **Tülu-3 路径**：SFT → DPO → RLVR(GRPO with verifiable rewards)，完整开源。
- **Anthropic / OpenAI 内部**（推测）：SFT → CAI / RLAIF → 多轮 RLHF，含 deliberative alignment 阶段。

**有些团队尝试跳过 SFT 直接 DPO**：
- 适用场景：base model 已经"差不多能听话"（如 instruction-tuned base）；
- 风险：偏好优化的起点差，DPO 易不收敛；
- 实证：跳过 SFT 通常掉 5-10 个点。

---

## 十、关键问答

**Q1**：SFT 用多少数据合适？
- 1k 起步（LIMA 证明可行），10k-100k 实用，工业大厂 100k-1M 精挑。
- 质量 >> 数量。Llama-3 团队明确说"宁缺勿滥"，他们从 10M 候选筛到 ~1M。
- 超过 1M 后边际收益接近 0，过拟合风险升。

**Q2**：SFT 到底算不算"灌知识"？
- 不算。SFT 是"激发已有能力 + 教格式"（Superficial Alignment Hypothesis）。
- 灌新事实会加剧 hallucination（Gekhman 2024）——模型学不到事实只学到"自信编"。
- 想加知识 → continued pretraining（让模型反复见到形成稳定表示）或 RAG（推理时查）。

**Q3**：Full SFT vs LoRA SFT 怎么选？
- 资源够 + 数据多 + 求极致 → Full SFT；经验差距 1-3 点。
- 70B+ 单机 / 多 LoRA 共享 base / 防遗忘 → QLoRA / LoRA。
- 中等数据（10k）小模型上，两者差距常小到可忽略。

**Q4**：为什么只对 assistant 算 loss？
- 数学上：我们的目标分布是 $p(y|x)$，不是 $p(x,y)$。算 prompt loss 等价于让模型同时拟合 prompt 分布，泛化变差。
- 工程上：prompt 高度模板化，让模型背 prompt 会加速灾难性遗忘。

**Q5**：SFT 后用什么评测知道好不好？
- **回退检查**最重要：MMLU、GSM8K、HumanEval、RULER 不能掉。
- **能力提升**：IFEval、MT-Bench、AlpacaEval-2、Arena-Hard。
- 别只看 loss，loss 低不等于 useful。

**Q6**：多轮对话怎么 SFT？
- 拼成一个长序列，每个 assistant turn 都算 loss（§4.2）。
- 用 tokenizer 的 chat template 严格切分 role。
- 至少 4K-16K 上下文训，避免长上下文退化。

**Q7**：SFT 之后还要不要 RLHF？
- 一般要。SFT 只能学**示范**，不能学**偏好对比**；安全、helpfulness、reasoning 都靠 RL 阶段强化。
- DPO 是 SFT 后最常见的下一步（offline、便宜、稳定，见 §11.3）。
- 如果你只关心"基本能听话"，SFT 已经够；但只要想刷 leaderboard 或商用，RL 阶段就跑不掉。

**Q8**：SFT 训完模型生成开头总是"Sure, I'd be happy to help..."怎么办？
- 减 epoch（1-2 即可）；
- 数据多样化（不同 system prompt、不同语气）；
- 在 DPO 阶段把"Sure, I'd be happy..."当 rejected，"直接开始回答"当 chosen，几千条对就能拉过来。

**Q9**：为什么 packing 之后效果反而变差？
- 99% 是 attention mask 没切 / position id 没重置。
- 用 FlashAttention varlen + 正确 cu_seqlens 重新跑，应该恢复甚至更好。
- 还有 1% 是 token boundary 错位（packing 把多个样本拼接处的边界 token 算错了 loss）。

**Q10**：SFT 训 70B 模型显存预算大概多少？
- BF16 mixed precision + AdamW + ZeRO-3：约 16 × 70 = 1120 GB（master + grad + optimizer states），再 + activation + KV cache 约 200-400 GB。
- 单机 8×H100 (640 GB) 跑不动 → 需要 16-32 张 H100 + ZeRO-3 / FSDP。
- 用 QLoRA + 4-bit base + grad checkpoint，单机 8×H100 勉强能跑。

---

## 十一、参考资料

**开山 / 原典**：
- [InstructGPT (Ouyang 2022)](https://arxiv.org/abs/2203.02155) ⭐⭐（定义 SFT + RLHF pipeline 的开山）
- [LIMA — Less Is More for Alignment (Zhou 2023)](https://arxiv.org/abs/2305.11206) ⭐⭐（Superficial Alignment Hypothesis）

**理论 / 机制**：
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? (Gekhman 2024)](https://arxiv.org/abs/2405.05904) ⭐（SFT 灌知识反而坏的实证）
- [Continual Pre-training for LLMs — A Survey (Wu 2024)](https://arxiv.org/abs/2402.18158)（灾难性遗忘机制）

**工业报告（详尽的 SFT 实战）**：
- [Llama-3 技术报告 — Post-training §4](https://arxiv.org/abs/2407.21783) ⭐⭐（数据清洗、配比、迭代式 SFT-DPO）
- [Tülu-3 (AllenAI 2024)](https://arxiv.org/abs/2411.15124) ⭐⭐（完整开放后训练流水）
- [Qwen2.5 Technical Report (2024)](https://arxiv.org/abs/2412.15115)
- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437)

**数据**：
- [Magpie — Self-aligning Data Synthesis (Xu 2024)](https://arxiv.org/abs/2406.08464) ⭐
- [WildChat-1M (2024)](https://arxiv.org/abs/2405.01470)
- [OpenHermes-2.5 dataset card](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [OpenMathInstruct-2 (NVIDIA 2024)](https://arxiv.org/abs/2410.01560)

**工程**：
- [HuggingFace — Chat Templates 文档](https://huggingface.co/docs/transformers/main/chat_templating) ⭐
- [TRL — SFTTrainer 文档](https://huggingface.co/docs/trl/main/sft_trainer) ⭐
- [FlashAttention varlen API](https://github.com/Dao-AILab/flash-attention)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) ⭐（开源 SFT/DPO 训练框架，packing/multi-turn 都开箱）

**深度博客**：
- [Sebastian Raschka — Understanding SFT, Instruction Tuning, RLHF (2024)](https://magazine.sebastianraschka.com/) ⭐⭐
- [Nathan Lambert — Interconnects: Post-training notes](https://www.interconnects.ai/) ⭐
- [Lilian Weng — LLM Powered Autonomous Agents (含 alignment 段)](https://lilianweng.github.io/) ⭐
