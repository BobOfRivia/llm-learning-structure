# 11.1 SFT（Supervised Fine-Tuning）

[← 返回框架](../../README.md) · [📎 materials.md → §11.1](../../materials.md)

---

## 〇、本节回答什么

> 怎么把 pretrained base model 调成"能听指令的 chat model"？SFT 数据怎么选/造？loss masking、packing、多轮怎么处理？数据量到底要多少？

SFT 是后训练 (Post-training) 的第一步，定位是**激发能力而非注入知识**——把 base model 里已存在的能力"对齐到 instruction-following 的输出格式"。

```
base model (next-token predictor)
        ↓  SFT (instruction tuning)
chat model (能听指令、按格式回复)
        ↓  RLHF / DPO / RLVR
aligned / reasoning model
```

---

## 一、SFT 的本质

### 1.1 数学形式

监督微调就是最大似然，标签为人类示范回答：

$$
\mathcal{L}_{\text{SFT}} = - \mathbb{E}_{(x, y) \sim \mathcal{D}} \Big[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \Big]
$$

- $x$：prompt（含 system / user / 历史多轮）
- $y$：目标 response
- **只在 response token 上算 loss**（prompt mask 掉）

### 1.2 SFT 不增加知识，只重定向行为

> "SFT 是把 base model 已经有的能力，导向 chat 格式"——这是 2023+ 共识

证据：
- [LIMA (Zhou 2023)](https://arxiv.org/abs/2305.11206)：1000 条高质量 SFT 数据，效果接近 InstructGPT
- [Superficial Alignment Hypothesis]：知识/能力在 pretrain 已习得，SFT 只学样式
- 反例：SFT 灌新知识会导致 hallucination 加剧（[Gekhman 2024](https://arxiv.org/abs/2405.05904) "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"）

→ 工程含义：**SFT 数据质量 >> 数量**

---

## 二、对话模板（Chat Template）

不同模型用不同 special tokens 划分角色：

```
ChatML (OpenAI / Qwen):
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>

Llama-3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
...<|eot_id|><|start_header_id|>user<|end_header_id|>
...<|eot_id|>

Gemma-2:
<start_of_turn>user
...<end_of_turn>
<start_of_turn>model
...<end_of_turn>
```

→ **训练和推理 template 必须严格一致**，否则掉点严重（template 错配是工业最常见 bug）。

HuggingFace `tokenizer.apply_chat_template()` 是事实标准。

---

## 三、Loss Masking

### 3.1 只对 assistant token 计算 loss

```
input:    [system] [user prompt] [assistant response] [eos]
label:    [-100  ] [-100        ] [response tokens   ] [eos]
                  ↑ 不计 loss      ↑ 计 loss
```

`-100` 是 PyTorch CrossEntropy 的 ignore_index。

### 3.2 多轮对话

```
轮1: [u1][a1][u2][a2][u3][a3]
loss:        [a1]    [a2]    [a3]      ← 三段都算 loss
```

每个 assistant turn 都参与 loss——比"只在最后一轮算"高效得多。

注意：**system prompt 不计 loss**，避免模型死记 system prompt 文本。

### 3.3 Packing（多个样本拼一个序列）

短样本 padding 浪费严重。Packing 把多个 (x, y) 拼到固定长度：

```
[seq1: u1 a1 eos] [seq2: u2 a2 eos] [seq3: u3 a3 eos] [pad]
```

- **必须配 attention mask 隔断**（防止 seq1 看到 seq2，否则跨样本污染）
- 现代框架：`packed_sequence` + flash-attn 的 `varlen` 接口
- 训练吞吐 ↑ 2-4×

---

## 四、SFT 数据策略

### 4.1 数据来源

| 类别 | 例子 | 备注 |
|------|------|------|
| 人工标注 | OpenAI / Anthropic 内部 | 质量最高，最贵 |
| Self-Instruct | Alpaca / WizardLM | 用 GPT-4 蒸馏 |
| 任务合成 | OpenMathInstruct / Magpie | 学科题 |
| 真实用户对话 | ShareGPT / WildChat | 分布最真实 |
| 难例挖掘 | Reject-sampling | 用更强模型筛 |

### 4.2 高质量 SFT 数据集（2024-2025）

- **Tülu-3** (AllenAI 2024)：完整后训练 pipeline 公开
- **OpenHermes-2.5** / **Magpie** (2024)：合成 + 真实混合
- **Dolma-instruct** (AllenAI)：完全开放
- **WildChat-1M** (2024)：真实用户对话，分布最真实
- **OpenMathInstruct-2** / **NuminaMath** (2024)：数学专项

### 4.3 数据配比经验

```
通用对话:  40-50%   (helpful chat)
数学/代码: 25-35%   (硬技能锚定)
推理/CoT:  10-15%   (think 风格)
安全/拒答: 5-10%    (refusal)
长上下文:  ~5%       (16K+ 样本)
```

→ 配比直接决定模型"性格"，不是越多越好。

### 4.4 数据量

| 数据量 | 效果 |
|-------|------|
| 1k (高质) | LIMA 证明可行，但脆弱 |
| 10k-50k | 实用 chat 起点 |
| 100k-500k | 当前工业主流（Llama-3、Qwen） |
| > 1M | 边际收益快速衰减，过拟合风险升 |

**Llama-3 报告**：精挑细选 1M 条 SFT，强调"宁缺勿滥"。

---

## 五、SFT 训练超参

| 超参 | 推荐 | 备注 |
|------|------|------|
| LR | 1e-5 ~ 5e-5 | 比 pretrain 小 1-2 量级 |
| Batch size | 64-256 (effective) | 太大反而过拟合 |
| Epochs | 1-3 | 多了直接掉点 |
| Schedule | Cosine / Linear | warmup 3-5% |
| Seq len | 4K-16K | 视数据 |
| Optimizer | AdamW | β1=0.9, β2=0.95 |
| Grad clip | 1.0 | |
| Loss spike | 极少（数据干净时） | |

**Full SFT vs LoRA SFT**：
- 工业大厂：full SFT（资源充足，效果天花板高）
- 中小团队：LoRA / QLoRA（见 §10.4）
- 经验：**LoRA SFT 比 full SFT 损失 1-3 个点**

---

## 六、SFT 的失败模式

### 6.1 灾难性遗忘（Catastrophic Forgetting）

SFT 后 base model 的某些能力可能下降（数学、长上下文）。

缓解：
- 在 SFT 数据里**混入预训练数据 5-10%**（replay）
- 用更小的 LR
- LoRA 替代 full SFT（保留原参数）

### 6.2 过拟合到样式

3+ epoch 后模型会模板化（开头总是 "Sure, I'd be happy to..."）。
→ 1-2 epoch 即可。

### 6.3 SFT 灌新知识引发 hallucination

Gekhman 2024：SFT 数据中含 pretrain 阶段未见过的事实 → 模型学会"瞎编合理回答"。
→ SFT 数据**不要加新知识**，只调样式。

### 6.4 多轮污染（packing 没 mask 好）

跨样本 attention 没切断 → 模型学到"看左侧别人的话"。
→ 用 flash-attn varlen 接口或显式 block-diagonal mask。

---

## 七、SFT 评测

SFT 阶段评测**不能只看 loss**，要看：

| 维度 | 评测 |
|------|------|
| Instruction-following | IFEval、MT-Bench |
| 知识能力 | MMLU、CMMLU（防回退） |
| 数学/代码 | GSM8K、HumanEval（防回退） |
| 对话偏好 | AlpacaEval-2、Arena-Hard |
| 安全 | ToxicChat、XSTest |
| 长上下文 | RULER（防退化） |

详见 §13 评测。

---

## 八、SFT 在后训练流水中的位置

```
pretrain (§10)
    ↓
SFT (§11.1)          ← 本节：教格式、激发能力
    ↓
DPO / PPO (§11.2-3)  ← 偏好对齐
    ↓
RLVR / GRPO (§11.4)  ← 推理能力强化
    ↓
deploy
```

**有些团队跳过 SFT，直接 DPO**：
- 适用：base model 已经"差不多能听话"
- 风险：偏好优化的起点差，DPO 难收敛

**Llama-3 路径**：SFT → DPO → SFT(rej-sampling) → DPO → ... 迭代
**DeepSeek-R1 路径**：base → R1-zero (纯 RL) → 重新 SFT → RLVR + SFT 混合

---

## 九、关键问答

**Q1**：SFT 用多少数据合适？
- 1k 起步（LIMA 证明可行），10k-100k 实用区间
- 工业大厂 100k-1M（数据干净 + 多样性高）
- 数据质量 >> 数量，宁缺勿滥

**Q2**：SFT 算不算"灌知识"？
- 不算。SFT 是"激发已有能力 + 教格式"
- 灌新事实 → 加剧 hallucination（Gekhman 2024）
- 想加知识 → continued pretraining 或 RAG

**Q3**：Full SFT vs LoRA SFT 怎么选？
- 资源够 + 数据多 + 求极致 → Full SFT
- 70B+ 单机 / 多任务多 LoRA / 防遗忘 → QLoRA / LoRA
- 经验差距 1-3 点

**Q4**：为什么只对 assistant 算 loss？
- 让模型学"在 user 输入下应该输出什么"，而不是学"如何复述 user prompt"
- prompt 上算 loss 会让模型背 prompt，泛化变差

**Q5**：SFT 后用什么评测知道好不好？
- 知识不掉（MMLU 不退步）
- 指令跟随提升（IFEval / MT-Bench）
- 不要只看 loss，loss 低不等于 useful

**Q6**：多轮对话怎么 SFT？
- 把整段拼成一个 sequence，每个 assistant turn 都算 loss
- 用 chat template 严格切分 role
- 上下文最好 4K-16K 训练，避免后期长上下文退化

**Q7**：SFT 之后还要不要 RLHF？
- 一般要。SFT 只能学示范，不能学偏好/对比
- 安全、helpfulness、reasoning 都靠 RL 阶段强化
- DPO 是 SFT 后最常见的下一步（见 §11.3）

---

## 十、参考资料

- [InstructGPT (Ouyang 2022)](https://arxiv.org/abs/2203.02155) ⭐⭐（开山，定义 SFT + RLHF pipeline）
- [LIMA (Zhou 2023)](https://arxiv.org/abs/2305.11206) ⭐（Superficial Alignment Hypothesis）
- [Llama-3 技术报告 — Post-training §](https://arxiv.org/abs/2407.21783) ⭐⭐（工业 SFT 详尽）
- [Tülu-3 (AllenAI 2024)](https://arxiv.org/abs/2411.15124) ⭐⭐（完整开放后训练流水）
- [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? (Gekhman 2024)](https://arxiv.org/abs/2405.05904) ⭐（SFT 灌知识反而坏）
- [Magpie — Self-aligning data synthesis (2024)](https://arxiv.org/abs/2406.08464)
- [WildChat: 1M ChatGPT Interaction Logs](https://arxiv.org/abs/2405.01470)
- [HuggingFace — Chat Templates 文档](https://huggingface.co/docs/transformers/main/chat_templating) ⭐
- [Sebastian Raschka — Understanding SFT](https://magazine.sebastianraschka.com/) ⭐
- [OpenAI Cookbook — Fine-tuning best practices](https://cookbook.openai.com/)
