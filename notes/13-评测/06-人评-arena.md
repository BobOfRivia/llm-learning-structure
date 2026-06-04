# 13.6 人评 / Arena（Chatbot Arena Elo / MT-Bench / Arena-Hard）

[← 返回框架](../../README.md) · [📎 materials.md → §13.6](../../materials.md)

---

## 〇、本节回答什么

> Benchmark 分数高 ≠ 用户喜欢用。人评（human preference）/ pairwise rating 是另一个**正交维度**——它测的是"在自由对话里你愿意继续用哪一个"。本节梳理：
>
> - **Chatbot Arena** 怎么把 pairwise vote → Elo / Bradley-Terry rating
> - **MT-Bench / Arena-Hard** 怎么用 LLM-as-Judge 离线复刻 Arena
> - **Style Control**：为什么"长 + markdown"会刷高 Arena 分
> - **对齐税 / 偏好 hack** 的争议

---

## 一、Chatbot Arena（LMSYS, 2023） — Elo 排行的来源

### 1.1 设计

```
1. 用户去 chat.lmsys.org（现 lmarena.ai） 提一个 prompt
2. 同时两个匿名模型 A、B 各自回答
3. 用户选 "A 更好 / B 更好 / 一样好 / 都不好"
4. 后台用 BT 模型拟合每个模型的 rating
5. 公开排行榜
```

### 1.2 累积数据规模

- 截至 2025 年初：**>2M 人类投票**
- 覆盖 **>200 个模型**
- 多语言、多任务（编程 / 数学 / 翻译 / 闲聊 / 创意写作 ...）

### 1.3 Bradley-Terry 模型

```
P(i 赢过 j) = exp(β_i) / (exp(β_i) + exp(β_j))
```

- β_i 是模型 i 的 latent rating
- 用 MLE 从全局对战记录拟合
- 然后归一化为 **Elo-like score**（基准 1000，每 100 分 ≈ 64% 胜率）

### 1.4 Elo 与 BT 的关系

- 国际象棋 Elo 是**在线增量更新**
- Arena 用 **离线 batch 拟合**的 BT model（更稳）
- 报数仍叫 "Arena Elo" 是约定俗成

### 1.5 现状（2025-初）

```
模型                            Arena Score
o1                              ≈1365
Gemini 2.0 Flash Thinking       ≈1360
Claude 3.5 Sonnet (new)         ≈1320
GPT-4o                          ≈1300
DeepSeek-V3                     ≈1305
DeepSeek-R1                     ≈1360
Llama-3.1-405B-Instruct         ≈1265
人类 reference                  N/A（Arena 是模型相对排名，不报人类）
```

### 1.6 Arena 子分类

按 task 切分：
- **Coding**
- **Math**
- **Long Query**（长 prompt）
- **Hard Prompts**（用户给"挑战题"）
- **Multi-turn**
- **中文 / 多语种**

→ 不同分项排名差异巨大：o1 在 Math / Coding 第一，但在创意写作 / 闲聊上低于 Claude。

---

## 二、Style Control（Arena 2024-10 重磅更新）

### 2.1 现象

发现 Arena 分数与"**回答长度 + markdown 排版**"高度相关：
- 长回答 win rate 高
- 有 bullet point / 加粗 / 表格 的 win rate 高
- → 模型可以**通过加长 / 加格式**刷 Arena 分

### 2.2 解决方案

LMSYS 引入 **Style-Controlled Rating**：

```
1. 把每条 prompt 的回答标注：长度 / markdown 元素数量
2. BT 模型加入 style 协变量
3. 算出"扣掉 style 影响"后的 rating
```

### 2.3 影响

启用 Style Control 后：
- GPT-4o 排名上升（它输出相对短/朴素）
- 部分长输出模型（早期 Llama-3-Instruct）下跌
- o1 / Claude 因输出长但内容硬核，整体仍居前

### 2.4 启示

> Arena 是有偏的——它测的是"人类**第一印象**偏好"，不完全等同于"真实能用"。
> 拿"用户投票"做 reward signal 时也要 style-correct，否则模型学会"长 + bullet"刷分。

---

## 三、MT-Bench（Zheng et al., NeurIPS 2023） — LLM-as-Judge 离线版

### 3.1 设计

- **80 道多轮对话**（写作 / 推理 / 数学 / 代码 / 角色扮演 / 提取 / STEM / 人文）
- 每题 2 轮
- 用 **GPT-4 作 judge** 打 1-10 分
- 报 **平均分** + 多模型对战

### 3.2 评分

```
单 score 模式：
  judge(model_a_response) ∈ {1..10}

pairwise 模式：
  judge(model_a, model_b) ∈ {A better, B better, tie}
```

### 3.3 现状

```
GPT-4:                  9.0+
Claude 3 Opus:          9.0+
绝大多数旗舰模型已 >8.5  → 区分度变差
```

→ MT-Bench 现已逐步退场，被 **Arena-Hard** 替代。

---

## 四、Arena-Hard（LMSYS, 2024-04） — Arena 的离线版

### 4.1 设计动机

- 让 Arena 排名**可离线复现**
- 用 GPT-4 当 judge（pairwise） vs 一个固定 baseline（GPT-4-0314）
- 选取 **500 道难 prompt**（用 LMSYS 真实 Arena 数据中"hard" cluster）

### 4.2 与 Arena 的对齐

Arena-Hard 与 Arena Elo 的 Spearman 相关 ≈ **0.97**（2024 测试）。

### 4.3 报数

```
Arena-Hard 报 "win rate vs GPT-4-0314"，例如：
GPT-4-Turbo:            82%
Claude 3.5 Sonnet:      82%
GPT-4o:                 79%
Llama-3.1-405B:         69%
```

### 4.4 Arena-Hard v2 (2024-08)

- 更新 baseline
- 加入 style control
- 难题更新

### 4.5 与 MT-Bench 的区别

| 维度 | MT-Bench | Arena-Hard |
| --- | --- | --- |
| 题目 | 80 | 500 |
| 难度 | 中 | 高（从真实 Arena 抽 hard） |
| 评分 | 单 score | pairwise vs baseline |
| 与 Arena 相关 | ≈0.85 | ≈0.97 |
| 风格控制 | ❌ | ✅ |

---

## 五、AlpacaEval / WildBench / AutoBench

### 5.1 AlpacaEval (Stanford, 2023) / v2

- 805 道 prompt
- GPT-4 当 judge pairwise vs GPT-4-Turbo baseline
- 报 **win rate** 和 **length-controlled win rate**（带 style 调整）
- 在 Llama-3 / Mistral 系列 paper 里常报

### 5.2 WildBench (AI2, 2024)

- 1024 道真实 user prompt（来自 WildChat 数据集）
- LLM-as-judge with reasoning
- 强调"真实分布"

### 5.3 AutoBench / Open Ko-LLM

- 多语言 LLM-as-judge benchmark

---

## 六、IFEval（Instruction Following Evaluation, Zhou et al., Google, 2023）

### 6.1 设计

- 测**指令遵循度**而非"质量"
- 541 道 prompt 含**可机械验证**的约束（"不要使用单词 X"、"答案恰好 3 段" ...）
- 25 种约束类型

### 6.2 评分

- **strict**（生成必须完全符合）
- **loose**（只要在容错范围内）
- 报 **prompt-level acc** 和 **instruction-level acc**

### 6.3 现状（Open LLM Leaderboard v2 必报）

```
GPT-4:                 80+
Claude 3.5 Sonnet:     85+
Llama-3.1 70B:         86
SOTA:                  >88
```

→ 旗舰模型饱和，但**小模型 / 蒸馏**用 IFEval 还能看出对齐质量。

---

## 七、LLM-as-Judge 的偏差 / 工程坑

### 7.1 同源偏见

- 用 GPT-4 当 judge → GPT-4 自己的回答更易被选 → **偏向同家族**
- 缓解：换 judge（Claude / Gemini 都当 judge 平均）

### 7.2 位置偏见

- pairwise 比较里，judge 偏向第一个出现的 response
- 缓解：**交换位置再判断**，结果取一致的

### 7.3 长度偏见

- judge 倾向"长 = 详细 = 好"
- 缓解：style control（Arena 做法）

### 7.4 格式偏见

- judge 倾向 markdown 排版好的
- 缓解：style control

### 7.5 Verbosity attack

- 模型对齐时可能学到"输出长 / 加 bullet"刷 judge → 训练 reward hack
- 缓解：训练时正则化长度

### 7.6 judge 漂移

- GPT-4 → GPT-4o → GPT-4-Turbo，不同 judge 给的分不稳定
- 缓解：**固定** judge 版本，所有报数都用同一 judge

---

## 八、Arena 与 benchmark 的关系

```
benchmark 高（MMLU/MATH/SWE-bench）  →  能力上限
Arena 高（人评）                      →  实际体验

两者并不强相关：
- o1 的 MATH ≈ R1 ≈ 95，但 Arena R1 ≈ o1，都 1360+
- Llama-3.1-70B 的 MMLU 80+，但 Arena 1245（中游）
- Claude 3 Haiku 的 MMLU 不算 SOTA，但 Arena 在小模型中靠前（用户喜欢它的"风格"）
```

### 8.1 "对齐税"争议

- Llama-3 / Mistral 一些迭代被指 "MMLU 提升但 Arena 不升"
- 原因：SFT 数据过窄 → 在 benchmark 上过拟合，在自由对话上失风格
- → 现代 post-training 必须**同时盯 benchmark 和 Arena**

### 8.2 reward model from preference

RLHF 训练用的 preference data 与 Arena 的 vote 是**同一类信号**。Arena 本身被业界用作**外部偏好评测**。但用 Arena 直接做 reward signal 有风险（style hack）→ 一般用作离线 eval。

---

## 九、关键问答

**Q1：Arena Elo 怎么算的？**
A：每次用户在两个匿名模型间投票 → 累积成 pairwise 对战记录 → Bradley-Terry 模型 MLE 拟合得到每个模型的 latent rating β → 归一化为 Elo-style 分数（基准 1000，差 100 ≈ 64% 胜率）。**不是国际象棋那种增量 Elo**，是离线 batch BT。

**Q2：Style Control 是什么？为什么重要？**
A：发现 Arena 投票与"输出长度 + markdown"高度相关 → 模型可以靠"加长加格式"刷分。Style Control 在 BT 拟合时加入 style 协变量，扣掉这部分影响。**启用后 Arena 排名更可信**。

**Q3：MT-Bench vs Arena-Hard 选哪个？**
A：**Arena-Hard**。MT-Bench (80 道) 题量少 + 现代模型饱和 + 与真实 Arena 相关性 0.85。Arena-Hard (500 道 + style control) 与 Arena 相关性 0.97，是 2024-2025 最常报的离线人评 proxy。

**Q4：用 LLM-as-judge 评测可信吗？**
A：**有条件可信**：
- 必须 **swap position** 消除位置偏差
- 必须 **style control** 消除长度/格式偏差
- 报数时 **固定 judge 版本**
- 跟人评保持 ≥0.9 相关性才算靠谱
- 同源（GPT-4 评 GPT-4）一定要警惕

**Q5：IFEval 测什么？为什么 Open LLM v2 加它？**
A：测**指令遵循的机械正确性**（不是质量）。例如"输出恰好 3 段"、"不能用单词 X"——可程序化检查。这是用来测 RLHF 后对齐是否破坏了模型基本"听话"能力，对小模型 ablation 尤其重要。

**Q6：Arena 高 ≠ benchmark 高 怎么解释？**
A：两者测的维度不同：
- benchmark 测**能力上限**（能不能解题）
- Arena 测**体验偏好**（用户喜不喜欢）
- 一个模型可能数学差但闲聊好（Claude 3 Haiku），或数学神但回答冷淡（早期 o1）
- 现代 post-training 必须**双盯**

**Q7：刷 Arena 的"窍门"有哪些？**
A：（虽然不应这么做，但要识别）
- 输出长一点
- 加 bullet point / 表格 / emoji
- 用 markdown
- 学一些"温暖语气"
- 故意展示"思考过程"
- → Style Control 部分缓解，但短期内仍有效，所以 Arena 排名要**结合 benchmark 一起看**。

---

## 参考资料

### 论文 / Benchmark
- ⭐⭐ [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference (LMSYS, 2024)](https://arxiv.org/abs/2403.04132)
- ⭐⭐ [MT-Bench / Judging LLM-as-a-Judge (Zheng et al., NeurIPS 2023)](https://arxiv.org/abs/2306.05685)
- ⭐⭐ [Arena-Hard-Auto (LMSYS Blog, 2024-04)](https://lmsys.org/blog/2024-04-19-arena-hard/)
- ⭐ [AlpacaEval / v2 (Stanford, 2023)](https://arxiv.org/abs/2404.04475)
- ⭐ [WildBench (Lin et al., AI2, 2024)](https://arxiv.org/abs/2406.04770)
- ⭐⭐ [IFEval (Zhou et al., Google, 2023)](https://arxiv.org/abs/2311.07911)
- ⭐ [Style Control in Chatbot Arena (LMSYS Blog, 2024-10)](https://lmsys.org/blog/2024-08-28-style-control/)
- ⭐ [Length-Controlled AlpacaEval](https://arxiv.org/abs/2404.04475)
- ⭐ [LLM Judges Are Biased — Survey](https://arxiv.org/abs/2402.10669)

### 排行榜与工具
- ⭐⭐ [lmarena.ai (Chatbot Arena Leaderboard)](https://lmarena.ai/)
- ⭐ [Arena-Hard-Auto GitHub](https://github.com/lmarena/arena-hard-auto)
- ⭐ [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- ⭐ [HuggingFace Open LLM Leaderboard v2 — IFEval](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

### 解读
- ⭐⭐ [Sebastian Raschka — How to evaluate LLMs](https://magazine.sebastianraschka.com/)
- ⭐⭐ [LMSYS Blog series（Arena/Style/MT-Bench 全部官方解读）](https://lmsys.org/blog/)
- ⭐ [What's wrong with Chatbot Arena? — debate articles 2024](https://blog.alignmentforum.org/)
- ⭐ [AI2 — WildBench design rationale](https://allenai.org/wildbench)

→ 下一节 [§13.7 评测污染](./07-评测污染.md)：n-gram overlap / canary / contamination report / 动态 benchmark。
