# 11.6 Constitutional AI / RLAIF（规模化 AI 反馈）

[← 返回框架](../../README.md) · [📎 materials.md → §11.6](../../materials.md)

---

## 〇、本节回答什么

> Anthropic 的 Constitutional AI 是什么？为什么用 AI 替代人类做 reward？RLAIF 与 RLHF 在 2024-2025 怎么混用？Self-Rewarding / LLM-as-Judge 是什么关系？

人类标偏好数据**贵且慢**——10w 条 pair 标注成本数十万美元、几个月。用 LLM 当 annotator 可以把数据规模放大 100×，这是 **RLAIF (RL from AI Feedback)** 的核心动机。

```
RLHF:   人 → 偏好数据 → RM/DPO 训
RLAIF:  规则/原则 + LLM judge → 偏好数据 → RM/DPO 训
```

Constitutional AI (CAI) 是 Anthropic 把 RLAIF 系统化的一套方法：
1. **写一份原则文档**（"constitution"）
2. **让 LLM 按原则自我批评 + 自我修改**
3. **得到的修改对当成 DPO/PPO 的偏好数据**

---

## 一、为什么需要 RLAIF

### 1.1 人类反馈的瓶颈

| 问题 | 表现 |
|------|------|
| 成本高 | $1-5 per pair，工业 RLHF 数据集 $几十万 |
| 速度慢 | 几周到几月迭代周期 |
| 一致性差 | 标注员主观、累、疲劳 |
| 难规模化 | 模型升级 → 重标 |
| 偏见 | 标注员人群偏差 |
| 危险任务难标 | 红队、武器、CBRN 等内容人类不愿标 |

### 1.2 LLM 当 annotator 的可行性

[RLAIF (Lee et al. 2023, Google)](https://arxiv.org/abs/2309.00267)：实验表明 LLM 判官在偏好对齐任务上**与人类一致性 70-85%**，与"两个人类之间一致性 ~75%"接近。
→ LLM 已可替代/补充人类反馈。

---

## 二、Constitutional AI（CAI, Anthropic 2022）

[Bai et al. 2022](https://arxiv.org/abs/2212.08073)。CAI 分两阶段：

### 2.1 Stage 1：Supervised — Self-Critique & Revision

```
原始模型 ──(被刺激产生 unsafe response)→ y_bad
↓ 自我批评（按原则 P）:
"上面的回答违反了原则 P_i，因为 ..."
↓ 自我修改:
y_good
↓ 把 (prompt, y_good) 当 SFT 数据
```

例子：
```
P:  "回答不应包含暴力煽动"
原始: "如何制作炸弹..."  → y_bad
critique: "这违反了反暴力原则"
revision: "我不能提供制作爆炸物的指导..."  → y_good
```

→ 模型先学到"按原则改写不当回答"，这一步是纯 SFT。

### 2.2 Stage 2：RL from AI Feedback

```
prompt → policy 采 2 条 (y_1, y_2)
LLM judge（看原则）→ 选哪个更好
→ 偏好对 (y_w, y_l) → 训 RM → PPO（或直接 DPO）
```

LLM judge 的 prompt 大概是：
```
原则: ...
问题: ...
回答 A: ...
回答 B: ...
按原则判断哪个更好？
```

→ 同 RLHF 流水，但**所有偏好数据都由 LLM 生成**。

### 2.3 Constitution 内容（公开示例）

Anthropic 的 constitution 包含 16+ 条原则，例子：
- "选择更助人、诚实、无害的回答"
- "不选择会煽动暴力或仇恨的回答"
- "更倾向于不夸大自身能力的回答"
- 部分原则改写自 UN Declaration of Human Rights

Anthropic 2024 公开了完整 [constitution 文档](https://www.anthropic.com/news/claudes-constitution)。

---

## 三、RLAIF vs RLHF 对比

| 维度 | RLHF | RLAIF |
|------|------|-------|
| Reward 来源 | 人类标 | LLM 标 |
| 成本 | 高 ($数十万) | 低 ($千-万) |
| 速度 | 慢 (周-月) | 快 (天-周) |
| 一致性 | 标注员间差 | 同模型一致性高 |
| 上限 | 受标注员水平限 | 受 judge 模型限 |
| 偏见 | 标注员人群偏 | judge 模型偏 |
| 危险任务 | 难 | 容易（LLM 愿意标） |

→ **2024-2025 工业主流是混用**：少量高质人类数据 + 大量 LLM 数据。

---

## 四、RLAIF 的几种典型形态

### 4.1 Pure RLAIF（Google 2023）

整个 reward 全由 LLM judge 给。
- 流水：与 RLHF 完全一致，只是 annotator 换成 LLM
- 效果：与 RLHF 接近，成本低 10-100×

### 4.2 Self-Rewarding LM（Meta 2024）

[Yuan 2024](https://arxiv.org/abs/2401.10020)：**同一个模型既当 policy 又当 judge**。

```
Iter t:
  π_t  → 采样 G 条 → π_t 当 judge → 排序
  → 取 best/worst → DPO pair → DPO update → π_{t+1}
```

亮点：**没有外部 reward 源**，自己产生偏好数据自训。
风险：自我强化的偏见。

效果：Llama-2-70B 用 3 轮 self-rewarding → AlpacaEval 上接近 Claude 2 / GPT-4。

### 4.3 LLM-as-Judge for Eval

把 GPT-4 / Claude 当**评测器**：
- AlpacaEval-2、Arena-Hard、MT-Bench 全靠 LLM judge
- 详见 §13 评测

### 4.4 Constitutional AI（Anthropic）

完整两阶段（§二）。在 Claude 1/2/3 全程使用，Claude 3.5+ 更深化。

### 4.5 Synthetic Preference Data（工业大量使用）

- 用 GPT-4 / Claude 自动生成 (chosen, rejected) → 训 DPO
- 例：UltraFeedback、Magpie-DPO、HelpSteer2
- 现实成本：每条 pair $0.001-0.01

---

## 五、CAI / RLAIF 的关键技术

### 5.1 Judge prompt 设计

```
判官 prompt 模板:

[原则/criteria]
帮助性、诚实、无害...

[评估方式]
- 选择 A 更好 / B 更好 / 平局
- 给出原因

[输出格式]
{"winner": "A", "reason": "..."}
```

工程注意：
- **避免位置 bias**：A/B 顺序 随机化（GPT-4 倾向先看的）
- **避免长度 bias**：评测前 normalize 长度
- **多 judge 投票**：减少单 judge 噪声

### 5.2 Reward Model 替代

不需要训单独 RM：
- 直接让 judge prompt → label → DPO
- 训 RM：用 LLM-标的 pair 训 RM，仍可走 PPO（节省 RL 阶段 LLM 调用）

### 5.3 Iterative Self-Improvement

```
loop t = 0..N:
    π_t 当 judge → 标偏好 → 训 → π_{t+1}
```

风险：**偏见放大**。需要外部锚（人类抽检、外部 benchmark）防漂移。

### 5.4 Mixed Human + AI

```
人类数据:  1k 高质（"金标准"）
AI 数据:   100k 合成
loss = α · L(人类) + β · L(AI),  α >> β
```

→ 工业最常见配方（Llama-3、Tülu-3、Qwen 都是混用）。

---

## 六、LLM Judge 的失败模式

| 偏见 | 表现 | 缓解 |
|------|------|------|
| Position bias | 偏爱先看的 | 双向评估（A/B 和 B/A 都跑） |
| Length bias | 偏爱长回答 | 长度归一化 / 显式 prompt 反指 |
| Style bias | 偏爱 markdown / 项目符号 | 多样化 reference |
| Self-bias | judge 偏爱"和自己像"的 | 用不同模型当 judge ensemble |
| Easy-task bias | 难题判错率高 | 难题用人类 |
| 越狱风险 | 被 policy 用 prompt injection 攻击 judge | judge prompt 加防御 |

→ LLM judge 不是 ground truth，需要**外部锚 + 抽检**。

---

## 七、2024-2025 工业实战

### 7.1 数据合成 pipeline

```
1. Seed prompts (人类挑或公开数据)
2. Policy 采 N 条 response
3. LLM judge 排序 → 取最佳/最差 = (y_w, y_l)
4. 过滤（去重、低质过滤、安全过滤）
5. DPO / PPO 训练
```

典型规模：10w-100w pair / 轮，多轮迭代。

### 7.2 代表数据集

- **UltraFeedback (2023)**：6w prompts × 4 model responses × GPT-4 评分
- **Magpie-DPO (2024)**：完全自合成 prompt + DPO
- **HelpSteer2 (NVIDIA 2024)**：人 + AI 混合，开源
- **Tülu-3 DPO data**：8w 条，半合成

### 7.3 Anthropic Claude 3.5+ 据传做法

- CAI 升级：原则越来越细
- Constitutional Classifier（2025）：用 CAI 数据训的安全分类器
- "Many-shot jailbreaking" 防御中 RLAIF 起核心作用

### 7.4 OpenAI Deliberative Alignment（2024）

[Deliberative Alignment](https://openai.com/index/deliberative-alignment/) (o1 时代)：
- 在 CoT 中显式让模型"思考安全规则"
- 与 CAI 思路类似，但放在 reasoning 阶段
- → o1 拒答更精准、对 jailbreak 更鲁棒

---

## 八、关键问答

**Q1**：RLAIF 真的能达到 RLHF 效果吗？
- [Google 2023] 在 summary 任务上证明可以
- [Anthropic CAI] 在 helpfulness/harmlessness 上证明可以
- **不能完全替代**：高难任务（数学、reasoning）AI judge 出错率高，仍需人类

**Q2**：用同一个模型当 judge 会不会偏？
- 会。模型偏爱"和自己像"的输出
- 解：用更强模型当 judge（GPT-4 评 Llama-3 训练）
- 或：judge ensemble（多模型投票）

**Q3**：Constitution 该写得多细？
- 太抽象（"做个好 AI"）→ 信号弱
- 太具体（"不允许说 XYZ 词"）→ 容易过拟合 / 反噬
- Anthropic 经验：10-30 条平衡原则 + 例子

**Q4**：Self-Rewarding 会自我强化偏见吗？
- 会，所以需要外部锚（人类 holdout 验证 + 外部 benchmark）
- 实证：3-5 轮后收益迅速衰减
- 工业实战极少做超过 5 轮 self-rewarding

**Q5**：CAI 主要解决 helpful 还是 harmless？
- 起源主要为 harmlessness（"无害但不过度拒答"）
- 现在已扩到 helpful 全维度
- Claude 的"在拒答时仍 helpful"风格主要来自 CAI

**Q6**：用 GPT-4 标 DPO 训 Llama 合法/合规吗？
- 商业用：OpenAI ToS 不允许用 GPT 输出训 competing model
- 研究用：广泛使用，但社区伦理讨论持续
- 推荐用开源 judge（Llama-3-70B-Instruct、Qwen2.5-72B、DeepSeek-V3）

**Q7**：RLAIF 还能怎么进化？
- Process-level CAI：批评/修改 CoT 每一步
- Multi-modal CAI：图片 + 文本同时打分
- Constitutional Classifier：把 CAI 输出当下游分类器
- Deliberative Alignment：把"想原则"嵌入 reasoning

---

## 九、本节与第 11 章总结

```
§11.6 RLAIF / CAI (本节)
   │
   └─ 数据来源层面的革命：人类 → AI judge
      └─ 与 §11.2 PPO / §11.3 DPO / §11.4 GRPO 正交组合
         任何 RL 算法都可换上 RLAIF 数据源
```

### 第 11 章小结

```
§11.1 SFT                  → 教格式、激发能力（不灌知识）
§11.2 RLHF (PPO)           → 经典 RL 对齐，上限高工程贵
§11.3 DPO 家族              → offline 简化，2024 工业主流
§11.4 RLVR / GRPO          → verifiable reward + reasoning RL
§11.5 Agentic RL           → 多轮工具调用 + 长 horizon
§11.6 Constitutional / RLAIF → 数据规模化，3 阶段 pipeline 完整
```

**后训练的 2025 标准 pipeline**：
```
SFT (§11.1) → DPO/Iterative DPO (§11.3) → RLVR/GRPO (§11.4) → [可选 Agentic RL]
                                                  ↑
                                            RLAIF 提供数据 (§11.6)
```

下一章 §12 推理优化（KV cache 优化、speculative decoding、量化、调度），把训完的模型部署到生产。

---

## 十、参考资料

- [Constitutional AI (Bai 2022, Anthropic)](https://arxiv.org/abs/2212.08073) ⭐⭐（开山）
- [RLAIF (Lee 2023, Google)](https://arxiv.org/abs/2309.00267) ⭐⭐
- [Anthropic — Claude's Constitution 公开版](https://www.anthropic.com/news/claudes-constitution) ⭐
- [Self-Rewarding Language Models (Yuan 2024, Meta)](https://arxiv.org/abs/2401.10020) ⭐
- [UltraFeedback (Cui 2023)](https://arxiv.org/abs/2310.01377) ⭐
- [Magpie — Self-Aligning Data (Xu 2024)](https://arxiv.org/abs/2406.08464)
- [HelpSteer2 (NVIDIA 2024)](https://arxiv.org/abs/2406.08673)
- [Tülu-3 (AllenAI 2024)](https://arxiv.org/abs/2411.15124) ⭐⭐
- [OpenAI Deliberative Alignment (2024)](https://openai.com/index/deliberative-alignment/) ⭐
- [Constitutional Classifiers (Anthropic 2025)](https://www.anthropic.com/research/constitutional-classifiers)
- [LLM-as-Judge — A Survey (Gu 2024)](https://arxiv.org/abs/2411.15594)
- [Length Bias in LLM-as-Judge (Singhal 2023)](https://arxiv.org/abs/2310.03716)
- [Position Bias in LLM-as-Judge (Wang 2024)](https://arxiv.org/abs/2305.17926)
- [Lilian Weng — Adversarial Attacks on LLMs (2023)](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) ⭐
