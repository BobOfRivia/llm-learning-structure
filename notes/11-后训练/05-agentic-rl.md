# 11.5 Agentic RL（多轮工具调用 + 长程信用分配）

[← 返回框架](../../README.md) · [📎 materials.md → §11.5](../../materials.md)

---

## 〇、本节回答什么

> 让 LLM 当 agent（用工具、写代码、调 API、修 PR）需要怎么训？和单轮 reasoning RL 区别在哪？长 horizon 上 credit assignment 怎么做？2025 的代表工作是什么？

Agentic RL = **在多轮工具交互 trajectory 上做 RL**。比 §11.4 的 reasoning RL 多了：
- **外部环境**（terminal / browser / IDE / sandbox）
- **超长 trajectory**（数十轮、上万 token）
- **稀疏 reward + 长程信用分配**

代表场景：
- SWE-Bench：让 agent 修真实 GitHub bug
- WebArena / VisualWebArena：让 agent 在浏览器完成任务
- Computer Use（Anthropic 2024）：操作 GUI 完成任务
- Cursor / Devin：编程 agent

---

## 一、为什么 reasoning RL 不够

R1 范式（§11.4）：**single-turn**——给问题，输出 CoT，输出答案，校验。

Agentic 场景：
```
User: 帮我修复 issue #1234
Agent: <think>需要先看代码</think>
       [tool: read_file("src/auth.py")]
       <observation>... 代码内容 ...</observation>
       <think>问题在 23 行</think>
       [tool: edit_file("src/auth.py", 23, "if x == None")]
       [tool: run_tests()]
       <observation>5 passed, 1 failed</observation>
       <think>测试还有失败，再改</think>
       ...
       [tool: submit_pr()]
       <observation>PR merged</observation>
```

特点：
- **轨迹 = think + tool call + observation 的交替序列**
- 一次任务可能 50+ 轮、数万 token
- Reward 只在最终（PR merged / test passed）→ **极度稀疏**
- 中间任何一步出错都可能链式失败

---

## 二、Agentic Trajectory 的形式化

把 LLM agent 建模为 POMDP：
```
state s_t:   prompt + history (think + tool + obs)
action a_t:  next 输出 (think text 或 tool call)
env:         tool execution → observation o_{t+1}
reward r_T:  最终任务校验
```

LLM 既做"决策"也做"工具调用语法生成"。

每个 episode 是一个 token 序列：
```
[user] ... [think_1] [tool_1] [obs_1] [think_2] [tool_2] [obs_2] ... [done] [reward]
```

**Tool call 用结构化 token**（function calling）：
```json
{"name": "read_file", "arguments": {"path": "src/auth.py"}}
```

---

## 三、Agentic RL 的核心难点

### 3.1 长 horizon + 稀疏 reward

50 轮交互 → 中间任何一步坏掉 → 最终 reward = 0 → 无信号区分"哪一步错了"。

解：
- **过程奖励**：给中间步骤加 dense reward（unit test 数、文件覆盖率等）
- **MCTS / Best-of-N**：sample 多条 trajectory 取好的
- **Hindsight relabeling**：失败的 trajectory 也学（"如果当时输出 X 就成功了"）

### 3.2 超长 trajectory 显存

单条 trajectory 50k+ token + 大量 KV cache → rollout 阶段成主要瓶颈。

解：
- **Trajectory packing**（多 episode 拼）
- **PagedAttention / vLLM** 加速 rollout
- **离线 trajectory pool**：rollout 一次，多次 update

### 3.3 工具执行的 wall-clock 时间

每个 tool call 可能要几秒到几分钟（编译、跑测试、浏览页面）→ RL rollout 极慢。

解：
- 并行 N 条 trajectory（rollout 期间 LLM 可批处理）
- Tool 调用异步化（async sandbox）
- 大规模 trainer：DeepSeek / OpenAI 内部有专门的 trajectory cluster

### 3.4 工具执行的非确定性

同一 action 不同时刻 obs 不同（如网页变化）→ 训练不稳。

解：
- 用**确定性沙盒**（Docker snapshot、leetcode）
- 加 retry / averaging

---

## 四、Agentic RL 的算法层

### 4.1 GRPO + 外部工具（最常见）

DeepSeek-R1 / Qwen2.5-Coder-RL 的范式：
```
for each prompt:
    rollout G trajectories (含 tool calls)
    每条 trajectory 算 reward (verifier on final output)
    GRPO update
```

→ 即 §11.4 的 GRPO，只是 trajectory 里多了 tool/obs token。

**关键技巧**：训练时**mask 掉 observation token 的 loss**——observation 是环境给的，不是 model 输出的，不该算 loss。

### 4.2 Trajectory-level vs Token-level Importance Ratio

长 trajectory 上 token-level $\rho$ 容易方差爆炸（成百上千 token 累乘）。
→ 用 **GSPO**（§11.4）的 sequence-level ratio 或更激进的 trajectory-level normalize。

### 4.3 Dense Reward 设计

```
Final reward:  PR merged = 1
Process bonus: 
  - 每跑一次成功的 test:    +0.1
  - 编译通过:               +0.05
  - 引入新 syntax error:    -0.1
```

→ 信号密度上升，但 reward hacking 风险也上升（agent 学会刷 test 数而不真修 bug）。

### 4.4 Replay / Hindsight

[HER: Hindsight Experience Replay] 思路移植：
- 失败 trajectory：假装目标是"达到当前状态" → 仍能学
- 适用于 navigation / 编辑类任务

### 4.5 RL + Behavior Cloning 混合

许多 agent task 已经有专家 trajectory（如人类 PR 修复历史）：
```
loss = α · BC_loss(expert trajectory) + (1-α) · RL_loss(self trajectory)
```
- α 早期大（先学专家），后期减小（自己探索）
- Devin / Codex agent 据传都用类似混合

---

## 五、代表性工作（2024-2025）

### 5.1 OpenAI o-series / Operator

- o1 reasoning + tool use 整合
- Operator (2025)：浏览器 agent，被认为是 o3 + agentic 训练
- 内部用 RL on browser actions（细节未公开）

### 5.2 Anthropic Computer Use (2024)

[Anthropic Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use)：Claude 3.5 Sonnet 操作 GUI。
- 训练：截图 + click/type action
- RL 信号：任务完成校验
- 极慢（每 action 1-3 秒），实用度仍受限

### 5.3 SWE-Bench / SWE-Agent 类（2024-2025）

- **OpenHands** (formerly OpenDevin)：开源 agent 框架
- **SWE-Gym** / **SWE-RL**：RL training pipeline 修真实 GitHub bug
- 当前 SOTA：Claude Sonnet 4.5 / GPT-5 / o3 ~ 50-65% SWE-Bench Verified

### 5.4 Devin / Cursor Agent

商业闭源，但公开信息显示：
- 大量 supervised data + RL 微调
- 显式规划模块 + 工具 routing

### 5.5 Search Agent

- **Search-R1** (2025)：RL on retrieval + answer
- **WebGPT** (OpenAI 2021)：早期版本（RM + browser tool）
- **Reflexion**：把失败经验当 context 再 rollout（不严格 RL，但思路相关）

### 5.6 General Agentic RL Framework

- **OpenRLHF** / **VeRL** / **TRL**：支持 multi-turn rollout
- **Ray + sandboxed environments** 是工业标配

---

## 六、Trajectory Token Layout（实战细节）

R1-style + tool 的标准格式：
```
<|im_start|>system
You are a coding agent...
Available tools: ...
<|im_end|>
<|im_start|>user
Fix issue #1234
<|im_end|>
<|im_start|>assistant
<think>I need to read the file first</think>
<tool_call>
{"name": "read_file", "arguments": {"path": "src/auth.py"}}
</tool_call>
<|im_end|>
<|im_start|>tool
... file content ...
<|im_end|>
<|im_start|>assistant
<think>The bug is on line 23</think>
<tool_call>...</tool_call>
<|im_end|>
...
<|im_start|>assistant
Done. PR submitted.
<|im_end|>
```

训练 loss mask：
- `system` / `user` / `tool`：mask 掉（不算 loss）
- `assistant`（think + tool_call）：算 loss
- 长 trajectory 用 packed batching

---

## 七、Agentic RL 的工程栈

```
┌──────────────────────────────────────┐
│ Trainer  (GRPO / DAPO / GSPO)         │
├──────────────────────────────────────┤
│ Rollout  (vLLM + parallel agents)     │
├──────────────────────────────────────┤
│ Environment cluster                   │
│   ├─ sandboxed code exec (Docker)    │
│   ├─ browser (Playwright / Chrome)   │
│   ├─ filesystem ops                  │
│   ├─ shell terminal                  │
│   └─ RAG / search backend            │
├──────────────────────────────────────┤
│ Verifier (任务成败判定)               │
└──────────────────────────────────────┘
```

**Rollout 阶段 95% 的时间花在工具/环境上**，不在 LLM forward。
→ 环境基础设施是 agentic RL 的核心壁垒（OpenAI/Anthropic/Google 都自建）。

---

## 八、Agentic 评测

不能用单 prompt benchmark，要 trajectory-level：

| Benchmark | 任务 |
|----------|------|
| SWE-Bench Verified | 修真实 GitHub bug |
| TauBench | 多轮客服对话 |
| WebArena / VisualWebArena | 浏览器任务 |
| OSWorld | GUI 操作 |
| AgentBench | 综合多任务 |
| GAIA | 长程信息整合任务 |
| LiveCodeBench (agent mode) | 实时编程 |

详见 §13 评测。

---

## 九、关键问答

**Q1**：为什么 agentic RL 不能直接用 PPO？
- PPO 可以用，但 long-horizon 让 value head 训练很难
- GRPO 用 group baseline 替代 critic，对长 trajectory 更稳
- 2025 业界多用 GRPO / GSPO + dense reward

**Q2**：Reward 太稀疏怎么办？
- Dense process reward：跑通的 test 数 / 文件覆盖率 / 中间检查
- 课程学习：先简单任务，再难任务
- Hindsight：失败也能学

**Q3**：Agentic SFT 还是 RL？
- **SFT 是起点**：用人类 trajectory / 强模型 trajectory bootstrap
- **RL 是上限**：让模型自己探索更优策略
- 工业实战：SFT (BC) → RL 是标准 pipeline

**Q4**：trajectory 过长（>32k）怎么办？
- Cap max trajectory length（典型 32k-64k）
- 用 long context 训练（详见 §8 长上下文）
- 中段截断 + summary（影响连贯性）

**Q5**：如何防止 agent 在沙盒里"作弊"？
- 沙盒 read-only 关键文件
- Verifier 用**独立**程序（不让 agent 改 verifier）
- 多 verifier ensemble

**Q6**：tool calling 准确率怎么提？
- SFT 阶段足够工具调用示范
- function calling schema 用 JSON-mode 强制
- RL 阶段惩罚 invalid call（reward = -0.1）

**Q7**：开源 agent 模型距离 GPT-5/Claude 多远？
- 2025 中：SWE-Bench Verified 开源最佳 ~30-40%，闭源 50-65%
- 主要差距：训练数据规模、环境基建、长程稳定性
- 蒸馏 + 社区基建（SWE-Gym 等）在缩小差距

---

## 十、本节与其他节关系

```
§11.4 RLVR / GRPO ──→ §11.5 Agentic RL (本节)
       │                    │
       │ 单轮 reasoning      │ 多轮 + 工具
       │ verifiable reward   │ 长 horizon
       │                    ↓
       │                §11.6 RLAIF (规模化 reward signal)
       │
§8 长上下文 ─→ trajectory 装得下
§9 MoE     ─→ 大 agent 模型常用 MoE 节省推理
```

---

## 十一、参考资料

- [Reflexion (Shinn 2023)](https://arxiv.org/abs/2303.11366)（trajectory 反思早期工作）
- [SWE-Bench (Jimenez 2023)](https://arxiv.org/abs/2310.06770) ⭐⭐
- [SWE-Bench Verified (OpenAI 2024)](https://openai.com/index/introducing-swe-bench-verified/) ⭐
- [SWE-Agent (Yang 2024)](https://arxiv.org/abs/2405.15793)
- [OpenHands (formerly OpenDevin)](https://github.com/All-Hands-AI/OpenHands) ⭐
- [WebArena (Zhou 2023)](https://arxiv.org/abs/2307.13854)
- [VisualWebArena (Koh 2024)](https://arxiv.org/abs/2401.13649)
- [Anthropic Computer Use blog](https://www.anthropic.com/news/3-5-models-and-computer-use) ⭐
- [Computer Use 论文 OS-Atlas / ShowUI 等](https://arxiv.org/abs/2410.23218)
- [Tau-Bench (2024)](https://arxiv.org/abs/2406.12045)
- [AgentBench (Liu 2023)](https://arxiv.org/abs/2308.03688)
- [GAIA (Mialon 2023)](https://arxiv.org/abs/2311.12983)
- [Search-R1 (2025)](https://arxiv.org/abs/2503.09516)
- [OpenRLHF / VeRL (开源 agent RL trainer)](https://github.com/OpenRLHF/OpenRLHF) ⭐
- [Lilian Weng — LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) ⭐⭐
- [Survey of LLM Agents (Wang 2024)](https://arxiv.org/abs/2308.11432)
