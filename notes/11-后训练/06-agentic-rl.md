# 11.6 Agentic RL（多轮工具调用 + 长程信用分配）

[← 返回框架](../../README.md) · [📎 materials.md → §11.6](../../materials.md) · [⇡ 章节导论](./00-章节导论.md)

---

## 〇、本节回答什么

> 让 LLM 当 agent（用工具、写代码、调 API、修 PR、操作浏览器）需要怎么训？和单轮 reasoning RL 区别在哪？长 horizon、稀疏 reward 上 credit assignment 怎么做？2024-2025 的代表工作是什么？为什么 agentic RL 的瓶颈不在 LLM 而在环境基建？

**本节在第 11 章的位置**（[§11.0 导论](./00-章节导论.md) 中的"算法主线终点"）：

- **算法主线终点**：Agentic RL = §11.5 GRPO 算法 + **多轮 trajectory** + **环境/工具栈**；
- **来自 §11.5**：算法层基本沿用 GRPO/DAPO/GSPO，几乎不动；
- **新增三件事**（也是本节的全部新难点）：
  1. 数据形态从单轮 CoT → 多轮 (think + tool + obs) 交错——loss mask 要 mask 掉 obs token；
  2. Trajectory 从 < 10K → 数万 token——KV cache、显存、长 horizon credit assignment 全变成主要矛盾；
  3. Rollout 从一次 forward → 等环境（编译、跑测试、浏览器）——**工程瓶颈从 LLM 转到 environment 集群**；
- **上游能力依赖**：§8 长上下文（trajectory 装得下）、§11.5 GRPO（算法基础）、§11.1 SFT（行为克隆 cold-start 起点）；
- **下游**：§13 评测的 agentic benchmark（SWE-Bench / WebArena / GAIA）、§12 推理 serving（生产环境的 agent 调度）；
- **正交话题**：训练数据（专家 trajectory）仍可用 §11.7 RLAIF 风格自动合成；显存仍可用 §11.2 PEFT。

→ 读法建议：**本节强烈依赖 §11.5**——如果 GRPO 还没掌握，请先回到 §11.5 再来本节。本节的算法层只是"GRPO + 多轮 trajectory 改造"，**主要篇幅在工程**。

Agentic RL 是 **§11.5 reasoning RL 在多轮工具交互轨迹上的自然延伸**：

```
§11.5 reasoning RL:  问题 → 长 CoT → 答案 → verify (single turn)
§11.6 agentic RL:    任务 → think → tool → obs → think → tool → ... → 完成 → verify (多 turn)
```

相比单轮 reasoning，agentic 多了三件事：

1. **外部环境**（terminal / browser / IDE / sandbox）——每次 tool call 都要真的执行；
2. **超长 trajectory**（数十轮、数万 token）——比单轮 CoT 长 5-10×；
3. **稀疏 reward + 长程信用分配**——只在最终任务完成时有 reward，中间几十步无信号。

代表场景：

- **SWE-Bench**：修真实 GitHub bug；
- **WebArena / VisualWebArena**：浏览器任务；
- **OSWorld**：GUI 操作；
- **Anthropic Computer Use**：操作桌面环境；
- **Devin / Cursor / Claude Code**：编程 agent；
- **Operator (OpenAI)**：浏览器 agent。

本节按这个顺序展开：

1. **为什么 reasoning RL 不够**——单轮和多轮的本质差别。
2. **形式化**：把 agent 建模为 POMDP，trajectory 的 token layout。
3. **核心难点**：长 horizon credit assignment 的数学障碍。
4. **算法**：在 GRPO/PPO 上的多轮扩展、observation mask、dense reward 设计。
5. **工程栈**：rollout / environment / verifier 三层架构，瓶颈分析。
6. **代表性工作 2024-2025**：SWE-Bench、Computer Use、Search-R1 等。

---

## 一、为什么 reasoning RL 不够

### 1.1 单轮 vs 多轮

R1 范式（§11.5）：**single-turn**——给问题，输出 CoT + 答案，verify。整段 trajectory 是模型内独立产生的 token 序列。

Agentic 场景：**multi-turn**——模型产生 action（tool call），环境产生 observation，模型基于 observation 产生下一个 action：

```
User: 帮我修复 issue #1234
Agent: <think>需要先看代码</think>
       [tool: read_file("src/auth.py")]
<env>: <observation>... 代码内容 ...</observation>
Agent: <think>问题在 23 行</think>
       [tool: edit_file("src/auth.py", 23, "if x is None")]
       [tool: run_tests()]
<env>: <observation>5 passed, 1 failed</observation>
Agent: <think>测试还有失败，再改</think>
       ...
       [tool: submit_pr()]
<env>: <observation>PR merged</observation>
```

特点：

- **轨迹 = think + tool call + observation 的交替序列**——agent 控制 action，环境控制 observation；
- **一次任务可能 50+ 轮、数万 token**；
- **Reward 只在最终**（PR merged / test passed）——极度稀疏；
- **中间任何一步出错都可能链式失败**——长 horizon 让单步错误被放大。

### 1.2 三个新挑战

R1 范式无法直接处理：

| 挑战 | R1 单轮 | Agentic 多轮 |
|---|---|---|
| Trajectory 来源 | 模型 forward 一次性产生 | 模型 + 环境交替产生 |
| Reward 时点 | 最后一个 token | 整个 trajectory 末尾 |
| Loss 计算 | 全部 token 算 loss | observation token 不能算 loss |
| Credit assignment | 整段 trajectory 共享 advantage | 50+ 步的 advantage 不同步 |
| Rollout 成本 | 一次 forward | 每步等环境（秒-分钟）|
| 训练显存 | 单条 CoT 数千 token | 单条 trajectory 数万 token |

这些差异让 agentic RL 在算法、数据、工程层面都需要新设计。

---

## 二、Agentic Trajectory 的形式化

### 2.1 POMDP 视角

把 LLM agent 建模为 **POMDP (Partially Observable Markov Decision Process)**：

```
state s_t:       prompt + 全部 history (think + tool call + obs)
action a_t:      next 输出（think text 或 tool call）
observation o:   tool 执行结果
transition:      env(a_t) → o_{t+1}
reward r_T:      最终任务校验（terminal reward）
```

LLM 既做"决策"也做"工具调用语法生成"——同一个 forward 既产生 think 文本也产生 JSON 格式的 tool call。

策略 $\pi_\theta(a_t | s_t)$ 是 LLM 在历史上的 next-token 分布——本质和 single-turn 一样，只是 state 里包含了 observation。

### 2.2 Trajectory 的 token layout

R1-style + tool calling 的标准格式（ChatML）：

```text
<|im_start|>system
You are a coding agent...
Available tools: read_file, edit_file, run_tests, submit_pr
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
... file content (来自环境) ...
<|im_end|>
<|im_start|>assistant
<think>The bug is on line 23</think>
<tool_call>{"name": "edit_file", ...}</tool_call>
<|im_end|>
<|im_start|>tool
File edited successfully
<|im_end|>
...
<|im_start|>assistant
Done. PR submitted.
<|im_end|>
```

每条 trajectory 是一长串这种交替结构。

### 2.3 Loss mask：observation 不能算 loss

**关键工程细节**：训练时 **observation token（来自环境的）必须 mask 掉**——它们不是模型输出，不该算 loss。

```
input_ids: [sys] [user] [assistant1] [tool obs1] [assistant2] [tool obs2] ... [last assistant]
labels:    [-1 ] [-1  ] [a1 tokens ] [-1       ] [a2 tokens ] [-1       ] ... [last a tokens]
                          ↑算 loss    ↑不算       ↑算         ↑不算
```

为什么？因为：

- Observation 是环境给的——你不能"让模型学如何生成 observation"；
- 如果不 mask，模型会试图拟合 observation 的统计形态，相当于灌入环境噪声；
- Observation 中可能含 token id 漂移（如代码内容有特殊字符），会污染梯度。

这一步在 SFT 阶段（imitation learning）和 RL 阶段都一样关键。**错了模型就废**——典型表现：agent 生成 tool call 后直接接着"幻觉式补 observation"，绕过环境。

### 2.4 Trajectory packing 的复杂性

R1-style 单 CoT packing 用 block-diagonal mask（§11.1 §4.3）。Agentic trajectory 有更多复杂度：

- 同 trajectory 内**不能** block-diagonal（assistant 必须能 attend 到前面所有 obs）；
- 跨 trajectory **必须** block-diagonal（不同任务不能互相 leak）；
- 加上 position id 重置：每 trajectory 从 0 开始。

工程上用 FlashAttention varlen + cu_seqlens（与 §11.1 相同接口）。但 batch 内 trajectory 长度差异极大（10K-50K），padding 浪费仍是问题。

---

## 三、长 horizon 信用分配（credit assignment）

### 3.1 数学难点

长 horizon RL 的根本难题是：**reward 只在 $T$ 时点给，前面 50 步的 action 怎么各自被 credit？**

经典 REINFORCE 把整条 trajectory 的 reward 摊到每个 token 上：

$$
\nabla J = \mathbb{E}\!\left[ \sum_t \nabla \log \pi_\theta(a_t | s_t) \cdot R \right], \quad R = \text{trajectory reward}
$$

问题：每个 $a_t$ 都被同一个 $R$ "credit"——好 step 和坏 step 拿到同样的 reward 信号，**梯度信号被稀释 $T$ 倍**。

**用 critic 给 step-level value**（PPO 的方法）：每个 $a_t$ 拿到 $A_t = Q(s_t, a_t) - V(s_t)$，但 critic 在 long horizon 上**难训**：

- $V(s_t)$ 的训练目标是 $\mathbb{E}[R | s_t]$；
- 在 50 步深处的 $s_t$ 看到 $R$ 时已经过了 50 步——signal 极弱；
- Critic 早期 50 步深的 $V$ 几乎是噪声，policy 拿到噪声 advantage 学不到东西。

→ Long horizon 是 **critic-based RL 最难的场景**。

### 3.2 实战策略

**(1) 用 GRPO 的 group baseline，避免 critic**

GRPO 把整条 trajectory 共享一个 advantage（来自 group baseline）。虽然丢失了 step-level 信号，但**整条 trajectory 是 unit**——这种粒度损失在长 horizon 上反而**稳**（critic 噪声大于 group baseline 的粒度损失）。

DeepSeek-R1 / Qwen2.5-Coder-RL / Tülu-3-RLVR 多数用 GRPO + trajectory-level advantage。

**(2) 加 dense reward（process reward）**

让中间步骤也有 reward 信号：

```
Final reward:  PR merged = 1
Process bonus: 
  - 每跑一次成功的 test:    +0.1
  - 编译通过:                +0.05
  - 引入新 syntax error:    -0.1
```

→ 信号密度上升，credit assignment 变浅（每个 action 不需要等 50 步才有信号）。

但**风险**：dense reward 比 sparse reward **更易被 hack**——agent 学会刷 test 数（多写无关测试）、刷编译次数（不停 syntactically 合法但语义无用的修改）。

平衡：dense reward 用作"辅助信号"（小权重，如 0.1），terminal reward 仍是主要信号（权重 1.0）。

**(3) Hindsight Relabeling（HER）**

[Hindsight Experience Replay (Andrychowicz 2017)](https://arxiv.org/abs/1707.05952) 的思路：失败 trajectory 也能学。

- 失败 trajectory $\tau$ 的实际 outcome 是 $o$（不是目标 $g$）；
- "如果目标本来是 $o$，这条 trajectory 是成功的"——把 $g$ 重标为 $o$，对 trajectory 给 reward = 1；
- 让模型在失败数据上学到 "navigate 到 $o$" 的策略。

在 agent 场景：模型修了一个 bug 但不是 issue 要求的那个 → 把"修了哪个 bug"当成目标重标，仍能学。

适合 navigation、编辑类任务，对开放式任务（"帮我做一份 PPT"）不适用。

**(4) RL + Behavior Cloning（BC）混合**

许多 agent task 已有专家 trajectory（人类 PR 修复历史、Stack Overflow 解答）：

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BC}}(\text{expert trajectory}) + (1-\alpha) \cdot \mathcal{L}_{\text{RL}}(\text{self trajectory})
$$

- $\alpha$ 早期大（先学专家），后期减小（自己探索）；
- 解决"冷启动 reward 全 0"问题（专家 trajectory 给个起点）；
- Devin / Cursor / 各大 SWE agent 都用类似混合。

实务比例：开始 $\alpha = 0.8$，逐步减到 $0.2$。

---

## 四、Agentic RL 算法层

### 4.1 GRPO + 外部工具（最常见的范式）

DeepSeek-R1 / Qwen2.5-Coder / Search-R1 等都用这个范式：

```
for each prompt (task):
    rollout G trajectories (含 tool calls + obs)
    每条 trajectory 算 reward (verifier on final outcome)
    GRPO loss + observation mask
    update
```

→ 即 §11.5 的 GRPO，trajectory 里多了 tool/obs token，loss mask 配合调整。

实现要点：

- **Rollout 是异步的**：每条 trajectory 独立，并行 N 条以提升 throughput；
- **Verifier 在终态判定**：例如 SWE-Bench 用 hidden test cases；
- **Observation token mask = -100**（不算 loss）；
- **KL 项**：相对 SFT ref 模型算（包含 obs token？通常只在 assistant token 上算 KL）。

### 4.2 Trajectory-level vs Token-level Importance Ratio

长 trajectory 上 token-level $\rho$ 容易方差爆炸（成千 token 累乘）。

**解 1**：用 **GSPO**（§11.5 §五）的 sequence-level ratio，每条 trajectory 一个 $\rho_{\text{seq}}$。

**解 2**：更激进的 **trajectory-level normalize**——按 trajectory 平均 log-prob 算 ratio，整条共享 clip。

**解 3**：**Token-level loss + 长度归一化**（DAPO 风格）——保留 token-level grad，但用 batch 总 token 数做分母。

工业现状：**GSPO 在 agentic 场景逐渐成主流**（Qwen-Agent / Devin 据传都用）。

### 4.3 Dense Reward 设计的取舍

dense reward 公式举例（SWE 任务）：

$$
r(\tau) = \underbrace{r_{\text{merge}}(\tau)}_{\text{terminal, weight 1.0}} + 0.1 \cdot \underbrace{\#\{\text{passed tests}\}}_{\text{process}} + 0.05 \cdot \underbrace{\#\{\text{successful compiles}\}}_{\text{process}} - 0.1 \cdot \underbrace{\#\{\text{new syntax errors}\}}_{\text{process}}
$$

设计原则：

- **Terminal reward 主导**（权重 ≥ 0.5）；
- **Process reward 是辅助**（权重 0.05-0.2）；
- **Process reward 必须可被 hack 时反向**（如 "新 syntax error" 给负 reward，防止 agent 不停 try-fail）；
- **总和有界**（normalize 到 [0, 2] 类似区间），避免 dense reward 累加压过 terminal。

### 4.4 Replay Buffer 与 off-policy 元素

纯 on-policy GRPO 每轮 rollout 后只用一次（multi-epoch 2-4）。Agentic RL 因为 rollout 极慢（每条 trajectory 几分钟），希望**重用 rollout 数据**——引入 replay buffer：

```
buffer:  最近 N 轮的 trajectory
sample: 50% 当前 rollout + 50% buffer 中的旧 trajectory
```

但 buffer 中的 trajectory 是用旧 policy 采的——严格说是 off-policy，importance ratio 已经漂得很远。

平衡：

- Buffer size 不太大（最近 5-10 轮）；
- 加 importance correction（PPO-style clip）；
- 旧数据权重小（buffer 的 advantage 乘 0.5 等）。

### 4.5 Tool-call validation reward

实战中 tool call 格式错误（JSON 解析失败）是高频问题。加 reward shaping：

```
invalid tool call (JSON 解析失败) → reward -0.5
valid tool call but wrong arguments → reward -0.1
valid + successful tool execution → reward 0（不奖励，避免刷工具调用）
```

→ agent 学会"先确保 tool call 合法，再追求语义正确"。

---

## 五、Agentic RL 的工程栈

### 5.1 系统架构

```
┌──────────────────────────────────────┐
│ Trainer  (GRPO / DAPO / GSPO)        │  ← LLM 训练（PyTorch + ZeRO/FSDP）
├──────────────────────────────────────┤
│ Rollout  (vLLM + parallel agents)    │  ← LLM 生成（vLLM 多实例并发）
├──────────────────────────────────────┤
│ Environment cluster                  │
│   ├─ sandboxed code exec (Docker)    │
│   ├─ browser (Playwright / Chromium) │
│   ├─ filesystem ops                  │
│   ├─ shell terminal                  │
│   └─ RAG / search backend            │
├──────────────────────────────────────┤
│ Verifier (任务成败判定)              │  ← 独立进程，agent 不可改
└──────────────────────────────────────┘
```

每层各自的瓶颈：

- **Trainer**：显存 + 通信（与 single-turn RL 同）；
- **Rollout (LLM)**：throughput——并发 N 条 trajectory 时，LLM forward 可批处理（vLLM 连续 batching 是关键）；
- **Environment**：**wall-clock 时间**——tool 执行真的需要时间（编译 30s、跑测试 60s、浏览器加载 5s）。

### 5.2 Rollout 阶段 95% 时间花在环境上

这是 agentic RL 与 single-turn RL 的根本差异：

- single-turn：rollout 时间 ≈ LLM forward 时间；
- multi-turn agentic：rollout 时间 ≈ Σ tool execution time，**LLM forward 只占 5-10%**。

工程含义：

- **环境基础设施是 agentic RL 的核心壁垒**——OpenAI / Anthropic / Google 都自建大型 sandbox cluster；
- **并发 N 条 trajectory 必须充分**——LLM rollout 浪费 CPU 时间没事，环境是限制；
- **环境必须 cacheable / snapshotable**——同一任务重复 rollout 时环境状态可快速恢复。

工业经验：一个 GRPO step 的 rollout 时间通常 **5-30 分钟**（取决于 trajectory 长度），相比 single-turn RL 慢 10-100×。

### 5.3 Sandboxed environment 的设计

代码执行类 agent：

- **Docker snapshot**：每个 trajectory 开始时从 base snapshot 启动一个 container；
- **网络隔离**：agent 不能访问外部网络（除非任务需要），防安全 + 防作弊；
- **资源限制**：CPU、内存、磁盘、wall time 都有 cap；
- **State machine**：agent 操作记录在 Git-like 状态树中，便于 hindsight。

浏览器类 agent：

- **Playwright / Selenium 控制 Chromium**；
- **截图 + DOM**：作为 observation 给 agent；
- **Action**：click(x,y) / type(text) / scroll；
- **环境快照**：失败可回滚到初始页面。

### 5.4 Verifier 的设计

Verifier 是 agentic RL 的"答案钥匙"——必须严格、独立、不可篡改：

- **独立进程**：与 agent 完全隔离，agent 不能读 verifier 代码；
- **黑盒接口**：verifier 只输出 0/1，不输出"为什么错"（避免 agent 通过 reward 信号反推 verifier 逻辑）；
- **Hidden test cases**：SWE-Bench 用 hidden tests，agent 只看到部分 visible tests；
- **Multi-verifier ensemble**：高赔率任务用多个 verifier 投票（如人 + AI judge + 程序）。

---

## 六、代表性工作（2024-2025）

### 6.1 SWE-Bench 系列

**[SWE-Bench (Jimenez 2023)](https://arxiv.org/abs/2310.06770)**：从 GitHub 真实 issue + 对应 PR 构造的任务集——给模型 issue 描述 + 仓库，让它产生 patch，跑 hidden tests 校验。

**[SWE-Bench Verified (OpenAI 2024)](https://openai.com/index/introducing-swe-bench-verified/)**：人工 verify 后的子集（500 题），任务 quality 高。当前主要榜单。

**当前 SOTA（2026 中）**：

- Claude Sonnet 4.5 / GPT-5 / o3-class：**60-75% SWE-Bench Verified**；
- 开源最佳：**40-50%**（Qwen3-Coder、DeepSeek-V3.1 等）；
- 蒸馏小模型：**30-40%**（R1-Distill-Coder-32B 类）。

**关键工作**：

- **OpenHands (formerly OpenDevin)**：开源 agent 框架；
- **SWE-Gym** (2024)：开源 SWE 训练环境（数千 Docker snapshot）；
- **SWE-RL** (Meta 2024)：开源 SWE RL training pipeline；
- **Agentless (Princeton 2024)**：不用 agent loop 也能做 SWE 的反例，提示 agent 框架不是唯一路径。

### 6.2 Anthropic Computer Use (2024)

[Anthropic Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use)：Claude 3.5 Sonnet 操作 GUI（截图 → action）。

技术要点：

- **多模态输入**：截图当 observation；
- **Action space**：click(x,y), type, scroll, key combo；
- **训练**：监督 + RL on 任务完成校验；
- **极慢**：每 action 1-3 秒，trajectory 几十步耗时几分钟；
- **实用度仍受限**：操作精度、长程稳定性、错误恢复都是难题。

后续工作：OS-Atlas、ShowUI、ScreenSpot 系列把"截图 + GUI action"做成标准化任务。

### 6.3 Search Agent

让模型用 search engine 找答案：

- **Search-R1 (2025)**：RL on retrieval + answer，把"用 Bing 查 + 总结"训成 reasoning trajectory；
- **WebGPT (OpenAI 2021)**：早期版本，用 RLHF + browser tool；
- **Reflexion (Shinn 2023)**：把失败经验放回 context 再 rollout（不严格 RL，但思路相关）。

Search agent 的特点：

- 工具数少（search、open_page、quote）；
- Trajectory 短（5-20 步）；
- Reward 是答案正确性（可用 LLM-judge）。

→ 比 SWE agent 简单很多，是入门 agentic RL 的好起点。

### 6.4 Devin / Cursor / Claude Code 类（商业闭源）

商业 agent 公开信息有限，但据透露：

- 大量 supervised data + RL 微调；
- 显式规划模块（plan-then-execute）+ 工具 routing；
- 任务级别的 reward（用户接受 PR / 用户给 thumbs up）。

Cursor / Claude Code 偏交互式（用户 in-the-loop），训练数据是真实用户 session 的脱敏版本。

### 6.5 General Agentic RL Framework

开源工具栈：

- **OpenRLHF**：支持 multi-turn rollout；
- **VeRL** (ByteDance)：DAPO 母框架，含 agent 扩展；
- **TRL**：含 multi-turn DPO/GRPO trainer；
- **Ray + sandboxed environments**：工业 distributed RL 标配。

---

## 七、Agentic 评测

不能用单 prompt benchmark，要 trajectory-level：

| Benchmark | 任务 | 特点 |
|---|---|---|
| **SWE-Bench Verified** | 修真实 GitHub bug | 工程任务标杆 |
| **WebArena** | 浏览器任务 | 用真实网站 mock |
| **VisualWebArena** | 浏览器 + 视觉 | 含截图理解 |
| **OSWorld** | GUI 操作 | 真桌面环境 |
| **TauBench** | 多轮客服对话 | 含工具 + 长对话 |
| **AgentBench** | 综合多任务 | 涵盖 8 大类 |
| **GAIA** | 长程信息整合 | 包含 web search + 推理 |
| **LiveCodeBench (agent mode)** | 实时编程 | 数据新鲜，防数据污染 |
| **AppWorld** | 应用 + API | 含百个虚拟 app |

评测难点：

- **不可复现**：网站状态变化、tool 行为变化、随机性高；
- **结果二值化粗糙**：失败原因多种（agent 错 / 环境 / 评测错），单一 0/1 难定位；
- **成本高**：评测一次需要跑数千次 rollout，每次几分钟。

详见 §13 评测章节。

---

## 八、关键问答

**Q1**：为什么 agentic RL 不能直接用 PPO？
- PPO 可以用，但 long-horizon 让 value head 训练很难（critic 在 50 步深的 state 上几乎是噪声）；
- GRPO 用 group baseline 替代 critic，对长 trajectory 更稳；
- 2025 业界多用 GRPO / GSPO + dense reward。

**Q2**：Reward 太稀疏怎么办？
- **Dense process reward**：跑通的 test 数、文件覆盖率、中间检查（权重小）；
- **课程学习**：先简单任务再难任务，让 reward 信号渐进；
- **Hindsight relabeling**：失败 trajectory 重标目标也能学；
- **BC + RL 混合**：用专家 trajectory bootstrap。

**Q3**：Agentic SFT 还是 RL？
- **SFT 是起点**：用人类 trajectory / 强模型 trajectory bootstrap，给模型基本工具调用能力；
- **RL 是上限**：让模型自己探索更优策略；
- 工业实战：**SFT (BC) → RL 是标准 pipeline**。
- 跳过 SFT 直接 RL：reward 全 0，policy 几乎学不动（与 R1-Zero 不同：R1-Zero 在已有强 base 上做 single-turn，agent 任务多轮难度高得多）。

**Q4**：Trajectory 过长（>32k）怎么办？
- **Cap max trajectory length**（典型 32k-64k）；
- 用 long context 训练（详见 §8 长上下文）；
- 中段截断 + summary（影响连贯性，谨慎用）；
- 把任务拆成子任务（"先 plan，再 execute"）。

**Q5**：如何防止 agent 在沙盒里"作弊"？
- **沙盒 read-only 关键文件**：verifier 代码、hidden tests 不可读；
- **Verifier 独立**：不让 agent 改 verifier；
- **Multi-verifier ensemble**：多个独立 verifier 投票；
- **Reward shaping 防漏洞**：发现 hack 模式立刻加负 reward。

**Q6**：tool calling 准确率怎么提？
- SFT 阶段大量工具调用示范；
- function calling schema 用 JSON-mode 强制；
- RL 阶段惩罚 invalid call（reward = -0.1）；
- 在 trajectory 末尾算个"工具调用率"作为辅助 metric。

**Q7**：开源 agent 模型距离 GPT-5/Claude 多远？
- 2026 中：SWE-Bench Verified 开源最佳 ~40-50%，闭源 ~60-75%；
- 主要差距：**训练数据规模、环境基建、长程稳定性**；
- 蒸馏 + 社区基建（SWE-Gym 等）在缩小差距。

**Q8**：长 trajectory 训练显存怎么省？
- **Gradient checkpointing**（必开）；
- **Sequence parallelism / ring attention**（§8.3）；
- **Off-policy 重用 rollout**（减少 rollout 次数）；
- **Trajectory packing 配 FlashAttention varlen**；
- 实在不够：缩短 max trajectory length。

**Q9**：Agentic RL 的训练时间为什么这么长？
- Rollout 阶段 95% 时间花在 tool execution（不是 LLM forward）；
- 每个 GRPO step 需要数百条 trajectory，每条几分钟；
- 一个 RL run 可能要数周（DeepSeek、ByteDance 内部 cluster）；
- **环境基建直接决定训练速度**——这就是为什么 OpenAI/Anthropic 自建 sandbox cluster。

**Q10**：能用 RL 训练 agent 不依赖大量人类示范吗？
- 理论上可以（R1-Zero 风格 explore），但实战极难——多轮 + 稀疏 reward 让 policy 几乎无法 cold start；
- 实际上**必须** SFT bootstrap：人类示范 + 大模型蒸馏 trajectory 是起点；
- "agent RL from scratch" 仍是开放研究问题。

---

## 九、本节与其他节关系

按 [§11.0 导论](./00-章节导论.md) 定位：Agentic RL 是**算法主线的终点**，它把 §11.5 GRPO 套在多轮 trajectory + 真实环境上。

```
   §11.1 SFT
      ↓ (BC cold-start：先用专家 trajectory imitate)
   §11.3 PPO ──┐
   §11.4 DPO  ─┴── 不用（chat 对齐可选独立做）
      ↓
   §11.5 RLVR / GRPO
   (算法层骨架，本节直接沿用)
      ↓ + 多轮 trajectory + 环境/工具 + 长 horizon
   §11.6 Agentic RL (本节, 算法主线终点)
      ↓
   §13 Agentic benchmark (SWE-Bench / WebArena / GAIA)
   §12 推理 serving (生产环境 agent 调度)

上游能力依赖（不是同级）:
  §8 长上下文      ── trajectory 几万 token，long-context 训练能力是前提
  §9 MoE          ── 大 agent 模型常用 MoE 省推理算力

正交话题:
  §11.7 RLAIF/CAI ── 专家 trajectory 可用 AI judge 自动评分
  §11.2 PEFT      ── LoRA + agentic RL 显存配方
```

**记忆口诀**：Agentic RL ≈ GRPO + 三件事——**obs mask + dense process reward + sandbox 集群**。算法层 95% 抄 §11.5，新难度 95% 在工程（环境基建是真正瓶颈：rollout 时间 95% 花在 tool execution，不在 LLM forward）。

理解 §11.5 是理解本节的前提；而本节又是理解未来 "general-purpose agent training" 的基础。

---

## 十、参考资料

**Agent 框架 / 范式**：
- [Reflexion (Shinn 2023)](https://arxiv.org/abs/2303.11366)（trajectory 反思早期工作）
- [Lilian Weng — LLM Powered Autonomous Agents (2023)](https://lilianweng.github.io/posts/2023-06-23-agent/) ⭐⭐
- [Survey of LLM Agents (Wang 2024)](https://arxiv.org/abs/2308.11432)
- [Tree-of-Thoughts (Yao 2023)](https://arxiv.org/abs/2305.10601)
- [ReAct (Yao 2022)](https://arxiv.org/abs/2210.03629)

**SWE / 编程 agent**：
- [SWE-Bench (Jimenez 2023)](https://arxiv.org/abs/2310.06770) ⭐⭐
- [SWE-Bench Verified (OpenAI 2024)](https://openai.com/index/introducing-swe-bench-verified/) ⭐
- [SWE-Agent (Yang 2024)](https://arxiv.org/abs/2405.15793)
- [OpenHands (OpenDevin)](https://github.com/All-Hands-AI/OpenHands) ⭐
- [SWE-Gym (2024)](https://github.com/SWE-Gym/SWE-Gym)
- [SWE-RL (Meta 2024)](https://arxiv.org/abs/2502.18449)
- [Agentless (Princeton 2024)](https://arxiv.org/abs/2407.01489)

**Web / GUI agent**：
- [WebArena (Zhou 2023)](https://arxiv.org/abs/2307.13854)
- [VisualWebArena (Koh 2024)](https://arxiv.org/abs/2401.13649)
- [OSWorld (Xie 2024)](https://arxiv.org/abs/2404.07972)
- [Anthropic Computer Use blog](https://www.anthropic.com/news/3-5-models-and-computer-use) ⭐
- [OS-Atlas (2024)](https://arxiv.org/abs/2410.23218)
- [ShowUI (2024)](https://arxiv.org/abs/2411.17465)

**Search / RAG agent**：
- [Search-R1 (2025)](https://arxiv.org/abs/2503.09516)
- [WebGPT (OpenAI 2021)](https://arxiv.org/abs/2112.09332)
- [Self-RAG (Asai 2023)](https://arxiv.org/abs/2310.11511)

**评测**：
- [Tau-Bench (2024)](https://arxiv.org/abs/2406.12045)
- [AgentBench (Liu 2023)](https://arxiv.org/abs/2308.03688)
- [GAIA (Mialon 2023)](https://arxiv.org/abs/2311.12983)
- [AppWorld (Trivedi 2024)](https://arxiv.org/abs/2407.18901)

**RL 算法基础**（多轮扩展）：
- [Hindsight Experience Replay (Andrychowicz 2017)](https://arxiv.org/abs/1707.05952)
- [Off-policy RL with LLM (2024+ 综述)](https://arxiv.org/abs/2405.14655)
- [VeRL — agentic RL trainer](https://github.com/volcengine/verl) ⭐
- [OpenRLHF — agent extension](https://github.com/OpenRLHF/OpenRLHF) ⭐

**深度博客**：
- [Lilian Weng — LLM Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) ⭐⭐
- [Sutton — The Bitter Lesson 的 agent 演绎](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [Anthropic Engineering blog — Computer Use 实战](https://www.anthropic.com/engineering)
