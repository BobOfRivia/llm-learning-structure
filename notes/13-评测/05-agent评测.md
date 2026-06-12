# 13.5 Agent 评测（GAIA / WebArena / OSWorld / τ-bench）

[← 返回框架](../../README.md) · [📎 materials.md → §13.5](../../materials.md)

---

## 〇、本节回答什么

> 2024 下半年起，"agent" 从概念变成产品（Anthropic Computer Use、OpenAI Operator、Manus、DeepSeek-V3 + tool）。Agent 评测的核心 narrative：
>
> - **能力维度**：tool use、browser、GUI、long-horizon planning
> - **环境**：从合成 sandbox → 真实 OS / web → 真实 SaaS API
> - **2025 现状**：SWE-bench Verified（代码 agent）+ τ-bench（tool use agent）+ OSWorld（GUI agent）+ GAIA（通用 agent）是四大常见报数

---

## 一、Agent 评测的特殊性

### 1.1 与"模型 benchmark"不同

```
模型 benchmark：单轮 prompt → 单轮 response → 评分
Agent benchmark：多轮工具调用 → 环境交互 → 任务完成检测
```

引入：
- **环境状态**（OS state / web state / database state）
- **工具集**（rule-based vs LLM-generated）
- **回合数 / token 预算**（影响算力上限）
- **scaffolding**（agent harness 框架）

### 1.2 评分困难

- "完成"的定义：state 检查 vs LLM-as-judge vs 测试通过
- **环境复现**：网页改版后无法复现
- **agent loop 上界**：n_steps 设置影响分数

---

## 二、GAIA（Mialon et al., Meta/HF, 2023）

### 2.1 设计

- **466 道**真实世界通用 agent 题
- 3 个难度档（Level 1 / 2 / 3）
- 答案是**单一字符串 / 数字 / 文件**（exact match）

```
Level 1（5 步内）：
  "What is the latest version of Photoshop available on Adobe website?"
Level 2（5-15 步）：
  "Find the name of the actor playing X in movie Y, then find their
   directorial debut, then find the year that movie was released."
Level 3（15+ 步，含文件操作）：
  含 Excel / PDF / 图片 / 视频读写，跨多个工具
```

### 2.2 工具集

GAIA 不限制工具，常见 agent 套件：
- web browser
- file reader (PDF / Word / Excel)
- code interpreter
- image viewer
- search engine

### 2.3 现状（2025 leaderboard）

```
Human:                      92%
GPT-4 + tools (2023):       15%
Claude 3.5 + Claude Code:   ≈45%
o1 + tools:                 ≈55%
SOTA agent (2025):          ≈75% (Manus / OpenAI Operator)
```

### 2.4 为什么 GAIA 是"通用 agent 第一基准"

- **任务真实**（不是合成）
- **不限工具**（测的是整套 system 能力）
- **答案唯一**（避免主观打分）
- **HuggingFace 维护排行榜**

---

## 三、WebArena（Zhou et al., ICLR 2024）

### 3.1 设计 — 自包含 web sandbox

- 自建 4 个**完整 web 应用**（基于开源软件）：
  - **OneStopShop**（电商，基于 magento）
  - **GitLab**
  - **Reddit**（基于 postmill）
  - **CMS**（content management）
- 加 mocked 工具：calculator / map / wiki
- **812 道任务**，跨上面 5 个站点

```
example task:
  "Post a question on the cooking subreddit about my pumpkin pie recipe."
```

### 3.2 评测

- 用**程序化检查**（看 DB 状态 / URL / 截图）确认任务完成
- 报 **success rate**

### 3.3 现状

```
GPT-4 (2024 初):            14%
Claude 3 Opus:              20%
GPT-4o + agent loop:        ≈30%
Claude 3.5 + WebVoyager:    ≈40%
人类:                       78%
```

### 3.4 VisualWebArena

- WebArena 的视觉版本
- agent 输入加上**截图**
- 评测多模态 + web navigation

### 3.5 OnlineMind2Web / Mind2Web

- 实时 web 任务（不在 sandbox）
- 更难复现

---

## 四、OSWorld（Xie et al., NeurIPS 2024） — 桌面 GUI agent

### 4.1 设计

- 真实 **Ubuntu / Windows / macOS VM**
- **369 道任务**，跨 GUI 应用：
  - 文件管理 / Office / Browser / VSCode / GIMP / Thunderbird ...
- 输入：截图 + 任务描述
- 输出：鼠标点击 / 键盘输入 / shell

```
task:
  "Open the LibreOffice file, change all 'foo' to 'bar', then save as PDF
   to ~/Desktop/result.pdf."
```

### 4.2 评测

- 程序化检查：文件是否存在 / 内容是否正确 / VM 状态是否符合
- 用 docker / VM 隔离

### 4.3 现状

```
Claude 3.5 Sonnet (Computer Use, 2024-10):   14.9%
GPT-4o + custom scaffolding:                  11.2%
人类:                                          72%
```

→ OSWorld 是 2024 最难的"实战 GUI agent" benchmark，分数普遍 <20%，**仍有巨大提升空间**。

### 4.4 Anthropic Computer Use

- 2024-10 Anthropic 发布 Claude 3.5 Sonnet (Computer Use beta)
- 核心是把"屏幕截图 + 键鼠输出"直接接入 API
- 在 OSWorld 拿到 14.9% SOTA

---

## 五、τ-bench（Yao et al., Sierra AI, 2024） — tool-use agent

### 5.1 设计

- **真实业务场景**（航空订票 / 零售客服）
- 模拟用户和 agent 多轮对话
- agent 需要调用 **真实 API**（getOrderDetails / cancelOrder ...）
- 用户由另一个 LLM 扮演（增加多样性）

### 5.2 评测

- 检查任务结束时**业务数据库的状态**是否符合预期
- 报 **pass@1 / pass@8**
- 同时考核：**正确执行 + 拒绝错误请求**

### 5.3 现状

```
τ-bench airline:
  GPT-4o:               37
  Claude 3.5 Sonnet:    46
  Claude 3.7 Sonnet:    58
  o1:                   ≈42（推理但工具调用规范差）
```

### 5.4 τ-bench 的意义

测的不是"通用智能"而是**业务流程级 tool use**：
- agent 能不能用对 API
- 能不能拒绝违反 policy 的请求
- 能不能多轮对话保持 state

→ 与"产品场景"最贴近的 benchmark 之一。

---

## 六、其他重要 agent benchmark

### 6.1 AgentBench (THUDM, 2023)

- 第一代综合 agent benchmark
- 涵盖 OS / DB / 知识图谱 / 卡牌游戏 / 网页 / shop / web 工具 / 横向 7 类
- 现已被更专门化的 benchmark 取代

### 6.2 ToolBench / API-Bank

- 单纯测 tool use（函数调用）
- 与 τ-bench 相比缺少多轮 / 业务流程

### 6.3 BrowseComp (OpenAI, 2024)

- 测搜索 / 浏览 / 综合信息能力
- 1266 道难题，需要多次搜索 + 综合
- 风格类似 GAIA 但更专注 browsing

### 6.4 Cybench

- 测 cyber-security agent（CTF 题）

### 6.5 BFCL (Berkeley Function-Calling Leaderboard)

- function-calling 基础能力 benchmark
- 700+ 测例，覆盖 simple / multiple / parallel / multi-step function calls

### 6.6 Web Arena 系列变体

- **WebVoyager**（Tsinghua, 2024）：真实 web，含视觉
- **WebShop**：电商 agent

### 6.7 ScreenAgent / ScreenAI

- 屏幕理解 + agent

### 6.8 SWE-bench Verified（§13.3 已述）

- 代码 repo agent → 最重要的 agent benchmark 之一

---

## 七、Agent 评测的几个工程坑

### 7.1 环境复现

- WebArena 自带 docker → 还能复现
- OSWorld VM 几十 GB → 起一个 VM 几分钟
- Mind2Web 真实 web → **网站改版直接打死**所有历史报数

### 7.2 算力代价

agent loop 每步生成几百 token，n_steps=20-50，每题 token 量 5K-50K。
- GAIA 466 题 × 20K token ≈ 10M token
- OSWorld 369 题 × 50K + 截图 → **每题 $0.5-2**
- SWE-bench Verified 500 题 × agent ≈ **$50-500 per model run**

### 7.3 scaffolding 影响巨大

同一模型用不同 harness 差 10-30 点。报数必须注明：
- 工具集（什么 functions 给模型）
- system prompt
- agent loop 框架（ReAct / CodeAct / Function Calling）
- 截图 / DOM 喂法（OSWorld）

### 7.4 Reward hacking / Test trick

agent 可能学到**绕过 hidden test** 的捷径。SWE-bench 早期就有：模型识别 hidden test 文件名 → 直接 mock 通过。需要 Verified 子集人工核。

### 7.5 LLM-as-judge bias

部分 agent benchmark 用 LLM 评分 → 评 GPT-4 用 GPT-4 评 → **同源偏见**。OSWorld / GAIA / τ-bench 都改用 **程序化检查** 避开这个。

---

## 八、Agent 评测分类总览

| Benchmark | 环境 | 工具 | 评分 | 难度档 | 适用 agent 类型 |
| --- | --- | --- | --- | --- | --- |
| **GAIA** | 真实 web + 文件 | 不限 | exact match | 1/2/3 | 通用 agent |
| **WebArena** | 自建 web sandbox | 浏览 | DB / URL 检查 | 单档 | web agent |
| **VisualWebArena** | 同上 + 视觉 | 浏览 | 同上 | 单档 | web + vision |
| **OSWorld** | 真实 VM | 鼠键 + shell | VM 状态检查 | 单档 | GUI agent |
| **τ-bench** | 业务 API | 业务 functions | DB 状态 | 单档 | tool-use agent |
| **SWE-bench Verified** | 真实 repo | shell + 编辑器 | 测试通过 | 单档 | code agent |
| **BFCL** | 模拟 | functions | exact | 多档 | function calling |
| **Cybench** | CTF 沙箱 | shell + 工具 | flag 抓到 | 单档 | security agent |
| **BrowseComp** | 真实 web | 浏览 | exact | 单档 | search agent |

---

## 九、与其他章节联动

- **§11.5 RLVR**：agent benchmark 提供天然 reward（任务完成 / 工具调用正确）；但 reward 周期长（5–30 分钟一题）→ 难直接做 RL 训练
- **§12.6 TTS**：agent 天然是 test-time scaling 场景（多步推理 + tool use）
- **§13.3**：SWE-bench Verified 是 code agent 评测，同时归 §13.3 和 §13.5

---

## 十、关键问答

**Q1：GAIA / WebArena / OSWorld 怎么选？**
A：按 agent 类型：
- 通用 agent → **GAIA**
- web 自动化 → **WebArena / VisualWebArena**
- 桌面 GUI / Computer Use → **OSWorld**
- 业务 API tool use → **τ-bench**
- 代码 → **SWE-bench Verified**
旗舰模型现在常报 3-4 个组合。

**Q2：为什么 OSWorld 分数都这么低？**
A：(1) 真实 OS 状态空间巨大 (2) GUI 元素定位难（截图 + 坐标） (3) 长 horizon 步骤（30+ 步） (4) 容错少（一个误点全盘崩）。**所有模型在 <20%**，是 2024-2025 最有提升空间的 agent benchmark。

**Q3：Anthropic Computer Use 是什么？**
A：Claude 3.5 Sonnet (computer-use beta, 2024-10) 把"看屏幕截图 + 输出鼠键命令"作为原生 API 能力。在 OSWorld 拿 14.9% SOTA。**信号**：agent 能力开始从 scaffolding 进入模型内部，模型直接懂 GUI。

**Q4：τ-bench 跟 BFCL 区别？**
A：BFCL = function calling **正确性**（API 调用语法 / 参数），单轮居多。τ-bench = **业务流程级**：多轮对话 + 多 API 调用 + 拒绝违反 policy + 用户由 LLM 扮演。τ-bench 更贴近真实产品。

**Q5：agent 评测的 scaffolding 怎么算？**
A：常见 scaffolding 框架：
- **ReAct**（Reasoning + Action 交替）
- **CodeAct**（让 agent 输出 Python 代码而非 JSON tool call）
- **Reflexion**（失败后让 agent 反思）
- **Tree-of-Agents**（多 agent 协作）
不同 scaffolding 同一模型差 10-30 点 → 报数必须注明。

**Q6：什么是 Aider Polyglot / OpenHands 这类 agent 框架？**
A：
- **Aider**：代码编辑专用 agent，本地 CLI + git
- **OpenHands**（前 OpenDevin）：通用 agent 框架，支持 bash/python/browser
- **SWE-agent**：SWE-bench 专用 agent
- **AutoGPT / BabyAGI**（2023 老古董）：现已退场
**OpenAI Codex / Anthropic Claude Code 等是闭源产品**，开源对标是 OpenHands / Aider。

**Q7：agent 评测能不能进 RLHF 训练？**
A：理论上可以，工程上难：
- 每条 rollout 5-30 分钟（vs 数学题秒级） → RL throughput 极低
- 用 **task-completion success** 做稀疏 reward → 信号弱
- 当前主流：用 LLM-as-judge 做 process reward（PRM 风格）+ outcome reward 组合
- DeepSeek-R1 等还没大规模 RL on agent benchmark；OpenAI 推测在 o3 / o4 引入

---

## 参考资料

### 论文 / Benchmark
- ⭐⭐ [GAIA: A Benchmark for General AI Assistants (Mialon et al., 2023)](https://arxiv.org/abs/2311.12983)
- ⭐⭐ [WebArena (Zhou et al., ICLR 2024)](https://arxiv.org/abs/2307.13854)
- ⭐ [VisualWebArena (Koh et al., 2024)](https://arxiv.org/abs/2401.13649)
- ⭐⭐ [OSWorld (Xie et al., NeurIPS 2024)](https://arxiv.org/abs/2404.07972)
- ⭐⭐ [τ-bench (Yao et al., Sierra AI, 2024)](https://arxiv.org/abs/2406.12045)
- ⭐ [AgentBench (Liu et al., ICLR 2024)](https://arxiv.org/abs/2308.03688)
- ⭐ [BFCL — Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- ⭐ [BrowseComp (OpenAI, 2024)](https://openai.com/index/browsecomp/)
- ⭐ [WebVoyager (He et al., 2024)](https://arxiv.org/abs/2401.13919)
- ⭐ [Anthropic — Developing a computer use model (2024-10)](https://www.anthropic.com/news/3-5-models-and-computer-use)
- ⭐ [Mind2Web (Deng et al., NeurIPS 2023)](https://arxiv.org/abs/2306.06070)
- ⭐ [Cybench (Zhang et al., 2024)](https://arxiv.org/abs/2408.08926)

### 工具与排行榜
- ⭐⭐ [GAIA Leaderboard (HuggingFace)](https://huggingface.co/spaces/gaia-benchmark/leaderboard)
- ⭐⭐ [SWE-bench Leaderboard](https://www.swebench.com/)
- ⭐⭐ [OSWorld Leaderboard](https://os-world.github.io/)
- ⭐ [WebArena Leaderboard](https://webarena.dev/)
- ⭐ [τ-bench leaderboard](https://github.com/sierra-research/tau-bench)
- ⭐⭐ [OpenHands (前 OpenDevin)](https://github.com/All-Hands-AI/OpenHands)
- ⭐ [SWE-agent](https://github.com/princeton-nlp/SWE-agent)
- ⭐ [Aider](https://aider.chat/)

### 解读
- ⭐ [Anthropic — Building Effective Agents (2024-12)](https://www.anthropic.com/research/building-effective-agents)
- ⭐ [LangChain — State of AI Agents 2024 report](https://blog.langchain.dev/)
- ⭐⭐ [Sierra AI — τ-bench blog post](https://sierra.ai/blog/benchmarking-ai-agents)
- ⭐ [Why agentic AI evaluation is hard — Patronus 2024](https://www.patronus.ai/)

→ 下一节 [§13.6 人评 / Arena Elo](./06-人评-arena.md)：Chatbot Arena / MT-Bench / Arena-Hard / Style Control。
