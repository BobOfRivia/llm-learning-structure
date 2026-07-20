# 8.4 Lost in the Middle 产生的原理

[← 返回框架](../../README.md) · [📎 materials.md → §8.4](../../materials.md)

> 现象由 [Liu et al. 2023 "Lost in the Middle"](https://arxiv.org/abs/2307.03172) 系统刻画；本节讲**为什么会这样**（机制），而不只是"它存在"。
> 现象侧的评测（NIAH / RULER 的热图）在 [§13.4 长上下文评测](../../notes/13-评测/04-长上下文评测.md)；缓解手段的位置见本节 §六。

---

## 〇、这一节回答什么

> 把关键信息放在长上下文的**开头或结尾**，模型答得准；放在**中间**，召回率显著下降，画出准确率就是一条 **U 型曲线**（首尾高、中间塌）。Liu 2023 在 20 文档 multi-doc QA 上测到：答案文档从第 1 位移到第 10 位，准确率掉 **30%+**。
>
> 问题：这**不是数据 bug，也不是没训好**，而是 **causal + softmax + 位置编码** 三者叠加出的**结构性位置偏置**。本节拆开这三个来源。

---

## 一、现象：U 型召回曲线

```
准确率
 高 │██                              ██
    │  ██                          ██
    │    ███                    ███
    │       ████            ████
 低 │           ██████████████
    └──────────────────────────────────→ 相关信息在 context 中的位置
      开头(primacy)      中间(谷底)      结尾(recency)
```

两个"抬升"分别对应两种心理学也熟悉的效应：

- **Primacy（首因）**：开头 token 被后面所有位置反复注意到 → 权重被"顶上来"。
- **Recency（近因）**：结尾 token 离查询/生成位置最近 → 天然高注意力。
- **中间谷底**：既不占首因、也不占近因，被两头挤到"低注意力区"。

> 注意：U 型是**关于位置**的偏置——与该位置内容是否 relevant **无关**。哪怕答案就在中间，注意力也系统性地少给它。

---

## 二、根因一：因果掩码 → 天然的"位置累积偏置"

Decoder-only 用 **causal mask**：位置 $t$ 只能看 $\le t$ 的 token。带来一个不对称结构：

- **靠前的 token 被"看"的次数更多**：位置 1 会被 $2,3,\dots,T$ 全部作为 key 访问；位置 $T-1$ 只被 $T$ 访问一次。于是**靠前的 key 有更多机会累积注意力质量**。
- 这一步**甚至不需要位置编码，也不需要训练**就存在：近期理论工作（"Lost in the Middle at Birth"）指出，U 型偏置在**随机初始化时就已存在**——它来自 causal attention + softmax 的结构本身，训练只能**减轻**而不能消除。

### 2.1 Attention Sink（注意力沉降）加剧首端

自回归模型倾向把**大量注意力质量倾倒到最前面的几个 token**（哪怕是 BOS 这种无语义 token）——这是 softmax "必须把权重加和到 1、总得放某处"的副产品（StreamingLLM 的观察，见 [§8.3 / §6.1](./03-kv节省.md)）。

- 后果：开头被 sink 抢走大量注意力 → 首端进一步抬高，中间相对更被稀释。
- 这也是 StreamingLLM "保留前几个 sink token + 滑窗"能维持长序列质量的原因。

---

## 三、根因二：softmax 注意力的"稀释"（长度本身就是敌人）

softmax 把注意力权重归一化到和为 1。序列越长，参与竞争的 key 越多，**单个中间 token 能分到的注意力平均值 $\approx 1/T$ 越小**。

- 一个"该被关注"的中间 token，其信号要在 $T$ 个竞争者里脱颖而出；上下文越长，越容易被大量无关 token 的背景注意力**淹没（dilution）**。
- 这是"**context 越长、单点召回越难**"的数学根源之一，也解释了为什么"声明 1M、有效只有 ~128K"（§13.4）。

> 记忆点：**U 型是"位置"的病，稀释是"长度"的病**，两者叠加就是 lost-in-the-middle 在超长上下文里格外严重的原因。

---

## 四、根因三：RoPE 的长程衰减（distance decay）

现代 LLM 多用 **RoPE**（[§2.2](../02-transformer-架构/02-位置编码.md)）。RoPE 把 Q/K 旋转、使 attention score 只依赖相对距离 $m-n$，且**score 期望随相对距离增大而衰减**（long-term decay，Su 2021 的设计目标之一）。

- 后果：query（通常在结尾附近）对**距离远的中间 token** 天然给更低的相似度 → 中间更吃亏。
- 结尾 token 因为**离 query 近**（相对距离小），衰减弱 → recency 抬升。
- 与位置外推的关系：当推理长度超过训练长度，中频维度角度落到 OOD（[§8.2](./02-位置外推.md)），中段位置的 score 分布进一步漂移，**放大**中间塌陷。

> 三根因关系：**causal mask 定基调（首端偏置 + 已在初始化存在）→ softmax 稀释随长度恶化 → RoPE 衰减把"远 = 中间"进一步压低**。它们同向叠加，不是互相独立的三选一。

---

## 五、为什么说"不是训练不足，是结构性的"

| 常见误解 | 实际 |
|---|---|
| "多喂点长样本就好了" | 训练能**减轻**（把有效长度往后推），但 U 型在初始化就存在、softmax 稀释是数学必然，**无法根除** |
| "换个更大的模型就好了" | 更大模型缓解但仍有 U 型；Liu 2023 在 GPT-3.5 / Claude 上都测到 |
| "把 context 开到 1M 就能用满" | claimed ≠ effective，正是 lost-in-the-middle + 稀释导致 RULER effective 常打 5 折（§13.4） |
| "只是位置编码的锅" | 位置编码是**其中一根**；causal + softmax 那两根即使换掉 RoPE 也在 |

---

## 六、缓解手段（分层）

按"改哪一层"归类，跨到别的章节：

**① Prompt / 检索层（最便宜，工程首选）**
- **把关键信息放首尾**：RAG 里对召回文档**重排（re-rank）**，最相关的放开头和结尾，次要的塞中间。
- **压缩中段**：对长文档先摘要/裁剪，减少稀释源。
- Anthropic 等给 prompt 加**结构**（XML 标签、显式"引用相关段落再作答"）也能拉回中段利用率。

**② 推理时干预 attention（研究方向）**
- **Found-in-the-Middle / 位置注意力校准**（[Hsieh et al. 2024](https://arxiv.org/abs/2406.16008)）：直接对 attention 分布做**去位置偏置校准**，让权重反映 relevance 而非 position。
- Attention 重加权 / 谱方法 highlight 相关段落。

**③ 位置编码 / 训练层（治本但贵）**
- 更好的**位置外推**（YaRN / LongRoPE，§8.2）减轻 RoPE 侧的中段漂移。
- 长上下文 curriculum + 中段监督（把针放中间训练，§8.1）把 U 型压平一些。
- 架构侧：混合/线性注意力（§7）没有 softmax 全局归一化，稀释特征不同（但有各自的召回短板）。

**④ 评测层（验证，不是缓解）**
- 用 NIAH 的**深度维度**、RULER 的多针任务专门测中段（§13.4）。

---

## 七、关键问答

**Q1：一句话讲 lost-in-the-middle 的成因。**
A：causal mask 让靠前 token 被累积注意更多（首因、attention sink），query 离结尾近产生近因，中间位置两头不靠；再叠加 softmax 随长度稀释、RoPE 对远距离 score 衰减 → 中段系统性低注意力，形成 U 型召回。

**Q2：这是训练数据问题吗？多训长文本能解决吗？**
A：主要**不是**数据问题。U 型偏置在**随机初始化时就存在**（causal + softmax 的结构性质），softmax 稀释是数学必然。长文本训练只能**减轻/后移**，无法根除。

**Q3：为什么 context 越长这个问题越严重？**
A：两个叠加——(1) softmax 把注意力归一化到 1，token 越多单点平均注意力越低（稀释）；(2) 中间位置离 query 更远，RoPE 长程衰减更狠。所以 claimed length 越大、effective length 打折越多（§13.4）。

**Q4：工程上最快的缓解？**
A：**重排**。RAG 召回后把最相关文档放**开头和结尾**、次要的放中间；配合中段摘要压缩、给 prompt 加结构。比改模型便宜得多。

**Q5：换掉 RoPE（比如用 ALiBi 或线性注意力）能消除吗？**
A：不能完全消除。RoPE 只是三根因之一；causal mask 的首端累积 + softmax 稀释与位置编码无关，换编码只动其中一部分。

**Q6：和"attention sink"是一回事吗？**
A：不是同一个，但相关。attention sink 是"注意力质量被倒给最前几个 token"的现象，它**加剧了 U 型的首端抬升**这一侧，是成因之一，而非全部。

---

## 八、与各章的联动

```
§2.2  RoPE long-term decay ─┐
§6.1  attention sink / StreamingLLM ─┼─→ §8.4 三根因
§8.1  长上下文训练（中段监督缓解）  │
§8.2  位置外推（减轻中段 OOD 漂移） ─┘
§8.3  KV 驱逐（H2O/SnapKV 也隐含"保首尾"直觉）
§13.4 评测：NIAH 深度维度 / RULER 多针任务验证中段
```

---

## 参考资料

- ⭐⭐ [Lost in the Middle: How Language Models Use Long Contexts (Liu et al. 2023)](https://arxiv.org/abs/2307.03172) —— 现象的原始系统刻画（U 型、30%+ 掉点）
- ⭐⭐ [Found in the Middle: Calibrating Positional Attention Bias (Hsieh et al. 2024)](https://arxiv.org/abs/2406.16008) —— 直接校准位置注意力偏置的缓解方案
- ⭐ [Attention Sink / StreamingLLM (Xiao et al. 2023)](https://arxiv.org/abs/2309.17453) —— 首端注意力沉降机制
- ⭐ [RoFormer / RoPE (Su et al. 2021)](https://arxiv.org/abs/2104.09864) —— RoPE 及其 long-term decay 设计
- ⭐ [The "Lost in the Middle" Problem — U-Shaped Attention Explained (Morph)](https://www.morphllm.com/lost-in-the-middle-llm) —— 机制科普图解
- ⭐ [Lost in the middle: how architecture and training data shape position bias (TechXplore, 2025)](https://techxplore.com/news/2025-06-lost-middle-llm-architecture-ai.html) —— "初始化即存在"的理论解读
