# 12.2 Serving 系统（vLLM / PagedAttention / Continuous Batching / SGLang / RadixAttention）

[← 返回框架](../../README.md) · [📎 materials.md → §12.2](../../materials.md)

---

## 〇、本节回答什么

> 同样的 GPU 同样的模型，为什么 vLLM/SGLang 比朴素 HuggingFace generate 快 10–24×？PagedAttention 解决了什么，Continuous Batching 又解决了什么？RadixAttention vs PagedAttention 是替代关系还是叠加？CUDA Graph、chunked prefill、prefix cache 各自在哪里发挥作用？

LLM serving 系统是把硬件算力翻译成 throughput 的中间层。它同时优化三件矛盾的事：
- **throughput**（每秒生成 token 总数）↑
- **latency**（单请求 TTFT / TPOT）↓
- **fairness**（不同请求之间不能互相阻塞）

vLLM (UC Berkeley, 2023)、SGLang (LMSYS, 2024)、TensorRT-LLM (NVIDIA) 是三大主流栈。

---

## 一、为什么 generate() 慢：朴素实现的三大病

### 1.1 静态 batch（static batching）

```
请求 1: ████████████████████████   (250 token)
请求 2: ███                        (30 token)
请求 3: █████████                  (90 token)

batch step→  →  →  →  →  →
        请求 2/3 早就结束了，但 GPU 还在等请求 1，全程占着 batch slot
```

短请求被拖到最长请求的尾巴 → GPU 利用率极低。

### 1.2 显存预分配 KV cache（pre-allocate to max_seq_len）

每个请求按 `max_new_tokens` 预留连续 KV 显存——一个 4K-out 的请求即使只生成 100 token，也占着 4K 的位置。**显存碎片化**（fragmentation）+ **内部碎片**（reserved unused）共占 60–80%。

### 1.3 prefill / decode 强耦合在一个 forward pass

prefill 是 compute-bound，decode 是 memory-bound，它们的最优 batch 配置完全不同。同步处理意味着 decode 阶段拖着一个长 prompt 在做 prefill，长尾极差。

---

## 二、Continuous Batching（in-flight batching）

> 又称 **iteration-level scheduling**。每一步（每生成一个 token）都重组 batch，结束的请求立刻退出、新到的请求立刻插入。

### 2.1 直观对比

```
Static batching:
  iter 1:  [req1, req2, req3]
  iter 2:  [req1, req2, req3]
  iter 3:  [req1, ----, req3]    ← req2 结束，槽位空着
  iter 4:  [req1, ----, ----]
  ...
  iter N:  [req1, ----, ----]    ← 只剩 req1，利用率惨

Continuous batching:
  iter 1:  [req1, req2, req3]
  iter 2:  [req1, req2, req3]
  iter 3:  [req1, req4, req3]    ← req2 结束，req4 立即填补
  iter 4:  [req1, req4, req5]
```

### 2.2 工程要点

- 每个 iteration 重新拼装 input_ids / position_ids / attention_mask
- 不同请求的当前位置不同 → attention 用 **paged / variable-length** kernel
- 调度器维护**等待队列**与**运行队列**，决定每步加入哪些新请求
- vLLM / TGI / TensorRT-LLM 都把这个能力做成默认

> **Continuous batching 单独就能比 static batching 提升 3–10× throughput**（Orca 论文 + vLLM 实测）。

---

## 三、PagedAttention（vLLM 的核心创新）

### 3.1 类比操作系统的虚拟内存

KV cache 像进程地址空间——传统方式给每个请求分配一整段连续物理内存（max_seq_len 预留），浪费且无法换页。

PagedAttention 把 KV 切成固定大小的 **block**（默认 16 token / block），用**页表**把逻辑位置映射到物理 block：

```
逻辑序列:  t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 ...
            └─block 0─┘└─block 1─┘└─...
                ↓           ↓
页表:     block 0 → 物理 page #57
          block 1 → 物理 page #02
          block 2 → 物理 page #91
```

### 3.2 三个直接收益

| 问题 | PagedAttention 解法 |
|------|---------------------|
| **外部碎片**（不同长度请求间） | 物理 page 共享池，按需分配 |
| **内部碎片**（max_len 预留） | 只在写入时分配新 block |
| **多 sequence 共享 prefix**（beam search / parallel sampling / prefix caching） | 共享同一组物理 block + 引用计数 |

实测显存利用率从 ~20% → ~95%，**有效 batch size 提高 2–4×**。

### 3.3 attention kernel 改造

标准 attention 假设 K/V 是连续 tensor。PagedAttention 需要 kernel 按页表 gather K/V：

```cuda
for each query token t:
    for each block b in page_table[seq_id]:
        K_block = global_kv_pool[b]   // gather
        compute QK^T, softmax, attn·V
```

vLLM 与 FlashAttention 团队联合开发了 **paged FlashAttention kernel**，把 paging + Flash 的 tiling 融合，性能接近原生 FlashAttention。

---

## 四、Prefix Cache / Prompt Cache

### 4.1 动机

LLM 推理大量请求**共享前缀**：
- 同一个 system prompt 跨用户共用
- few-shot example 反复出现
- 多轮对话每轮的历史前缀重复

如果共享前缀的 KV 算一次就好，**省下的就是 prefill FLOPs**。

### 4.2 PagedAttention 上叠加 prefix cache

PagedAttention 天然支持引用计数共享：把已经算过的 prefix block 留在 KV pool，新请求只要前缀匹配，直接 page table 指过去就行。

vLLM 的 **automatic prefix caching (APC)**：
- 用前缀的 hash 索引 block
- LRU 淘汰
- 命中率高时 TTFT 降 70%+

### 4.3 RadixAttention（SGLang 的强化版）

SGLang 把 prefix cache 升级成 **Radix Tree（基数树/前缀树）**：

```
Radix Tree:
  root
    └── "You are a helpful..." (system prompt)
         ├── "Translate this..." (用户 A 分支)
         │     ├── "Q1: ..."
         │     └── "Q2: ..."
         └── "Summarize this..." (用户 B 分支)
```

- 不同请求的最长共享前缀自动复用
- 多轮对话天然落到一条 path
- LRU 在树上做（叶子 → 根）

> **RadixAttention vs PagedAttention 关系**：RadixAttention 是 **prefix sharing 算法**，PagedAttention 是 **KV 内存管理**。SGLang 内部仍使用类似 PagedAttention 的物理 block 池；Radix 决定如何**编排引用**。两者叠加。

### 4.4 命中率敏感的场景

- chatbot 多轮（最高，命中率常 > 80%）
- few-shot prompt 评测（接近 100%）
- code agent（system + tool description 重复）
- RAG（retrieved chunks 不同，前缀部分仍共享）

---

## 五、Chunked Prefill（分块 prefill）

### 5.1 prefill / decode 的资源冲突

```
batch 内同时有：
  请求 A：还在 prefill 4K prompt   ← 大量 FLOPs
  请求 B：在 decode 第 100 个 token ← 微量 FLOPs

朴素调度：要么先做完 A 的 prefill 再让 B decode（B 严重 TBT 抖动）
          要么 prefill 一个 token-budget 后 yield 给 decode
```

### 5.2 chunked prefill 思路

把一个长 prompt 的 prefill 切成大小为 $C$（如 512 / 1024 token）的 chunk，每个 chunk 与正在 decode 的请求**拼成同一个 forward pass**，按 token-budget 控制总计算量。

效果：
- decode 延迟稳定（每步 budget 固定）
- prefill 长度对短请求 TTFT 影响小
- batch utilization 高

vLLM、SGLang、TRT-LLM 都已默认开启 chunked prefill。SARATHI / DistServe 等论文给出了完整理论。

### 5.3 chunked prefill 与 PD 分离（§12.4）的关系

**chunked prefill** = 在同一节点上 prefill + decode 交错调度
**PD disaggregation** = 在不同节点上分别跑 prefill 和 decode

短期：单机 chunked prefill 足够；
长期 / 大集群：PD 分离更优。两者目的相同（解 prefill-decode 冲突），手段不同。

---

## 六、CUDA Graphs

### 6.1 为什么 decode 阶段 launch overhead 显著

decode 每步只算 1 token，**单次 forward 里 CUDA kernel 数量数百个**，每个 kernel launch ~10μs。小 batch 时 launch overhead 占 30%+。

### 6.2 CUDA Graph 解法

把一次完整 forward 录成静态 graph，重复 replay：

```
warmup: 录制 forward 中所有 kernel launch 的拓扑
runtime: graph.replay() ← 一次 host call 启动整条图
```

- launch overhead 从 N × 10μs → 1 × 10μs
- decode batch 越小，相对收益越大

**问题**：CUDA Graph 要求 shape 静态。所以做法是按 batch_size 离散化（如 1,2,4,8,16,32,64,128），每档录一个 graph，runtime padding 到最近档位。

vLLM / TensorRT-LLM 都默认开 CUDA Graph for decode。

---

## 七、调度器（Scheduler）

### 7.1 vLLM 调度的两条队列

```
WAITING queue:
  └── 新到的请求，待 prefill

RUNNING queue:
  └── 在 KV pool 中已有 block 的请求，正在 decode 或被抢占等待

每个 iteration:
  1. 检查 RUNNING 队列里能否 decode（KV 显存够 1 个新 block）
  2. 若不够 → preempt（swap to CPU or recompute）
  3. 检查 WAITING 队列是否可加入（chunked prefill budget）
  4. 拼装本次 forward
```

### 7.2 抢占策略

KV pool 满时必须腾位置：
- **swap-out**：把整个 sequence 的 KV 拷到 CPU memory（恢复贵）
- **recompute**：丢掉 KV，下次再 prefill（适合短 prompt）

vLLM 默认 recompute 短请求、swap 长请求。

### 7.3 优先级 / SLA 调度

新出现的 P/D 分离 + 多租户系统需要：
- 排队公平性（FIFO / weighted）
- TTFT SLO（先 prefill 急的）
- TPOT SLO（保持 decode 节奏）

SGLang、TRT-LLM、Mooncake 都引入了 priority-aware scheduler。

---

## 八、三大栈对比

| | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 出身 | UC Berkeley, 2023 | LMSYS, 2024 | NVIDIA |
| 核心创新 | **PagedAttention** | **RadixAttention** + 前端 DSL | NVIDIA 全栈优化 |
| KV 管理 | paged block pool | paged + radix tree | paged（TRT-LLM PA） |
| Prefix cache | APC（自动） | RadixAttention（自动） | 支持 |
| Continuous batching | ✅ | ✅ | ✅ (in-flight batching) |
| Chunked prefill | ✅ | ✅ | ✅ |
| 量化 | AWQ / GPTQ / FP8 / NVFP4 / SqueezeLLM | AWQ / GPTQ / FP8 | 全套 + NVFP4 ⭐ |
| 推测解码 | Medusa / EAGLE / NGRAM | EAGLE / Lookahead | EAGLE / Medusa / draft model |
| PD 分离 | 实验中 / 部分支持 | 部分支持 | 是（与 Triton + Dynamo） |
| 易用性 | OpenAI-compatible API 几行启动 | 前端有 RadixAttention DSL | 偏 C++ / Python 工程化 |
| 生态 | 开源、社区最大 | 开源、增长最快 | 商业、NVIDIA 硬件优化最深 |

> 2024–2025 的"事实标准"：**vLLM + SGLang 在开源端互相追赶**，TensorRT-LLM 在 NVIDIA 硬件上吞吐天花板更高。Mooncake/DistServe 等 PD 分离系统通常作为更上层架构。

---

## 九、关键指标体系

```
吞吐指标:
  Total throughput = tokens / second （所有请求合计）

延迟指标:
  TTFT (Time To First Token)   ← prefill 时间
  TPOT (Time Per Output Token) ← decode 时间 / token
  Tail latency P95/P99         ← 长尾

资源指标:
  GPU utilization (SM occupancy)
  KV cache hit rate
  Memory utilization (KV pool 利用率)
```

**优化哲学**：throughput 与 latency 经常有 trade-off（大 batch 提 throughput 但每 token latency 升高）。chunked prefill + PD 分离 + 推测解码是当前主要的 Pareto 推进手段。

---

## 十、关键问答

**Q1：PagedAttention 和 Continuous Batching 是同一回事吗？**
A：不是，且经常被混淆。Continuous batching 是**调度策略**（每步重组 batch）；PagedAttention 是**内存管理机制**（分页存 KV）。两者正交叠加：你可以单独有 continuous batching 而不分页（如早期 TGI），但分页极大降低了 fragmentation，使 continuous batching 的 batch size 上限提高。

**Q2：RadixAttention 比 PagedAttention 更先进吗？**
A：不是替代关系，是不同抽象层。PagedAttention 解决"**KV 物理内存怎么存**"，RadixAttention 解决"**多请求之间前缀怎么共享**"。SGLang 内部仍用类似分页的物理 block 池；Radix Tree 在这之上索引共享前缀。生产里两者叠加。

**Q3：Chunked prefill 为什么能降低 TTFT 长尾？**
A：朴素调度下，一个 16K 长 prompt 一旦开始 prefill，会独占 GPU 直至结束——后排小请求 TTFT 被拖到秒级。chunked prefill 把这个 prefill 切成 N 个 chunk，每个 chunk 留出 token-budget 给其他请求的 decode/prefill，长 prompt 的总时间几乎不变，但短请求 TTFT 降到 100ms 级。

**Q4：CUDA Graph 会不会影响推测解码？**
A：会。推测解码每次的"接受 token 数"是动态的（1～k+1），shape 不固定 → 与 CUDA Graph 静态 shape 假设冲突。解决办法：为每个可能的接受数录一个 graph，或者用 conditional graph / dynamic graph API。TRT-LLM 在这方面工程更成熟。

**Q5：prefix cache 命中率怎么算 / 怎么调？**
A：命中率 = 命中的前缀 token 数 / 请求总输入 token 数。调节方向：（1）增大 KV pool 容量给 prefix cache；（2）按租户分 cache namespace 避免污染；（3）保留固定 system prompt 的 pinned cache；（4）跨节点 prefix cache 共享（KV transfer，见 Mooncake §12.4）。

**Q6：vLLM 抢占（preempt）什么时候触发，怎么影响 SLO？**
A：当 KV pool 满、新 token 无 block 可分配时触发。被抢占的请求：短 prompt → recompute（下次重做 prefill），长 prompt → swap to CPU（KV 拷回拷出）。抢占会使被抢请求的 TPOT 出现毛刺，长尾 P99 飙升。缓解：留 KV pool reserve、降低 max concurrent、用 PD 分离把 decode 隔离。

**Q7：为什么 SGLang 在 agent / 多轮 场景常常比 vLLM 快？**
A：agent 场景里 system prompt + tool description + 历史多轮**重复率极高**。RadixAttention 自动把这些公共前缀树形组织，命中率经常 > 80%；同样的 7B 模型，SGLang 在 ReAct/CoT 类工作负载上比 vLLM 吞吐高 1.5–3×。普通单轮请求两者接近。

---

## 参考资料

### 必读论文 ⭐⭐
1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM SOSP 2023 ⭐⭐
2. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — OSDI 2022（Continuous Batching 奠基） ⭐⭐
3. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — LMSYS, 2023 ⭐⭐
4. [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) ⭐

### 工程文档 / 博客
5. [vLLM Documentation](https://docs.vllm.ai/) ⭐⭐
6. [SGLang docs + RadixAttention blog](https://lmsys.org/blog/2024-01-17-sglang/) ⭐⭐
7. [TensorRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/) ⭐
8. [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference) — continuous batching 早期开源参考

### 调度与系统设计
9. [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — prefill/decode 拆分
10. [DistServe](https://arxiv.org/abs/2401.09670)
11. [Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079) ⭐⭐

### 综述 / 入门
12. [LLM Inference Performance Engineering: Best Practices (Databricks)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) ⭐
13. [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)
