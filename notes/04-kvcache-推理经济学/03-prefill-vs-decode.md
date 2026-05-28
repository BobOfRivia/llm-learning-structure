# 4.3 Prefill vs Decode

[← 返回框架](../../README.md) · [📎 materials.md → §4.3](../../materials.md)

---

## 一、两个阶段的定义

LLM 推理的请求生命周期：

```
用户输入 prompt: "请讲一个故事"
       │
       ▼
┌─────────────────────────────────────────────┐
│ Stage 1: PREFILL                            │
│  - 一次性算 prompt 的所有 token             │
│  - 输出 first token + 填满 KV-cache         │
│  - 时长 ≈ TTFT (Time To First Token)        │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│ Stage 2: DECODE                             │
│  - 自回归生成，每步 1 个 token              │
│  - 复用 KV-cache，append 新 K,V             │
│  - 速度 ≈ TPOT (Time Per Output Token)      │
│  - 直到 <eos> 或达到 max_tokens             │
└─────────────────────────────────────────────┘
```

---

## 二、Prefill 和 Decode 的本质差异

| 维度 | Prefill | Decode |
|------|---------|--------|
| 单步算 token 数 | prompt_len (e.g., 2k) | **1** ⭐ |
| 每层 GEMM 形状 | $(L_p, d) \times (d, d)$ | $(1, d) \times (d, d)$ |
| Attention 形状 | $L_p \times L_p$ | $1 \times (L_p + t)$ |
| Arithmetic Intensity | 高（与 GEMM 训练接近） | 低（B 决定） |
| Bound | **compute-bound** | **memory-bound** ⭐ |
| 瓶颈资源 | TFLOPs | HBM 带宽 |
| 用户感知 | TTFT（首 token 延迟） | TPOT（流式速率） |
| 单步可并行 | 整段 token 并行 | 只能跨 request 并行 |

> 同一份模型权重，**prefill 跟训练像，decode 是另一种 workload**。优化策略完全不同。

---

## 三、两个阶段的成本结构

**Prefill 算量**（一次）：
$$\text{FLOPs}_{prefill} \approx 2 N L_p$$

其中 $N$ = 模型参数量，$L_p$ = prompt 长度。
6ND 法则的"1/3"（forward 的算量）。

**Decode 算量**（每输出 token）：
$$\text{FLOPs}_{decode} \approx 2 N$$

→ 输出 $T$ 个 token：$\text{FLOPs}_{decode}^{total} \approx 2NT$

**总成本**：
$$\text{FLOPs}_{total} \approx 2N(L_p + T)$$

但实际 wall time **不等比**：
- prefill 一次性算完，GPU 利用率高 → wall time ∝ $L_p / \text{TFLOPs}$
- decode 每步 memory-bound → wall time ∝ $T / \text{bandwidth}$

> 同样多的 FLOPs，prefill 几秒、decode 几十秒——这是 LLM 服务的核心反直觉。

---

## 四、关键 SLO 指标

工业部署对两个阶段定 SLO：

| 指标 | 含义 | 典型目标 |
|------|------|---------|
| **TTFT** | 第一个 token 的延迟 | < 500ms（chat）/ < 2s（长上下文）|
| **TPOT / ITL** | inter-token latency | < 50ms（≈ 20 tokens/s 用户阅读速度）|
| **E2E latency** | 端到端总延迟 | TTFT + T × TPOT |
| **Throughput** | tokens/s（系统级）| 越大越省钱 |
| **Goodput** | 满足 SLO 的吞吐 | 真正可用的吞吐 |

**Trade-off**：
- 增大 batch → throughput ↑，但 TPOT ↑（mem-bound 也有 batch 上限）
- 增大 prompt → TTFT ↑
- 在线（chat）追求 latency；离线（batched inference）追求 throughput

---

## 五、Prefill 与 Decode 的"互相干扰"

传统调度："**iteration-level batching**"把 prefill 请求和 decode 请求混在一个 batch 里：
```
batch step:
  req A (decode, 1 token)
  req B (decode, 1 token)
  req C (prefill, 2048 tokens)   ← 整个 batch 都被它拖慢
```
- prefill 是大 GEMM，但 decode 的 attention 是 memory-bound
- 二者放一起：prefill 让 batch 整体变长 → 单步延迟暴涨
- 表现："stragglers" 拖累 TPOT

---

## 六、Chunked Prefill / 分离式部署

### 6.1 Chunked Prefill（vLLM / SGLang）
把长 prefill 切成 chunk，每步只算一个 chunk，与 decode 请求拼成一个 batch：
```
step 1: prefill chunk 1 (256 tok) + decode batch
step 2: prefill chunk 2 (256 tok) + decode batch
...
```
- 让 TPOT 稳定，TTFT 略增（但可控）
- 是 2024+ 主流方案

### 6.2 Disaggregated Serving（DistServe / Splitwise）
**物理上把 prefill 和 decode 分到不同 GPU/节点**：
- Prefill 节点：compute-heavy GPU（H100, B200）
- Decode 节点：memory-heavy GPU（H200, MI300X 带宽更高）
- 之间通过 KV-cache 迁移（RDMA + GPUDirect，详见 [§4.6 RDMA 与 GPUDirect](./06-rdma与gpudirect.md)）

代表系统：
- **DistServe** ([Zhong et al. 2024](https://arxiv.org/abs/2401.09670))
- **Splitwise** (Microsoft)
- **Mooncake** (Moonshot, DeepSeek 也类似)

收益：每阶段独立调度，goodput 提升 4-7×。

---

## 七、计算示例

**场景**：Llama-3 70B BF16，H100 80G x 8 张，TP=8。
- prompt = 4k, output = 1k
- prefill FLOPs ≈ 2 × 70G × 4k = 560 TFLOPs → 单卡 70 TFLOPs
- H100 BF16 ≈ 990 TFLOPs，MFU 50% → ~500 TF/s → prefill 时间 ~140ms ⭐
- decode 每 token ≈ 2 × 70G = 140 GFLOPs → 单卡 17.5 GFLOPs
- decode 的瓶颈是带宽：读 70G 参数 / TP=8 / 3.35 TB/s ≈ ~2.6ms/token
- 1k 输出 ≈ **2.6 s**（远大于 prefill）

→ 大部分延迟在 decode。这正是为什么 KV-cache、GQA、speculative decoding 等优化对应的都是 decode。

---

## 关键问答

**Q1**：为什么"长 prompt + 短输出"和"短 prompt + 长输出"是完全不同的优化问题？
- 长 prompt + 短输出：prefill 占主导（compute-bound）→ 关心算力 / FlashAttn / chunked prefill
- 短 prompt + 长输出：decode 占主导（memory-bound）→ 关心 KV-cache / GQA / spec decoding
- RAG / 文档问答 / 摘要：第一类；对话 / 代码生成 / agent：第二类

**Q2**：为什么 TTFT 难以优化？
- prefill 是大 GEMM，已经接近峰值算力
- 唯一减 TTFT 的办法：缩短 prompt（prompt compression、prefix cache）
- 或者用更小的 draft model 做"提前 first token"

**Q3**：Prefix Cache 是什么？
- 多条请求共享 prompt 前缀（system prompt + 历史对话）
- 缓存这部分 KV，新请求来时直接复用
- vLLM、SGLang 都支持
- 对 multi-turn chat / agent / RAG 收益最大（多达 70% prompt 共享）

**Q4**：Speculative decoding 为什么不能用在 prefill？
- prefill 已经是 compute-bound，没有"算力闲置"
- spec decoding 是把闲置算力换成 effective tokens
- prefill 不需要、也无法获益

**Q5**：Continuous batching 解决了什么？
- 传统 batch：所有请求一起开始、一起结束 → 短请求等长请求 → GPU 闲置
- continuous：每步动态加入新请求、退出完成的请求 → 永远满载
- vLLM 的主要创新点之一

**Q6**：什么时候选 disaggregated serving？
- 高并发、长 context 场景（如 RAG、code agent）
- 跨节点 RDMA 带宽足够（>200 Gbps）
- 流量稳定（让两端独立 scaling 有意义）

---

## 参考资料

- [Pope et al. 2022 — Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [Kwon et al. 2023 — vLLM / PagedAttention](https://arxiv.org/abs/2309.06180)
- [Agrawal et al. 2023 — Sarathi: Chunked Prefills + Decode](https://arxiv.org/abs/2308.16369)
- [Zhong et al. 2024 — DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670)
- [Patel et al. 2023 — Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)
- [Mooncake KVCache-centric Disaggregated Architecture (Moonshot)](https://github.com/kvcache-ai/Mooncake)
- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104)
- [Anyscale — Continuous Batching Survey](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [LMSYS — SGLang blog](https://lmsys.org/blog/2024-01-17-sglang/)
