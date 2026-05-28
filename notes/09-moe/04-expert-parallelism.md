# 9.4 Expert Parallelism、All-to-All 与 Capacity

[← 返回框架](../../README.md) · [📎 materials.md → §9.4](../../materials.md)

---

## 〇、本节回答什么

§9.1-9.3 解决"算法 / 架构层" → 本节是**工程层**：

> 256 个 expert（DeepSeek-V3）一块 GPU 装不下，必须分卡。token 在不同 GPU 间跳转，是 MoE 的通信噩梦。
>
> 怎么把 expert 切分到多卡 / 多机？All-to-All 通信怎么搞？capacity / token drop 怎么协调？

主线：**Expert Parallelism（EP）** + **All-to-All** + **Overlap** + **Capacity 调度**。

与 §10 并行化（TP/DP/PP/SP）配合，构成"5D 并行"。

---

## 一、为什么需要 Expert Parallelism

### 1.1 朴素方案的问题

如果每张 GPU 都存全部 N 个 expert（数据并行 DP）：
- 一个 expert 64MB（典型 7B FFN 切分），256 个 = 16GB
- 加上 activation / KV，单卡装不下
- 而且每张卡都得算每个 expert → 浪费 = $\Omega(N/K)$ 倍

→ 必须 **切分 expert 到不同 GPU**。

### 1.2 Expert Parallelism 的定义

```
EP（Expert Parallelism）:
  把 N 个 expert 平均分到 EP_size 张 GPU 上
  每张 GPU 只持有 N / EP_size 个 expert 的参数
  token 通过 router 决定去哪张 GPU → 跨卡 All-to-All
```

```
4 张 GPU, N=8 expert, EP=4:
  GPU 0: Expert 0, 1
  GPU 1: Expert 2, 3
  GPU 2: Expert 4, 5
  GPU 3: Expert 6, 7
```

→ 单卡参数 = N/EP × FFN，可控。

---

## 二、All-to-All 通信

### 2.1 一次 MoE forward 的通信流程

```
Step 1: 每 GPU 上的 token 算 router → 知道每 token 该去哪个 expert
Step 2: 按 expert 编号分组 token
Step 3: ────── ALL-TO-ALL #1（dispatch） ──────
         每 GPU 把"应该去远端"的 token 发出去
         同时收下"应该来本地"的 token
Step 4: 本地 expert 计算 FFN(本地 token)
Step 5: ────── ALL-TO-ALL #2（combine） ──────
         把 expert 输出送回原 GPU
Step 6: token 在原 GPU 上加权求和（top-K 输出）
```

→ 一次 MoE 层 forward = **2 次 All-to-All**（dispatch + combine），backward 还要再 2 次 → 共 4 次。

### 2.2 通信量公式

每张 GPU 每次 All-to-All 传输：
$$\text{volume} = \frac{T \cdot K \cdot d}{EP_{size}}$$

- $T$：batch token 数
- $K$：top-K
- $d$：hidden dim
- $EP_{size}$：EP 卡数

实测：在 8 卡 H100 + InfiniBand 上，1B token batch 的 All-to-All 单次约 50-200ms，**占 MoE 总时间 30-50%**。

→ 通信是 MoE 的主要瓶颈，不是计算。

### 2.3 优化 All-to-All

```
1. NVLink / NVSwitch：单机 8 卡内极快（900GB/s）
2. InfiniBand：跨机（400Gbps）— 比 NVLink 慢一个量级
3. 分层 All-to-All：先节点内 reduce，再跨节点 → 减少跨节点流量
4. Overlap：用计算掩盖通信
```

---

## 三、Overlap 通信与计算（关键工程）

### 3.1 思路

```
朴素：
  comm → compute → comm → compute    (串行)

Overlap:
  comm₁ ─→ comm₂ ─→ comm₃
        compute₁ ─→ compute₂        (并行)
```

把 forward 切成 N 个 chunk：
- 当 chunk_i 在计算时，chunk_{i+1} 已经在通信
- 用 CUDA stream 并行

### 3.2 DeepSpeed-MoE / Megatron-MoE 的实现

DeepSpeed-MoE 在 [Rajbhandari et al. 2022](https://arxiv.org/abs/2201.05596) 提出：
- 把 token 分成 chunks，stream 并发 dispatch + compute
- 实测可掩盖 ~50% 的 All-to-All 时间

DeepSeek-V3 进一步：
- 使用 **bf16 通信 + fp8 计算** 混合精度
- 自定义 NVSHMEM-based 通信 kernel
- 见 [DSv3 论文 §3.2](https://arxiv.org/abs/2412.19437)

### 3.3 Pipeline Parallel × EP 的 bubble

EP 是 layer-wise 操作（每个 MoE 层都通信），和 Pipeline Parallel 的 micro-batch 调度耦合：
- bubble（pipeline 空泡）会被 All-to-All 进一步放大
- 解法：增大 micro-batch 数（GPipe 1F1B）+ 1F1B scheduling 与 EP 重叠

---

## 四、Capacity Factor 与 Token Dropping（再聊）

§9.2 提到 capacity 的算法侧。本节看工程侧。

### 4.1 Capacity 决定 buffer 大小

```
每 expert 的 input buffer = capacity_factor × (T·K/N)
```

- capacity = 1.0：紧
- capacity = 1.5：留 50% buffer，多数情况不 drop
- capacity = 2.0：很宽松

工程实现：buffer 是**固定大小**的张量（避免动态分配），不够装就 drop。

### 4.2 Dropless 怎么处理 buffer

[MegaBlocks](https://arxiv.org/abs/2211.15841) / DeepSeek 的 dropless 实现：
- 不用固定 buffer，按实际 token 数分配
- 用 block-sparse GEMM 处理变长输入
- 通信时按实际量传（不 pad）

→ "dropless" 在工程上不是免费的：需要更复杂的 kernel + 通信调度。

### 4.3 多机 EP 的"hot spot"问题

当负载不均时：
- 某些 GPU（持有热门 expert）收到远超平均的 token
- 该 GPU 显存爆 / 算力饱和
- All-to-All 跨这个 GPU 的链路堵塞

→ 这就是为什么 §9.2 的 **负载均衡** 在工程上比算法上更重要：**不均衡 = 通信热点**。

---

## 五、5D 并行：EP 与其他并行的组合

LLM 训练里常见 5 个并行维度：

```
DP   (Data Parallel)            ── 数据切分
TP   (Tensor Parallel)          ── 层内切分（如 Megatron）
PP   (Pipeline Parallel)        ── 层间流水
SP   (Sequence Parallel)        ── 序列切分（长 context，§10）
EP   (Expert Parallel)          ── MoE 专家切分（本节）
```

DeepSeek-V3 训练实例（节选）：
- DP = 32
- TP = 1（DeepSeek 实测 TP 收益小 → 不切）
- PP = 16
- EP = 64
- 总 GPU = 32 × 16 = 512 个 device group，每组 EP 内通信

→ **EP 通常与 DP 嵌套**：DP 复制整套 MoE，EP 在 DP 内分 expert。

### 5.1 EP × TP 的冲突

如果 TP 把单个 expert 也切到多张 GPU：
- expert 内部 GEMM 也变成跨卡 → 二次通信
- 通常**不与 TP 叠加**，要么用 EP，要么用 TP

DeepSeek-V3：放弃 expert 上 TP，纯靠 EP。

---

## 六、推理时的 MoE 调度

### 6.1 推理 vs 训练的差异

- 训练：batch 大、token 多 → All-to-All 可以摊薄
- 推理：batch 小（甚至 1）、decode 阶段一次一个 token → All-to-All 几乎不能摊薄

```
训练：1000 token × 8 expert dispatch = 8000 跨卡 token (摊薄)
推理 decode：1 token × 8 expert dispatch = 8 跨卡 token (开销固定)
```

→ MoE 模型的 **decode 阶段** 是它和 dense 模型最大的劣势点。

### 6.2 推理时的优化

1. **Expert Caching**：把"热"的几个 expert 复制到所有 GPU，冷的留远端
2. **Expert Offload**：CPU/SSD 上存全部 expert，按需 load
3. **Replicated Expert**（vLLM）：所有 GPU 复制 expert，省 All-to-All（但要每 GPU 显存装得下）
4. **EP=1 推理**：单卡装全部 expert（小 MoE 模型如 Mixtral 7B × 8 可行）

### 6.3 工业现状

- vLLM：支持 EP + tensor parallel + replicated expert
- SGLang：类似，加 RadixAttention
- TensorRT-LLM：支持，性能最优但闭源 kernel
- DeepSeek 自研 [SGLang fork](https://github.com/sgl-project/sglang) 优化 V3

---

## 七、关键问答

**Q1**：EP 和 DP 怎么嵌套？
- 通常：DP 在外，EP 在内（先复制整模型，再在副本内切 expert）
- 也可：DP 在 expert 维度内（每个 expert 自己 DP）— 仅当 expert 大到一张卡装不下
- DeepSeek-V3 是前者

**Q2**：All-to-All 比 All-Reduce 慢吗？
- 通信量公式相同，但 All-to-All 更"零散"（每对 GPU 都要单独发送）
- 对网络拓扑要求更高
- 但比 All-Reduce 灵活（不要求每方有相同数据）

**Q3**：为什么 DeepSeek-V3 用 EP=64 这么大？
- 256 expert / EP=64 = 每卡 4 个 expert，刚好显存够
- 太小 EP：单卡装不下；太大 EP：通信占比涨
- 64 是 H100 + InfiniBand 集群上的 sweet spot

**Q4**：MoE 推理为什么比训练困难？
- 训练 batch 大，All-to-All 通信被多 token 摊薄
- decode 一次一个 token → All-to-All 单位通信几乎是固定开销
- 实测：MoE 模型 decode latency 比同激活 dense 高 1.5-3×（除非用 replicated expert）

**Q5**：单机 8 卡能跑 DeepSeek-V3 吗？
- 671B 总参，FP8 量化约 670GB → 8 卡 H100 (80GB×8=640GB) 紧
- BF16 需要 ~1.3TB → 多机
- vLLM / SGLang 上单机 + INT4 量化 + EP=8 可以勉强跑

**Q6**：expert 跨节点通信怎么优化？
- 节点内：NVLink/NVSwitch（900GB/s）
- 跨节点：InfiniBand 400Gbps（~50GB/s）
- 分层 All-to-All：先节点内合并，再跨节点
- DeepSpeed-MoE / DeepSeek 都用类似策略

**Q7**：MoE 的工程难度排序？
- 训练算法（router、aux loss）：相对成熟
- 通信优化（All-to-All overlap）：中等难度
- 推理 serving（EP + 调度）：**最难**，每个家都在自研
- 长 context + MoE：当前最 active 的研究方向

---

## 八、本节与其他节关系

```
§9.1 路由        ─→ 决定 token 去哪个 expert（"目标")
§9.2 负载均衡    ─→ 让目标分布均匀（无 hot spot）
§9.3 架构变体    ─→ N、K、shared expert（决定通信量基数）
§9.4 EP（本节）  ─→ 把 expert 切到多卡 + All-to-All
       │
       ├─→ §10 并行化（5D 并行整体框架）
       ├─→ §12 推理优化（MoE serving 是难点）
       └─→ §9.5 选型（EP 难度是 MoE vs dense 权衡的关键）
```

---

## 参考资料

- [Lepikhin et al. 2020 — GShard (sharding 原始)](https://arxiv.org/abs/2006.16668) ⭐
- [Rajbhandari et al. 2022 — DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) ⭐
- [Hwang et al. 2022 — Tutel: Adaptive Mixture-of-Experts](https://arxiv.org/abs/2206.03382)
- [Gale et al. 2023 — MegaBlocks](https://arxiv.org/abs/2211.15841) ⭐
- [DeepSeek-V3 技术报告 (§3 训练系统)](https://arxiv.org/abs/2412.19437) ⭐⭐
- [Megatron-LM MoE 文档](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [vLLM MoE Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [SGLang DeepSeek 优化博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) ⭐
- [NVIDIA NCCL All-to-All docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [Sebastian Raschka — Distributed training for MoE](https://magazine.sebastianraschka.com/p/the-state-of-llms-in-2024)
- [HuggingFace — How to train MoE efficiently](https://huggingface.co/blog/moe)
