# 9.4 Expert Parallelism、All-to-All 与 Capacity

[← 返回框架](../../README.md) · [📎 materials.md → §9.4](../../materials.md)

---

## 〇、本节回答什么

§9.1-9.3 解决"算法 / 架构层" → 本节是**工程层**：

> 256 个 expert（DeepSeek-V3）一块 GPU 装不下，必须分卡。token 在不同 GPU 间跳转，是 MoE 的通信噩梦。
>
> 怎么把 expert 切分到多卡 / 多机？All-to-All 通信怎么搞？capacity / token drop 怎么协调？

本节按"算账→优化→约束→嵌套"四步：

1. **算账**：朴素 EP 单次 forward 需要几次 All-to-All？通信量 vs 计算量比是多少？什么时候通信变瓶颈？
2. **路由优化**：通过 **Device-Limited / Node-Limited Routing**（DSv3 grouped TopK）控制通信扇出
3. **执行优化**：通信-计算 overlap，DualPipe，FP8 通信
4. **嵌套**：EP 与 DP / TP / PP / SP 五维并行如何嵌套（5D 并行）

---

## 一、为什么需要 Expert Parallelism

### 1.1 朴素方案的显存账

设 DSv3 配置：

- 256 routed expert + 1 shared，每个 $\approx 60M$ 参数（FP8 量化后）
- 总 expert 参数 $\approx 257 \times 60M = 15.4 GB$（FP8 时）
- BF16 训练时翻倍 ≈ 31 GB **per layer**（共 61 层 → 1.9 TB）

显然单卡装不下，必须切分。

**两种朴素方案**：

**方案 A — 数据并行（DP）下复制 expert**：
- 每张 GPU 都存全部 expert
- 每张 GPU 算自己 batch 的 router 决定 → 算自己 batch 的 expert
- 通信：只有梯度 all-reduce，无 token 跨卡传输
- **问题**：显存爆（每卡都装 1.9 TB），且每卡都得算每个 expert → 浪费 N/K 倍算力

**方案 B — 把 expert 切到不同 GPU（EP）**：
- 每张 GPU 只持有 N / EP_size 个 expert
- token 通过 router 决定去哪张 GPU → 跨卡传输
- 通信：All-to-All 把 token 送到目标卡，算完再送回
- **问题**：通信开销大，All-to-All 是 MoE 的主要瓶颈

DSv3 选 EP=64：每卡 4 个 expert，刚好 H100 80GB 显存能装下。

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

## 二、All-to-All 通信详解

### 2.1 一次 MoE forward 的通信流程

```
Step 1: 每 GPU 上的 token 算 router → 知道每 token 该去哪个 expert
Step 2: 按 expert 编号分组 token，组装 send buffer
Step 3: ────── ALL-TO-ALL #1（dispatch / forward） ──────
         每 GPU 把"应该去远端"的 token 发出去
         同时收下"应该来本地"的 token
Step 4: 本地 expert 计算 FFN(本地 token)
Step 5: ────── ALL-TO-ALL #2（combine / forward） ──────
         把 expert 输出送回原 GPU
Step 6: token 在原 GPU 上加权求和（top-K 输出）

Backward:
Step 7: ─ ALL-TO-ALL #3（combine reverse / backward） ─
Step 8: 反传 expert 计算
Step 9: ─ ALL-TO-ALL #4（dispatch reverse / backward） ─
Step 10: router 反传
```

→ 一次 MoE 层 forward + backward = **4 次 All-to-All**。

### 2.2 通信量精确公式

设：
- $B$ = batch size（per GPU），$T$ = sequence length（per token = 1 token）
- 总 token 数（per GPU）= $B \cdot T$
- $K$ = top-K
- $d$ = hidden dim
- $N$ = expert 数
- $EP$ = expert parallel size

**Dispatch (forward) 通信量**（per GPU per layer）：

```
每 token 发 K 份到 K 个 expert (假设这 K 个 expert 平均分散到 K 张卡)
每份大小 = d (FP8) 字节
每 GPU 发出: B·T·K·d 字节
每 GPU 收到: 同样 B·T·K·d 字节 (对称)
```

**完整 MoE 层 4 次 All-to-All 总量**：

$$\text{Total comm} = 4 \cdot B \cdot T \cdot K \cdot d \text{ bytes per layer per GPU}$$

**与计算量的比例**：

每 GPU 的 expert 计算 FLOPs ≈ $B \cdot T \cdot K \cdot d \cdot d_{\text{ff}} \cdot 2$（前向 + 反向 × 2）

通信-计算比：

$$\frac{\text{Comm bytes}}{\text{Compute FLOPs}} = \frac{4 \cdot B \cdot T \cdot K \cdot d}{B \cdot T \cdot K \cdot d \cdot d_{\text{ff}} \cdot 2} = \frac{2}{d_{\text{ff}}}$$

对 DSv3（$d_{\text{ff}}^{\text{expert}} = 2048$），通信:计算 ≈ $1:1024$ FLOPs/byte。

**这意味着**：

- 单机内（NVLink 900 GB/s，H100 算力 ~1000 TFLOPS）：通信 ~1μs，计算 ~1ms → 通信几乎不可见
- 跨机（InfiniBand 50 GB/s）：通信 ~20μs，计算 ~1ms → 通信占 ~2%
- **但实际情况更糟**：负载不均、kernel launch overhead、batch 小 → 实测通信占 **30-50% MoE 总时间**

### 2.3 实测：通信是 MoE 的主要瓶颈

实测：在 8 卡 H100 + InfiniBand 上，1B token batch 的 All-to-All 单次约 50-200ms。

为什么实测比理论慢这么多：

| 原因 | 影响 |
|---|---|
| 负载不均 | 热门 expert 链路堵塞，单 GPU 成瓶颈 |
| Kernel launch overhead | 每次 All-to-All 有固定开销 |
| Token 排序与重组 | 需要 sort + scatter，额外计算 |
| 同步开销 | All-to-All 是同步原语，落后的卡拖累全队 |
| 跨节点带宽差 | InfiniBand 比 NVLink 慢 20× |

→ 这就是为什么 **§3 路由优化** 和 **§4 执行 overlap** 是 MoE 工程的核心。

### 2.4 All-to-All vs All-Reduce

| 维度 | All-Reduce（DP 用） | All-to-All（EP 用） |
|---|---|---|
| 通信模式 | 每对 GPU 交换 + 累加 | 每对 GPU 单向发送 |
| 数据要求 | 每方有相同形状的数据 | 每方有不同的 send buffer |
| 总通信量（per GPU） | $2 \cdot M (P-1)/P$ | $M \cdot (P-1)/P$ |
| 单次时间 | NCCL Ring/Tree 优化好 | 较散乱，对网络拓扑敏感 |
| 在 MoE 中位置 | 梯度同步 | token dispatch/combine |

→ All-to-All 在通信量上其实更小，但**网络拓扑友好度差**，所以工程上要做分层、overlap 等优化。

---

## 三、Device-Limited / Node-Limited Routing（核心优化）

### 3.1 问题：朴素 TopK 的通信噩梦

DSv3 的 N=256 expert 分布在 EP=64 卡上（每卡 4 个 expert）。
朴素 top-8 routing：

- 每 token 可能选 8 个 expert，分布在最坏 8 张不同 GPU 上
- → 每 token 要发到 8 个远端 → All-to-All 通信扇出 = 8

跨节点更糟：如果 8 个 expert 跨 8 个节点 → 8 次跨节点 InfiniBand 通信。

### 3.2 Node-Limited Routing：硬性限制扇出

DSv3 论文 §2.1.1 引入 **node-limited routing**：

> 限制每个 token **最多发送到 $M$ 个节点**（DSv3 用 $M = 4$）。

实现方式（grouped TopK）：

```
Step 1: 把 N=256 expert 分成 g=8 组，每组 32 expert
        每组对应 1 个节点（节点 0 持有 expert 0-31，节点 1 持有 32-63，...）

Step 2: 对每个 token，计算每组的 "组得分"
        group_score_j = sum(top-2 sigmoid scores in group j)
        # 每组取该组内最高的 2 个 expert 的 sigmoid 加和

Step 3: 选 top-M=4 组（即 token 最多发到 4 个节点）

Step 4: 在被选中的 4 个组里做全局 top-K=8
        即从 4 × 32 = 128 个候选 expert 里选 8 个
```

### 3.3 数学公式

$$g_j = \sum_{i \in \text{top-2 of group } j} s_i, \quad j = 1, \dots, 8$$

$$\text{Selected groups} = \arg\text{TopM}_j(g_j), \quad M = 4$$

$$\text{TopK} = \arg\text{TopK}_{i \in \text{selected groups}}(s_i + b_i), \quad K = 8$$

→ 最终每 token **最多发到 4 个节点**，而不是最坏的 8 个节点 → 跨节点通信 50%。

### 3.4 收益与代价

| 维度 | 朴素 TopK | Node-Limited |
|---|---|---|
| 最坏跨节点扇出 | $K = 8$ | $M = 4$ |
| 平均跨节点通信 | ~6-7 nodes | ~3-4 nodes |
| 路由质量 | 全局最优 | 局部最优（被限在 M 组内）|
| 实测质量损失 | baseline | < 0.5% PPL |
| 实测通信节省 | baseline | ~30-40% |

→ DSv3 论文实验：node-limited 几乎不损质量，但通信大幅减少。这是**算法与硬件协同设计**的典范。

### 3.5 Device-Limited Routing（更细粒度的限制）

某些场景下还有更细的限制：

- **Device-limited**：单 GPU 级别，限制每 token 最多发到 D 张 GPU
- **Rack-limited**：机架级别，限制跨机架通信

DSv3 没用 device-limited（在 node-limited 之外），但 Megatron-MoE 有这个 option。

### 3.6 与 Expert 放置策略的配合

为了让 node-limited routing 有效，**expert 在节点上的物理分布**必须配合：

- **简单方案**：按 expert id 顺序分布（expert 0-31 在节点 0，32-63 在节点 1...）
- **聚类方案**：把相关的 expert 放同节点（提升 node-内命中率）
- **复制方案**：热门 expert 在多节点复制（提升 cache 命中）

DSv3 用顺序分布 + node-limited routing 已经够好。

---

## 四、Overlap 通信与计算

### 4.1 朴素串行的代价

```
朴素：
  Layer N comm₁ → Layer N compute → Layer N comm₂
                                         ↓
                                    Layer N+1 comm₁ → ...

  时间 = 4 · comm_time + compute_time + ...   (全串行)
```

如果 comm_time ≈ 30% compute_time，那 MoE 层时间 = $1.6 \times$ pure compute。

### 4.2 Overlap：用计算掩盖通信

```
Overlap:
  comm₁ ─→ comm₂ ─→ comm₃
        compute₁ ─→ compute₂       (并行)

  时间 ≈ max(comm_total, compute_total)
```

把 forward 切成 chunks：
- 当 chunk_i 在计算时，chunk_{i+1} 已经在通信
- 用 CUDA stream 并发

### 4.3 DeepSpeed-MoE 的 chunk 策略

[Rajbhandari et al. 2022 — DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 提出：

- 把 token 分成 $C$ 个 chunk（如 $C = 4$）
- 用两个 CUDA stream 分别跑 comm 和 compute
- 流水线调度让 comm 和 compute overlap

实测可掩盖 ~50% 的 All-to-All 时间，让 MoE 层时间从 $1.6 \times$ 降到 $\sim 1.2 \times$ pure compute。

### 4.4 DSv3 的 DualPipe（双流水线）

DSv3 论文 §3.2.1 提出更激进的方案：**DualPipe**。

**核心思想**：不只是 MoE 层内 overlap，而是**整个 forward + backward 全程 overlap**。

```
DualPipe 调度：

  Pipeline 1: F_0 ─→ F_1 ─→ F_2 ─→ ... ─→ F_n
  Pipeline 2:        B_0 ─→ B_1 ─→ B_2 ─→ ... ─→ B_n

  其中 F_i = 第 i 个 micro-batch 的 forward
       B_i = 第 i 个 micro-batch 的 backward
       两条 pipeline 并行执行 → forward 和 backward 同时跑
```

**关键技巧**：

1. **MoE 层的 forward 通信** 与 **MoE 层的 backward 计算** overlap
2. **MoE 层的 backward 通信** 与 **MoE 层的 forward 计算** overlap
3. 由两个 CUDA stream + 精细的依赖管理实现

**收益**：

- DSv3 实测 DualPipe 把 MoE 通信几乎完全隐藏
- pipeline bubble（PP 流水线空泡）也被压缩
- 整体训练 throughput 提升 ~30%

### 4.5 FP8 通信（DSv3 关键优化）

DSv3 还做了另一项：**通信用 FP8，计算用 BF16**（或混合）。

- All-to-All 数据用 FP8（1 字节 per element）
- 接收端反量化到 BF16 做 expert FFN 计算
- 输出再 FP8 量化送回

通信量 → BF16 的 50%（FP8 是 1 字节 vs BF16 是 2 字节）。

**质量影响**：

- FP8 通信 + 适当 scaling → 训练 loss 几乎不变
- DSv3 报告：FP8 训练相比 BF16 训练，最终质量差距 < 0.5%

### 4.6 Pipeline Parallel × EP 的 bubble

EP 是 layer-wise 操作（每个 MoE 层都通信），和 Pipeline Parallel 的 micro-batch 调度耦合：

- bubble（pipeline 空泡）会被 All-to-All 进一步放大
- 解法：增大 micro-batch 数（GPipe 1F1B）+ 1F1B scheduling 与 EP 重叠
- DSv3 用 DualPipe 解决了这个

---

## 五、Capacity Factor 与 Token Dropping（工程视角）

§9.2 提到 capacity 的算法侧。本节看工程侧。

### 5.1 Capacity 决定 buffer 大小

```
每 expert 的 input buffer = capacity_factor × (T·K/N)
```

- capacity = 1.0：紧，dropless 模型可用
- capacity = 1.5：留 50% buffer，多数情况不 drop
- capacity = 2.0：很宽松，浪费显存

工程实现：buffer 是**固定大小**的张量（避免动态分配），不够装就 drop。

### 5.2 Dropless 怎么处理 buffer

[MegaBlocks](https://arxiv.org/abs/2211.15841) / DSv3 的 dropless 实现：

- 不用固定 buffer，按实际 token 数分配
- 用 block-sparse GEMM 处理变长输入
- 通信时按实际量传（不 pad）

→ "dropless" 在工程上不是免费的：需要更复杂的 kernel + 通信调度。

### 5.3 多机 EP 的 "hot spot" 问题

当负载不均时：

- 某些 GPU（持有热门 expert）收到远超平均的 token
- 该 GPU 显存爆 / 算力饱和
- All-to-All 跨这个 GPU 的链路堵塞

→ 这就是为什么 §9.2 的 **负载均衡** 在工程上比算法上更重要：**不均衡 = 通信热点**。

**协同设计**：

- §9.2 的 aux-loss-free bias → 保证 batch level 均衡
- §9.2 的 sequence-level loss → 保证 sequence 不打堆
- §9.4 的 node-limited routing → 限制最坏扇出
- §9.4 的 DualPipe → overlap 隐藏剩余通信

→ 这是个**算法 + 系统**的联合优化，单看任何一层都不够。

---

## 六、5D 并行：EP 与其他并行的组合

LLM 训练里常见 5 个并行维度：

```
DP   (Data Parallel)            ── 数据切分（最常用）
TP   (Tensor Parallel)          ── 层内切分（Megatron）
PP   (Pipeline Parallel)        ── 层间流水
SP   (Sequence Parallel)        ── 序列切分（长 context，§10）
EP   (Expert Parallel)          ── MoE 专家切分（本节）
```

**DSv3 训练实例**（节选）：

- DP = 32（数据并行复制 32 份）
- TP = 1（DSv3 实测 TP 收益小 → 不切）
- PP = 16（16 段 pipeline）
- EP = 64（256 expert / 4 = 64 张 GPU 持 expert）
- 总 GPU = 32 × 16 = 512 个 device group，每组 EP 内通信

→ **EP 通常与 DP 嵌套**：DP 复制整套 MoE，EP 在 DP 内分 expert。

### 6.1 EP × TP 的冲突

如果 TP 把单个 expert 也切到多张 GPU：

- expert 内部 GEMM 也变成跨卡 → 二次通信
- 通常**不与 TP 叠加**，要么用 EP，要么用 TP

DSv3：放弃 expert 上 TP，纯靠 EP。

### 6.2 EP × PP 的协调

- PP 把不同 layer 分到不同 GPU stage
- 每个 stage 内还要做 EP（如果该 stage 有 MoE 层）
- → EP 通信只在 stage 内发生，不跨 PP stage
- DSv3 的 DualPipe 把 PP bubble 和 EP 通信一起优化

### 6.3 完整嵌套图

```
DP=32 (复制 32 套整模型)
  └─ PP=16 (每套切 16 段)
       └─ EP=64 (每段内 64 张卡分 expert)
              └─ 单卡持有 4 个 expert，FFN 计算

总 GPU = 32 × 16 × 64 = 32768 张 H100
DSv3 实际用了 ~2000 张 H100，训练 ~2 个月
```

---

## 七、推理时的 MoE 调度

### 7.1 推理 vs 训练的差异

- 训练：batch 大、token 多 → All-to-All 可以摊薄
- 推理：batch 小（甚至 1）、decode 阶段一次一个 token → All-to-All 几乎不能摊薄

```
训练：1000 token × 8 expert dispatch = 8000 跨卡 token (摊薄)
推理 decode：1 token × 8 expert dispatch = 8 跨卡 token (开销固定)
```

→ MoE 模型的 **decode 阶段** 是它和 dense 模型最大的劣势点。

### 7.2 推理时的优化

1. **Expert Caching**：把"热"的几个 expert 复制到所有 GPU，冷的留远端
2. **Expert Offload**：CPU/SSD 上存全部 expert，按需 load（适合极大模型）
3. **Replicated Expert**（vLLM）：所有 GPU 复制 expert，省 All-to-All（但要每 GPU 显存装得下）
4. **EP=1 推理**：单卡装全部 expert（小 MoE 模型如 Mixtral 7B × 8 可行）

### 7.3 推理时的 Continuous Batching

vLLM 的 continuous batching 让 batch 中的 prefill + decode 混杂：

- prefill 阶段每 sequence 几千 token → All-to-All 摊薄好
- decode 阶段每 sequence 1 token → All-to-All 摊薄差
- 混合 batch → 推理引擎要做精细的调度

DSv3 的 SGLang fork 专门优化这个：把 prefill 和 decode 的 MoE 通信分开调度。

### 7.4 工业现状

- vLLM：支持 EP + tensor parallel + replicated expert
- SGLang：类似，加 RadixAttention
- TensorRT-LLM：支持，性能最优但闭源 kernel
- DeepSeek 自研 [SGLang fork](https://github.com/sgl-project/sglang) 优化 V3

### 7.5 推理 latency 拆解

DSv3 在 8 卡 H100 推理（FP8）实测：

| 阶段 | 时间占比 |
|---|---|
| MLA attention | ~30% |
| Shared expert FFN | ~10% |
| Routed expert FFN | ~25% |
| All-to-All (forward dispatch) | ~15% |
| All-to-All (forward combine) | ~15% |
| Router + 其他 | ~5% |

→ All-to-All 占 30% 总推理时间，是 MoE 推理的最大开销。

---

## 八、关键问答

**Q1**：EP 和 DP 怎么嵌套？

- 通常：DP 在外，EP 在内（先复制整模型，再在副本内切 expert）
- 也可：DP 在 expert 维度内（每个 expert 自己 DP）— 仅当 expert 大到一张卡装不下
- DSv3 是前者

**Q2**：All-to-All 比 All-Reduce 慢吗？

- 通信量公式不同：A2A ≈ $M(P-1)/P$，AllReduce ≈ $2M(P-1)/P$
- 实际上 A2A 通信量更小
- 但 A2A 网络拓扑不友好（每对都要单独发），实测往往更慢
- AllReduce 有 Ring / Tree 优化，A2A 难做类似优化

**Q3**：为什么 DSv3 用 EP=64 这么大？

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
- **Node-limited routing**（DSv3 §3.2）：限制跨节点扇出
- DualPipe overlap：用计算掩盖剩余通信

**Q7**：MoE 的工程难度排序？

- 训练算法（router、aux loss）：相对成熟
- 通信优化（All-to-All overlap）：中等难度
- 推理 serving（EP + 调度）：**最难**，每个家都在自研
- 长 context + MoE：当前最 active 的研究方向

**Q8**：Node-limited routing 会影响质量吗？

- DSv3 论文实验：node-limited (M=4) 相比无限制 TopK，质量损失 < 0.5%
- 通信节省 ~30-40%
- 是个"几乎免费"的优化

**Q9**：DualPipe 比传统 1F1B 强多少？

- 1F1B：单流水线，bubble ~20%
- DualPipe：双流水线，bubble < 5%，且 MoE 通信被掩盖
- DSv3 实测：训练 throughput +30%

**Q10**：FP8 通信会掉点吗？

- DSv3 测：FP8 vs BF16 通信，最终 benchmark 差异 < 0.5%
- 关键是 scaling 策略（per-block 量化 + dynamic scaling）
- 通信量减半，推理速度显著提升

---

## 九、本节与其他节关系

```
§9.1 路由        ─→ 决定 token 去哪个 expert（"目标")
§9.2 负载均衡    ─→ 让目标分布均匀（无 hot spot）
§9.3 架构变体    ─→ N、K、shared expert（决定通信量基数）
                  └─ Granularity 越大 → 通信压力越大 → 更需要 §9.4 优化
§9.4 EP（本节）  ─→ 把 expert 切到多卡 + All-to-All + Node-limited + DualPipe
       │
       ├─→ §10 并行化（5D 并行整体框架）
       ├─→ §12 推理优化（MoE serving 是难点）
       └─→ §9.5 选型（EP 难度是 MoE vs dense 权衡的关键）
```

---

## 参考资料

**EP 原始与基础**：
- [Lepikhin et al. 2020 — GShard (sharding 原始)](https://arxiv.org/abs/2006.16668) ⭐
- [Rajbhandari et al. 2022 — DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) ⭐
- [Hwang et al. 2022 — Tutel: Adaptive Mixture-of-Experts](https://arxiv.org/abs/2206.03382)

**Dropless / Kernel**：
- [Gale et al. 2023 — MegaBlocks](https://arxiv.org/abs/2211.15841) ⭐
- [NVIDIA Cutlass Grouped GEMM](https://github.com/NVIDIA/cutlass)

**DSv3 系统优化**：
- [DeepSeek-V3 技术报告 §3 训练系统](https://arxiv.org/abs/2412.19437) ⭐⭐（DualPipe / FP8 通信 / node-limited）

**框架与工具**：
- [Megatron-LM MoE 文档](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [vLLM MoE Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [SGLang DeepSeek 优化博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) ⭐
- [NVIDIA NCCL All-to-All docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

**综述**：
- [Sebastian Raschka — Distributed training for MoE](https://magazine.sebastianraschka.com/p/the-state-of-llms-in-2024)
- [HuggingFace — How to train MoE efficiently](https://huggingface.co/blog/moe)
