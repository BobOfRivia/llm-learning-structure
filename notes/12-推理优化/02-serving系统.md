# 12.2 Serving 系统（vLLM / PagedAttention / Continuous Batching / SGLang / RadixAttention）

[← 返回框架](../../README.md) · [📎 materials.md → §12.2](../../materials.md)

> **本节要回答的根本问题**：把一个已经训好的 LLM 放到 GPU 上「能跑」很容易（HuggingFace `model.generate()` 三行代码），但把同一块卡的 throughput 拉满、把 P99 latency 压到 SLO 之下，从 2022 年至今催生了一整套 serving 系统。这一节把 vLLM / SGLang / TensorRT-LLM 背后的 **5 大支柱**——continuous batching、PagedAttention、prefix cache、chunked prefill、CUDA Graph——逐个拆到「为什么这样设计、内部数据结构是什么、kernel 怎么改、代价在哪、什么时候会失效」的层次。读完应该能：
>
> 1. 在 HF generate 跑出 10 tok/s 的同一台 H100 上，解释 vLLM 凭什么能跑出 200 tok/s；
> 2. 在面试里把 PagedAttention 和 Continuous Batching 的关系讲清楚（它们不是同一件事）；
> 3. 看到一条 P99 TBT 抖动曲线，能定位到调度器、prefill 干扰还是抢占；
> 4. 在 vLLM / SGLang / TRT-LLM 之间根据 workload（聊天 / agent / 高吞吐离线）做出有依据的选型。

---

## 〇、本节的叙事主线

朴素 HF generate vs SOTA serving 在同一张 H100 上的 throughput 差距常常是 **10–24×**。这条沟不是靠一个 trick 跨过去的，而是 5 个正交优化叠加的结果。我们按下面的因果链走：

```
HF generate 慢                ── 三大病 ──┐
   ├── 静态 batch 尾巴空转              ──→ Continuous Batching（§2）
   ├── KV cache 预分配碎片              ──→ PagedAttention   （§3）
   │                                     ├── Prefix Cache    （§4）
   │                                     └── RadixAttention  （§5）
   └── prefill ↔ decode 资源冲突        ──→ Chunked Prefill  （§6）
                                         ──→ PD 分离 → §12.4

  decode 阶段 kernel launch overhead  ──→ CUDA Graph        （§7）

  以上能力如何被同一个调度器编排        ──→ Scheduler        （§8）
  开关一个个打开的累计效果               ──→ 总览叠加         （§9）
  三大栈如何把这些组合落地              ──→ vLLM/SGLang/TRT  （§10）
  指标体系（throughput / TTFT / TBT / Goodput）── 评测       （§11）
```

主线一句话：**serving 系统的本质是「调度 + 内存管理」**。Attention 算子的优化（FlashAttention、量化、推测解码）是单点 kernel 工作，由 §12.1 / §12.3 / §6.2 覆盖；本节关心的是「上层怎么把请求编排成连续不断的、几乎不浪费 GPU 周期的 forward 流」。

---

## 一、为什么朴素 `generate()` 慢：从一组实测数据出发

### 1.1 实测对比（Llama-3-8B / H100-80G, 200 并发, 输入 512 / 输出 256）

| 栈 | Throughput（tok/s）| TTFT P50 | TPOT P50 | KV 利用率 |
|---|---|---|---|---|
| HF `generate` + static batch=4 | ~600 | 4.8 s | 95 ms | ~12% |
| HF + static batch=64 | ~1,800 | 13 s | 38 ms | ~28% |
| vLLM v0.6 默认 | ~14,000 | 380 ms | 18 ms | ~92% |
| SGLang v0.4（chat workload） | ~17,000 | 320 ms | 17 ms | ~95% |

（数字来自 vLLM/SGLang 官方 benchmark 与社区复测的中位数；不同输入长度差异较大。）

差距来自三个症状，对应三个根因。

### 1.2 症状一：Static batching 的尾巴空转

```
Static batching, 一个 batch 跑完才接下一个:

t→  ════════════════════════════════════════════
req1 ████████████████████████████████  (出 250 token)
req2 █████                              (出 30 token, 但要等)
req3 ████████████                       (出 90 token, 但要等)
                ↑空转       ↑空转  ↑空转
GPU 在 req2/req3 早已结束后，仍在为 req1 单独算 forward
batch 槽位被「已死」请求占着无法替换
```

#### 1.2.1 一次 iteration 内，GPU 到底在算什么

为什么「req2 在第 30 步就出了 EOS，槽位却必须等 req1 跑到第 250 步才能让出来」？把视角放进单次 `model.forward()` 里看就清楚了。

**张量布局**（朴素 HF `generate(batch_size=4, max_new_tokens=256)`）：

```
input_ids:      [B=4, max_len=256]                              ← 编译期固定形状
attention_mask: [B=4, max_len=256]                              ← 1=真 token, 0=padding
KV cache:       [B=4, num_layers, 2, max_len, n_heads, head_dim]
                  ↑
                  B 维度在 CUDA kernel 启动时就锁死了
```

**先约定一个词:什么叫 "slot"**

上面所有张量的 batch 维度 `B=4` 都有 4 个位置,每一个位置就叫一个 **slot(槽位)**。一个 slot 在它服务的请求结束之前,**固定绑定**一个请求——这是 batched kernel 的硬约束(同一个 GEMM 一次只能跑一个固定 shape)。可以画成:

```
            B 维度 (batch axis)
slot 0  →  [ req1: input | KV[req1, all layers] | output ]
slot 1  →  [ req2: input | KV[req2, all layers] | output ]
slot 2  →  [ req3: input | KV[req3, all layers] | output ]
slot 3  →  [ req4: input | KV[req4, all layers] | output ]
            ↑                                        ↑
            shape 编译期定死                          forward 完整跑完一次
```

理解 slot 的两个推论:

1. **「槽位被已死请求占着」= batch 维度的某一行 tensor 还指向那个请求的 KV**。即使请求已经 EOS,这一行的 Linear / Attention 照常算,只是输出被 attention_mask 抹成 0(见 §1.2.2)。
2. **「让出 slot」≠ 改个指针**。要真的把 slot 1 从 req2 切到队列里的 req4,得先**释放** req2 在 KV cache 第二行(`KV[1, ...]`)的所有 layer 数据、再把 req4 的 prompt 跑一次 prefill 灌进去——这就是下面要展开的「换 slot 要做的三件事」。

§1 / §2 / §3 之后出现的「slot」都是这个含义:**batch 维度上一个被请求占据的固定位置**。

**单步 decode iteration 干的事**(按算子展开):

```
1. 取每个 slot 的 next_token_id      → [B] 个 id
2. embed → Linear(Q,K,V)             → [B, 1, 3*d]   ← batched GEMM
3. attention(Q[B,1,d],
             K_cache[B,t,d],
             V_cache[B,t,d])         → [B, 1, d]    ← batched attention
4. Linear → softmax → sample          → [B] 个 next_token
5. 写回 K_cache[B,t+1], V_cache[B,t+1]
```

**关键点**:步骤 2 / 3 / 4 全是形状为 `[B, ...]` 的 batched 算子,**B 不能在 forward 中途换**。如果你想「让 req2 退出、把队列里的 req4 塞进 slot 1」,要做三件事:

1. **清 KV**:slot 1 里还压着 req2 的 1+ GB 历史 KV,要先释放;
2. **塞 prompt**:req4 的 prompt 是 800 个 token,得先跑一次 shape 为 `[1, 800, d]` 的 prefill 把它的 KV 灌进去——这和当前 decode 的 `[B, 1, d]` 完全不是一个 kernel;
3. **重录 graph**:整个 forward 的 CUDA Graph(如果开了)要重新 capture,一次 50–200 ms。

朴素栈做不到这三件事,**所以只能用最笨的策略:让 batch 跑到所有请求都 EOS,再开下一个 batch**。这才是「槽位被已死请求占着」的底层原因——不是不想放,是 tensor shape 和 kernel 启动模型不让你放。

**两种 batching 的本质区别**(一张对照表):

| | static batching | continuous batching |
|---|---|---|
| 类比 | 拼车 | 出租车队 |
| 规则 | 所有人到终点才能停 | 谁到了谁下,司机立刻接新客 |
| 调度粒度 | 整个 batch 跑完 | 每个 token step |
| 主要代价 | 利用率低 | 工程复杂(要支持变长 attention + 动态 batch) |
| 对 kernel 要求 | 一个 shape 跑到底 | 每 iter 重组 batch + variable-length attention |

§2 的 Continuous Batching「难」,难就难在要让 §2.3 的 Selective Batching、§3 的 PagedAttention、§7 的 piecewise CUDA Graph 同时把上面三条约束**全部**解开。

#### 1.2.2 尾巴空转的另一面：padding token 真的在烧算力

ASCII 图里那个「空转」还藏了一层更隐蔽的浪费——**已经 EOS 的 slot,padding 也照样跑 forward**:

```
iter 50, batch=4, 实际状态:
  slot 0 (req1): 真实 token, position=50    ← 真有用
  slot 1 (req2): <PAD>      (iter 30 就 EOS 了, batch 还在跑)
  slot 2 (req3): <PAD>      (iter 12 就 EOS 了)
  slot 3 (req4): 真实 token, position=50    ← 真有用

forward 算的依旧是 [B=4, ...] 的 batched GEMM:
  ├─ slot 1 / slot 2 的 Linear / Attention 也照常算了
  ├─ 输出虽然被 attention_mask 抹成 0, 不影响采样
  └─ 但 FLOPs 是物理上真的执行了, 不是「跳过」
```

**量化一下**:H100 上跑 8B 模型 batch=64 的单步 decode,大约消耗 1.0 TFLOPs 计算 + 16 GB KV 读取。如果 64 个 slot 里有 40 个已经 EOS,**这 40 个 slot 的 0.6 TFLOPs 算力和 10 GB 带宽就是干烧的**——GPU 不知道、也没办法跳过哪些 slot 的输出会被丢弃。

下面的数学公式 $U = \sum L_i / (B \cdot \max L_i)$ 正是把这种「padding 也算 FLOPs、只有真 token 才算有用功」量化出来的形式化表达。

#### 1.2.3 数学化

**数学化**：设 batch 大小 $B$，请求 $i$ 的输出长度 $L_i$，则总 wall time 是 $\max_i L_i$，但只有 $\sum_i L_i$ 个 token 是「有用算力」。GPU 有效利用率：

$$
U = \frac{\sum_{i} L_i}{B \cdot \max_i L_i}
$$

在 LLM 生产里输出长度分布是**重尾**的（少量长输出占主导）。$\max L_i / \overline{L} \approx 5$–$10$，所以 static batching 的 $U$ 常常只有 **10–20%**。这是症状一。

→ **解法**：每个 token step 都重组 batch，让结束的请求立刻退出、新请求立刻进入。这就是 §2 的 **Continuous Batching**。

### 1.3 症状二：KV cache 按 max_seq_len 预分配，导致三种碎片

朴素实现给每个请求预留 `max_new_tokens` 的连续 KV 显存（这是为了让 attention kernel 能按连续 tensor 写入）。一个 4K-out 的 slot 即使最终只生成 100 token，4K 的位置都被占着。把碎片拆细：

```
内部碎片 (internal):   预留 4K 实际写 100，浪费 3.9K              ← 单请求内
外部碎片 (external):   slot 4K, 4K, 4K 之间的 gap 没法给小请求    ← 请求之间
预留碎片 (reserved):   还没开始用、但已经 commit 的 slot          ← 调度边界
```

vLLM 论文（SOSP 2023）测得朴素实现的「**实际 KV / 总 KV 显存**」只有 **20–40%**——意思是 60–80% 的 KV 显存被白白挂在那里。结果：能塞进 GPU 的并发请求数少，throughput 直接被显存上限掐住。

→ **解法**：把 KV cache 切成定长 block（默认 16 token / block），用页表把逻辑位置映射到物理 block，按需分配。就是 §3 的 **PagedAttention**。

### 1.4 症状三：prefill 与 decode 强耦合在一个 forward pass

prefill 阶段算 $N$ 个 token 的 attention（compute-bound, 接近 roofline 上限），decode 阶段每步只算 1 个 token（memory-bound, 受限于显存带宽）。两者的最优 batch 配置**完全不同**：

| 阶段 | bottleneck | 单 batch 内 token 数 | batch size 偏好 |
|---|---|---|---|
| prefill | compute（matmul） | 上千 | 小 batch / 单请求即可饱和 |
| decode | memory（KV 读） | 1 / req | 大 batch 才能均摊 KV 带宽 |

朴素调度把它们塞到同一次 forward：要么先做完一个长 prompt 的 prefill 再让别人 decode（后排请求 TTFT 被拖到秒级），要么 decode 一步等所有 prefill 完成（GPU 空转）。

→ **解法**：把长 prefill 切成定长 chunk，每个 chunk 和正在 decode 的请求一起进 forward，按 token budget 控制每步总算力。就是 §6 的 **Chunked Prefill**。

### 1.5 三个症状 → 三个根因 → 三个解法

```
症状                根因                       解法                目标指标
静态 batch 空转  → 调度粒度 = request    → Continuous Batching → throughput ↑
KV 碎片         → 显存粒度 = max_len    → PagedAttention      → 并发数 ↑
prefill 抢 GPU  → 异质算力混在一个 fwd  → Chunked Prefill     → TBT 稳定 ↓
                                          (+ Prefix Cache)    → TTFT ↓
                                          (+ CUDA Graph)      → decode kernel overhead ↓
```

5 个支柱里前 3 个是必备（throughput 翻 10×），后 2 个是 latency 长尾必备（命中率高的 workload 再翻 2–6×）。剩下章节逐个展开。

---

## 二、Continuous Batching（迭代级调度）的本源：Orca

> 学术名：**iteration-level scheduling**（OSDI 2022, Orca）。又叫 **in-flight batching**（NVIDIA 命名）、**continuous batching**（HuggingFace TGI 命名）——同一件事。

### 2.1 直观对比

```
Static batching:
  iter 1:  [req1, req2, req3]
  iter 2:  [req1, req2, req3]
  iter 3:  [req1, ----, req3]    ← req2 结束, 槽位空着到本 batch 结束
  iter 4:  [req1, ----, ----]
  ...
  iter N:  [req1, ----, ----]    ← 最坏情况, GPU 只在为一个请求算

Continuous batching:
  iter 1:  [req1, req2, req3]
  iter 2:  [req1, req2, req3]
  iter 3:  [req1, req4, req3]    ← req2 结束, req4 立即插入
  iter 4:  [req1, req4, req5]
  ...                              GPU 永远在 max_batch 上算
```

**每个 token step 都做一次调度决策**，结束的请求立刻退出、等待队列里的新请求立刻填入。

### 2.2 工程难点 #1：变长 batch 怎么进 attention kernel

iteration-level scheduling 要求把不同进度的请求拼到同一个 forward。问题：

- 请求 A 在第 17 个 decode step，KV 长度 = prompt + 17
- 请求 B 是新到的，第 1 个 decode step，KV 长度 = prompt
- 请求 C 是新到的，prefill 阶段，输入 800 token

非 attention 的算子（Linear / GeLU / LayerNorm）对 token 位置无关，**只看 token 总数**——把所有请求的 active token 拼成一条 `[total_tokens, d]` 的张量就能 batched matmul。但 attention 算子里 Q/K/V 的形状依赖每个请求各自的长度，无法直接堆成一个 4D tensor。

### 2.3 Orca 的解法：Selective Batching

Orca 论文给出的妙招：**把 Attention 单独从 batched 路径里抠出来**。

```
batched 路径（token-wise stacking）:
  Linear(QKV proj)  ──┐
  GeLU              ──┤  所有请求的 token 拼成 [T_total, d]
  Linear(out proj)  ──┘

per-request 路径（sequential / variable-length）:
  Attention(Q_i, K_i, V_i)  for each request i
```

- 非 Attention 算子：把 batch 内所有请求的 active token 拼成一条长 tensor，一次大 matmul，**batch 起来效率最高**；
- Attention 算子：每个请求单独算（早期 Orca 是 sequential 循环；现代用 variable-length / paged FlashAttention kernel 在一个 GPU launch 里并行处理不同长度）。

这是 continuous batching 能在 transformer 上跑起来的关键工程前提。**所以「continuous batching」实际上是「selective batching + iteration-level scheduling」的组合**——只是后来大家口语化合并了。

### 2.4 收益与边界

- **静态 → 连续**：Orca 论文报告吞吐提升约 **6.7×**（同等延迟下）；vLLM/TGI 复测在不同 workload 下 **3–10×**。
- **必要前提**：你要有一个能处理变长 Q/K/V 的 attention kernel——这是 PagedAttention / FlashAttention v2 / FlashInfer 的共同入口。
- **不变的事情**：continuous batching 不解决「KV 显存装不下」问题，所以 batch size 仍受 KV pool 容量限制。这就为 §3 的 PagedAttention 留下了大坑。

### 2.5 演化时间线

```
2022.07  Orca (OSDI)            — iteration-level + selective batching 提出
2023.05  HF TGI                  — 开源 continuous batching, 但仍连续 KV
2023.06  vLLM v0.1               — continuous batching + PagedAttention 一并落地
2023.09  vLLM SOSP 论文           — 把范式正式化
2024.01  SGLang                  — continuous batching + RadixAttention
2024.10  vLLM V1 alpha           — 调度器重写, 进一步降低 Python 调度开销
2025.01  vLLM V1 默认             — unified scheduler, prefill/decode 一视同仁
```

---

## 三、PagedAttention：把 OS 虚拟内存搬进 KV cache

### 3.1 OS 虚拟内存的类比

操作系统让进程「看到一段连续的虚拟地址空间」，但物理内存是按页（4KB）分散分配的。这样：

- 多个进程共用物理内存池，按需 demand-page；
- 没有「为一个进程预留 4GB 连续物理内存」的浪费；
- fork() 子进程时父子共享 page，copy-on-write 只在写入时复制。

**KV cache 几乎是同构问题**：
- 「进程」= 一条请求的序列
- 「虚拟地址」= token 位置 0, 1, 2, …
- 「物理内存」= 显存里的 KV pool

PagedAttention 直接照搬：把 KV 切成定长 **block**（默认 16 token），用 **block table** 把每个请求的 token 位置映射到任意物理 block。

### 3.2 数据结构

```
全局 KV pool:
  shape = [num_blocks, num_layers, 2 (K/V), block_size, num_kv_heads, head_dim]
  num_blocks ≈ free_gpu_mem / per_block_bytes

每个请求维护:
  block_table = [phys_block_id_0, phys_block_id_1, ...]
                ↑ 逻辑 block 索引                   ↑ 指向 KV pool

block_manager:
  free_block_pool: List[block_id]           # 当前空闲的 block
  allocated_count[block_id]: int            # 引用计数（被几条 sequence 用着）
  hash_to_block[hash]: block_id             # APC 的 hash → 物理 block 映射
```

举例：block_size = 16，一个请求当前序列长度 50，则它需要 `ceil(50/16) = 4` 个 block，block_table 长度为 4，每个槽是一个物理 block id。

### 3.3 块大小为什么是 16

vLLM 默认 block_size = 16，这是 GPU 算子效率与内部碎片之间的折中：

| block_size | 内部碎片浪费 | attention kernel 效率 |
|---|---|---|
| 4 | 平均浪费 2 token / 块 | kernel 每次只能 gather 4 token，访存效率低 |
| 16 | 平均浪费 8 token / 块 | tile 大小匹配 warp（32）/ tensor core (16) 节奏 |
| 64 | 平均浪费 32 token / 块 | 跨请求复用粒度太粗，共享前缀失效 |
| 256 | 平均浪费 128 token / 块 | 退化为接近 static batching |

**实测**：block_size ≥ 16 throughput 相近，block_size = 16 比 block_size = 8 在 batch=64 时吞吐高约 **1.27×**（vLLM 论文）。SGLang 默认 block_size = 1 但配合 radix tree（更细粒度的复用），代价是 attention kernel 要做更多 gather；FlashInfer 内核 (§10.4) 直接支持任意 block_size。

### 3.4 Paged FlashAttention：kernel 怎么改

标准 FlashAttention 假设 K/V 在显存里是 `[seq_len, num_heads, head_dim]` 的连续 tensor。Paged 版本要按 block table 间接寻址：

```
for each (request, head, query_tile):
    # 取出该请求的 block 列表
    blocks = block_table[request_id]
    # 把 K/V 分块加载到 SRAM 做 FlashAttention tiling
    for blk in blocks:
        K_tile = KV_pool[blk, layer, K]   # gather
        V_tile = KV_pool[blk, layer, V]
        accumulate FlashAttention(Q_tile, K_tile, V_tile)
```

关键 trick：**block 内 KV 仍然连续**，所以一个 block 内的 attention 仍然是连续访存；跨 block 才需要 gather。FlashAttention 的 tile 大小通常是 block 的整数倍，性能损失 < 5%。这就是为什么「分页」几乎是免费的。

paged FlashAttention 现在由 **FlashInfer**（MLSys 2025 best paper）统一提供 kernel；vLLM、SGLang、TRT-LLM 都在切到这套统一引擎。

### 3.5 写入路径（append 一个新 token）

```
decode step on request r:
  L = current sequence length of r
  blk_idx = L // block_size
  offset  = L % block_size

  if offset == 0:
      # 需要新分配一个 block
      new_blk = block_manager.allocate()  # 从 free pool 取
      block_table[r].append(new_blk)
  else:
      new_blk = block_table[r][blk_idx]

  KV_pool[new_blk, :, :, offset, :, :] = new_K, new_V
```

每 16 个 token 才发生一次「分配 block」操作，绝大多数 step 只是往现有 block 里塞数据，**O(1) 复杂度**。

### 3.6 共享路径：引用计数 + Copy-on-Write

多条 sequence 共享前缀（beam search、parallel sampling、prefix cache、多轮对话）时，PagedAttention 用引用计数处理：

```
beam search, beam=4 起步:
  4 条 beam 都指向同一组 prefix block
  block_manager.refcount[blk] = 4

某一步 beam i 选了不同 token, 要写入 block b:
  if refcount[b] > 1:
      new_blk = allocate()
      copy_block(b → new_blk)         # Copy-on-Write
      block_table[i][...] = new_blk
      refcount[b] -= 1
  else:
      write in place
```

这是 PagedAttention 比静态 KV 强大的关键：**逻辑共享在 page table 层面实现，物理上不复制，直到真正写入时才 CoW**。这条机制是 §4 prefix cache 和 §5 RadixAttention 的物理基础。

### 3.7 收益数字

| 指标 | Naive | PagedAttention |
|---|---|---|
| KV 显存利用率 | 20–40% | 90–96% |
| 同等显存下的 max batch | $B$ | 2–4 $B$ |
| 同 workload 吞吐（叠加 continuous batching） | 1× | 14–24× |

### 3.8 代价与替代方案：vAttention

PagedAttention 不是免费的——它要求**所有 attention kernel 都支持 paging**。每个新算子（推测解码 verifier、MLA 解码、sliding window 等）都要单独适配，工程量大。

[**vAttention** (ASPLOS 2025, Microsoft)](https://arxiv.org/abs/2405.04437) 给出另一条路：用 **CUDA 虚拟内存 API**（cuMemAddressReserve / cuMemMap）让操作系统级别的虚拟内存帮你「连续地址 + 按需物理 page」。kernel 仍然看到连续 KV tensor，不需要任何改动，OS 在缺页时分配物理页。

| 维度 | PagedAttention | vAttention |
|---|---|---|
| 虚拟连续性 | 否（page table） | 是（CUDA VM） |
| Kernel 改动 | 每个 kernel 都要适配 | 不需要 |
| 实测吞吐 | baseline | 比 paged FlashAttention 高 **1.23×**，比 paged FlashInfer 高 **1.45×** |
| 工程复杂度 | 调度器管 block table | 调度器管 VA 段 |
| 生态 | vLLM/SGLang/TRT-LLM 全在用 | 论文工作，少量生产用 |

预测：长期看 vAttention 路线（不动 kernel）会胜出；短期仍是 PagedAttention 的天下，因为生态先发优势太大。这一节我们仍以 PagedAttention 为主线叙述。

---

## 四、Prefix Cache / Automatic Prefix Caching（APC）

### 4.1 动机：工业 workload 的前缀重复率

LLM 推理的 prompt 绝大多数情况下是**结构化拼接**的：

```
[system prompt][few-shot 例子][工具定义][历史对话...][本轮 query]
   常量            常量          常量        增量          变量
```

实测重复率（vLLM 团队 + 公司侧统计）：

| Workload | 典型前缀长度 | 用户之间重复率 |
|---|---|---|
| 单轮 chatbot（system prompt 共用） | 200–2000 token | > 80% |
| Few-shot 评测 | 1K–8K | ≈ 100% |
| Code agent（system + tool description 重复） | 3K–10K | 80–95% |
| 多轮对话（同一 conversation 内） | 随轮次增长 | 100%（同一会话） |
| RAG（检索内容不同，前缀仍共享） | 400–1500 | 60–80% |

prefill 的算力开销正比于 prompt 长度。**把共享前缀算一次复用** = 直接抹掉这部分 prefill FLOPs，TTFT 显著下降。

### 4.2 vLLM APC：hash 链 + 物理 block 复用

PagedAttention 天然支持引用计数，所以 prefix cache 的硬件层已经准备好。剩下的问题是：**「给一段 token 序列，怎么快速找到它对应的物理 block」**。

vLLM 的 APC 算法：

```
对一条请求的 prompt = [t_0, t_1, ..., t_{N-1}]:
  分块: block_0 = t_0..t_15, block_1 = t_16..t_31, ...

  每个 block 的 hash 不仅基于本块内容, 还包含「所有在前的 token」:
    hash(block_k) = H( hash(block_{k-1}) || tokens_in_block_k )

  在 global hash table 里查 hash(block_k):
    命中 → 复用该物理 block, refcount += 1, 跳过本块 prefill
    未命中 → 算 prefill, 把结果存进新 block, hash_table[hash] = new_blk
```

**为什么 hash 要包含前缀**：保证「同样的最长前缀才能匹配」。例如两个请求都以「You are helpful」开头，但第二个请求第 17 个 token 后开始说「Translate」、第三个说「Summarize」——只有它们的 block_0（前 16 个 token）共享，block_1 必须各算各的。这种「**chain hash**」结构等价于在 token 序列上做了一棵隐式的 trie。

可选用 **SHA-256** 防碰撞（vLLM 提供 flag）。默认用更快的 hash + 长度校验，碰撞概率 << 1/万亿。

### 4.3 命中率与 KV 显存的权衡曲线

APC 占用 KV pool，会和「正在 active 的请求 KV」抢资源。曲线大致是：

```
命中率 ↑
   |     ╭───── (饱和)
   |   ╱
   | ╱
   |╱
   +────────────→ 留给 cache 的 KV 显存比例

吞吐 ↑
   |    ╱╲
   |   ╱  ╲    ← 留太多 cache 反而 active batch 不够
   |  ╱    ╲
   | ╱      
   +────────────→
```

工程经验：留 **30–50%** KV pool 给 prefix cache，剩余给 active 请求；vLLM 用 LRU 自动平衡。命中率高的 workload（chat / agent）TTFT 降 70%+，命中率低的（每条请求 prompt 都完全不同）几乎无收益、有少量 hash 开销。

### 4.4 失效与共享的统一：引用计数

```
请求到达:
  prefill 前 hash 检查
  命中 → block_table 直接指过去, refcount++
  未命中 → 分配 + 写 + 注册 hash

请求结束:
  block_table 上每个 block refcount--
  refcount == 0 时 block 入 free pool (LRU 队列)

需要分配新 block 时:
  free pool 不空 → 直接拿
  free pool 空了 → 从 LRU 队列尾部驱逐一个 cache block
```

注意：**cache block 不是「专门一块区域」**，它和 active KV block 共用同一个物理 pool，只是 refcount=0 的 block 处于「可被回收但暂时保留」的状态。这避免了静态划分带来的浪费。

### 4.5 多租户的安全：cache salting

危险：用户 A 的 system prompt 里包含敏感信息（系统级 key、私有指令），用户 B 通过精心构造的 prompt 利用 cache hit / miss 的**侧信道**推测 A 的内容（命中 = TTFT 突然降低）。

vLLM 引入 **cache salting**（2025）：每个租户分配一个 salt，hash 时混入 salt：

```
hash(block) = H(salt || prev_hash || tokens)
```

不同 salt 之间 hash 永远不会撞——**租户之间物理上无法 cache 复用**，付出的代价是同租户内仍可复用。这是「安全 vs 效率」的标准权衡。

### 4.6 命中 / 未命中两条路径对照

```
未命中 (cold):
  请求到达 → prefill 全长 prompt → 1 token decode → 2 token decode → ...
   TTFT ≈ prefill 时间

命中 90% (warm):
  请求到达 → prefill 仅最后 10% token → 1 token decode → ...
   TTFT ≈ 0.1 × 完整 prefill 时间
   省下的算力可以服务更多 batch → 全局吞吐也涨
```

这就是为什么 SLO 设计要区分「cold TTFT」和「warm TTFT」（§11.2 详述）。

---

## 五、RadixAttention（SGLang）与 Cache-Aware Scheduling

### 5.1 为什么 hash 不够：分支与子前缀

vLLM APC 用「block 粒度的 hash 链」，本质是一棵**隐式 trie**，但操作受限于 block 大小（16 token 对齐）。考虑这种 workload：

```
请求 1: "You are helpful. Translate this: hello"
请求 2: "You are helpful. Translate this: bye"
请求 3: "You are helpful. Summarize this: ..."
```

前 16 个 token 一致 → 共享 block_0。第二个 block 内容不同，但「Translate this:」这个**子串**仍然在请求 1 和 2 之间共享。block 粒度的 hash 表达不了「**block 内的部分共享**」。

SGLang 把这件事做彻底——直接维护一棵显式的 **Radix Tree**（基数树/压缩前缀树），节点存任意长度的 token 序列。

### 5.2 Radix Tree 数据结构

```
Radix Tree:
  root
   └── "You are helpful. " (system prompt, 32 token)
        ├── "Translate this: " (16 token)
        │     ├── "hello"
        │     └── "bye"
        └── "Summarize this: " (16 token)
             ├── "<doc 1>"
             └── "<doc 2>"
```

每个节点存：
- 一段 token 序列（任意长度）
- 对应的 KV block 引用列表
- 子节点指针
- 引用计数 / LRU 时间戳

请求到达时，从 root 开始**最长前缀匹配**（LMP, Longest Matching Prefix）：
- 完整匹配某个节点的全部 token → 沿子节点继续；
- 匹配到节点中间某个位置 → **裂开**该节点（split），上半部分变成新节点（共享）、下半部分变成新节点（独占）；
- 找不到匹配 → 在最深匹配节点下新建子节点。

这种结构能精确表达**任意粒度**的前缀共享。

### 5.3 Cache-Aware Scheduling：让命中先走

vLLM 调度是「FIFO + 资源够就跑」。SGLang 在 Radix 树上做了一步聪明的扩展：**让命中长前缀的请求优先调度**，近似一次**深度优先遍历**：

```
WAITING queue:
  req A: 命中前缀 200 token (system prompt)
  req B: 命中前缀 1500 token (system + few-shot + tool)
  req C: 命中前缀 0

优先级排序:
  B > A > C   ← 按命中长度排
```

为什么这样好：B 跑的时候它命中的所有 block 还在 cache 里（如果先跑 C，B 的命中 block 可能被 C 的 prefill 挤掉）。把命中长的请求**连续**调度，能最大化「物理复用」的时间窗。

但要避免**饥饿**——C 永远命中 0 会被饿死。SGLang 加了 starvation timeout：等待超过 $T_{\max}$ 的请求强制提到队首。

### 5.4 LRU on tree：从叶子到根淘汰

KV 满了要驱逐。Radix 树的驱逐有一个天然顺序：

```
只能从叶子开始驱逐 (refcount == 0 的叶子)
  ↓
叶子被驱逐后, 父节点若也变成无引用叶子, 继续驱逐
  ↓
按 LRU 时间戳挑「最久没用」的叶子
```

这保证了**深的、特化的、可能只服务一个用户的子树先被丢弃，浅的、公共的（system prompt）几乎永远在**。这就是为什么 SGLang 在多用户共用同一 system prompt 的场景命中率能稳定在 80%+。

### 5.5 实测：agent / 多轮场景的优势

在 ReAct 类 agent / multi-turn chat workload 上（system + tool + 历史复用率高）：

- SGLang 比 vLLM 吞吐高 **30–6.4×**（前缀共享越长差距越大）；
- 同模型 H100 上 Llama-3-8B，SGLang ~16,200 tok/s vs vLLM ~12,500 tok/s（PremAI 2025 测试，约 **29%** 领先）；
- 70B+ 模型差距缩到 **3–5%**（KV 总量大，attention 算子本身占的比重高，prefix 优化边际收益减少）；
- 单轮 / 无前缀复用场景两者接近。

### 5.6 RadixAttention vs PagedAttention 的层次关系

这是面试常被搞混的关键点。它们**不是替代关系**：

| 层次 | PagedAttention | RadixAttention |
|---|---|---|
| 解决什么 | KV 物理内存怎么存 | 多请求之间前缀怎么共享 |
| 类比 | 操作系统的虚拟内存 / 页表 | 文件系统的目录树 / inode 共享 |
| 数据结构 | block table + free pool | radix tree on token sequence |
| 是否替代对方 | 否 | 否 |

**事实**：SGLang 内部仍然有一个分页风格的物理 block pool（block_size 可以是 1，但仍然是「**block 池 + 引用**」结构）；RadixAttention 在它上面再加一层**逻辑共享索引**。SGLang = PagedAttention 思想 + Radix Tree 索引 + cache-aware scheduling。

vLLM 也在向 SGLang 学：vLLM 的 APC 哈希链是「弱化版 trie」，从 trie 角度看就是固定 block 粒度的特例。

---

## 六、Chunked Prefill：prefill 与 decode 的和平共处

> 经典工作：[**SARATHI-Serve** (OSDI 2024)](https://arxiv.org/abs/2403.02310)。vLLM、SGLang、TRT-LLM 均默认开启。

### 6.1 prefill 与 decode 的 roofline 错位

```
            Arithmetic Intensity (FLOPs / byte)
             ←───── memory-bound ────│──── compute-bound ─────→
                                     │ ridge ≈ 200 (H100 FP16)
  decode    ●  ~1                    │
            （每步算 1 个 token,      │
             读 KV 大量 byte）        │
                                     │
  prefill                            │           ●  ~2000
                                     │           （N 个 token 共享 KV 读,
                                     │            算力被打满）
```

- **decode 是 memory-bound**：每步只算 1 token，主要时间花在「读 KV」上。增大 batch 能均摊带宽（每步多 token 但 KV 只读一次）→ 吞吐随 batch 线性增长直到带宽饱和。
- **prefill 是 compute-bound**：1 个长 prompt 自己就能打满算力。增大 batch 收益小，反而抢 decode 的位置。

混在一个 forward 时矛盾：要么 prefill 主导（GPU 在长长地算 prefill，decode 请求 P99 TBT 飙升），要么 decode 主导（凑大 batch 等所有人，prefill 被挤后排，TTFT 飙升）。

### 6.2 朴素 vLLM v0：先 prefill 后 decode 的 TBT 抖动

vLLM v0 默认调度（pre-Sarathi）：**优先做 prefill**，prefill 期间 decode 暂停。

```
t→
  ──────────────────────────────────────────────
   prefill(8K prompt)     decode  decode  decode
   ████████████████████   ░       ░       ░
                          ↑
                          所有正在 decode 的请求在这 8K prefill 期间冻结
```

后果：长 prompt 一来，所有 active 请求 TBT 出现一个 **2–10 秒**的尖峰，用户感觉「卡了一下」。即使 P50 TBT 是 20ms，P99 也会到秒级。

### 6.3 Sarathi-Serve：chunked prefill + stall-free batching

核心想法：**把长 prefill 切成定长 chunk，每个 chunk 和 decode 拼在同一个 forward**。

```
token budget T = 2048 (A100) or 8192 (H100)

iter k 的 forward 内容:
  decode 部分: R 条 active 请求 × 1 token = R tokens
  prefill 部分: 长 prompt 的一个 chunk = T - R tokens

要求: R + chunk_len ≤ T (token budget 守恒)
```

每个 iteration 都把「正在 decode 的人」+「正在 chunked-prefill 的人」拼成同一次 forward，按 token budget 算。这样：

- **decode 永远不停**：每个 iteration 都给 active 请求出 1 个 token，TBT 抖动消除；
- **prefill 不再独占**：长 prompt 被切成 N 个 chunk 跨多个 iter 处理，TTFT 增加一点点（多了几 ms 的调度开销）；
- **batch utilization 高**：每个 iter 的 token 数稳定在 budget 附近。

### 6.4 Token budget 怎么选

token budget T 决定每个 iter 的总算力上限：

```
T 太小:
  - chunk 太碎, prefill 进度慢, TTFT 长尾差
  - prefill 阶段不能打满算力 (compute-bound 需要大 chunk)

T 太大:
  - 单次 forward 时间长, decode 的 TBT 也跟着长
  - 失去 chunked 的稳定性优势
```

实测的 ridge point（饱和算力的最小 chunk 大小）：

| GPU | 推荐 token budget |
|---|---|
| A100-80G | ~2048（vLLM 默认） |
| H100-80G | ~8192 |
| H200 | 8192–16384 |
| MI300X | ~4096 |

调优口诀：**调高 T 直到 prefill 算力打满；再调低 T 直到 P99 TBT 满足 SLO**。两者中间通常有一个 sweet spot。

### 6.5 vLLM V1 的进一步统一

V1（2025.01 默认）做了一个更激进的设计：**调度器不再区分 prefill 和 decode**——只看「请求 r 需要算多少 token」。一个请求的 prefill 阶段每次摊 chunk_len 个 token，decode 阶段每次摊 1 个，**都进同一个调度字典**：

```python
scheduled = {req_id: num_tokens_to_compute_this_iter}
total = sum(scheduled.values()) ≤ T
```

这种「token 平等」视角让 chunked prefill 成为默认行为，不需要单独开关。

### 6.6 与 PD 分离（§12.4）的关系

- **Chunked prefill**：单节点上 prefill / decode 在**时间维度**交错——同一组 GPU 在同一秒里既做 prefill chunk 也做 decode；
- **PD 分离 (Splitwise / DistServe / Mooncake)**：把它们**空间分离**到不同节点——prefill 节点专门做 prefill，decode 节点专门做 decode，KV 通过 RDMA 传过来。

| 维度 | Chunked Prefill | PD 分离 |
|---|---|---|
| 部署成本 | 0（单机即可） | 需要多机集群 + KV 传输 |
| 适合规模 | 小到中（< 100 GPU） | 大（数百到数千 GPU） |
| 适合 workload | TTFT/TBT 都重要 | prefill / decode 异构负载 |
| 关系 | 同一 GPU 时分复用 | 分到专门 GPU 上 |

短期单机用 chunked prefill 就够；长期大集群转 PD 分离。两者**目的相同**（解决 prefill-decode 冲突）、**手段不同**、**可以叠加**（PD 分离里 decode 节点上仍可能开 chunked prefill 处理 partial KV transfer 后的尾部）。

---

## 七、CUDA Graph：吃掉 decode 阶段的 launch overhead

### 7.1 病灶：每个 kernel launch ~5–10 μs

decode 一步的 forward 里有数百个 CUDA kernel（embedding lookup、各种 matmul、layernorm、attention、softmax……）。每个 kernel launch 有固定 host-side 开销：

- CPU 把 kernel 参数序列化推到 stream queue
- driver 在 GPU 端建调度上下文
- 中间还有 cudaMalloc / synchronize 等点位

每次 **5–10 μs**。一次 decode 步 200 个 kernel → 1–2 ms 纯 launch overhead。当 decode 本身 kernel 时间也是 5–15 ms 时，**launch 占 10–30%**——不是噪声，是大头。

### 7.2 CUDA Graph：把整次 forward 录成静态图

CUDA Graph 是 NVIDIA 提供的「**录制一次 → 重复 replay**」机制：

```
warmup:
  cudaStreamBeginCapture()
  执行一次完整 forward            ← 拓扑被录下来
  cudaStreamEndCapture() → graph

runtime:
  graph.replay()                  ← 一次 host call 启动整条 graph
```

收益：N 次 launch → 1 次 launch。decode 步的纯 launch overhead 从 1–2 ms 降到几十 μs。**小 batch / 短 decode 步的相对收益最大**。

### 7.3 静态 shape 的代价：batch 离散化

CUDA Graph 要求每次 replay 的 tensor shape 完全一致——而 serving 系统每步的 active batch size 是动态的。解法是**离散化**：

```
启动时预录:
  graph_1  = forward(batch_size=1)
  graph_2  = forward(batch_size=2)
  graph_4  = forward(batch_size=4)
  graph_8  = forward(batch_size=8)
  graph_16 = ...
  ...
  graph_128 = forward(batch_size=128)

runtime:
  actual_bs = 13
  padded_bs = round_up_to_bucket(13) = 16     # 浪费 3 个 slot
  graph_16.replay()
```

代价：少量算力浪费（padding 部分），但相比 launch 开销的节省非常划算。vLLM 默认录 batch ∈ {1, 2, 4, 8, ..., 256}。

### 7.4 vLLM V1 的 piecewise CUDA Graph

更激进的方案是 **piecewise capture**（vLLM V1, 2025）：

```
完整 forward 被切成多段:
  [非 attention 段] → [attention] → [非 attention 段] → [attention] → ...
       ↑ 这些段录成 graph        ↑ 这些段动态执行
```

非 attention 部分（QKV / O / MLP / norm）token-wise 容易 graph，shape 只依赖 token 总数；attention 部分（变长 KV、paged gather）shape 复杂，**不录 graph**，正常 launch。

```
piecewise:  ▣▣▣ ─ A ─ ▣▣▣ ─ A ─ ▣▣▣ ─ ...
            graph 段        graph 段       graph 段
```

收益：非 attention 段（占 60–70% 总 kernel 数）的 launch 开销被消除；attention 段保留灵活性。这是 vLLM V1 throughput 提升的重要来源之一。配合 **torch.compile** 把多个小 kernel fuse，效果再叠加一层。

### 7.5 与推测解码的冲突

推测解码（§12.3）每步实际接受的 token 数是动态的 **1 到 k+1**——同样违反 CUDA Graph 静态 shape 假设。常见解法：

- 为每个可能的接受数 m ∈ {1, 2, ..., k+1} 录一个 graph；
- 或用 conditional graph node（CUDA 12+ API）；
- 或干脆给推测解码 forward 关掉 graph，单纯 launch（接受 token 多时收益已经够大）。

TRT-LLM 在 graph + 推测解码的工程化上最成熟，vLLM 在追赶。

---

## 八、Scheduler 的全局视角

前面 §2–§7 把一个个能力拆开讲，这一节把它们整合在调度器视角下——**每个 iteration 调度器实际做什么决策**。

### 8.1 双队列结构

```
WAITING queue (FCFS / priority):
  └── 新到的请求，尚未占用 KV pool

RUNNING queue:
  └── 已有 KV block，正在 decode 或 chunked-prefill
       │
       └── 子状态: ACTIVE (本 iter 要算)
                  SWAPPED (KV 在 CPU)
                  PREEMPTED (KV 已丢弃, 等重做)
```

vLLM V1 把这两个队列合一为「unified scheduler」——所有请求都进同一个字典 `{req_id: num_tokens_this_iter}`，prefill 中的请求每步推进 chunk_len 个 token、decode 中的每步推进 1 个，**统一按 token budget 切**。语义更干净，也更适合 chunked prefill 默认开启。

### 8.2 每个 iteration 的决策流程

```
def schedule_one_iter():
    scheduled = {}        # req_id -> num tokens
    token_budget = T

    # 1) 决定 RUNNING 队列里哪些请求本步能继续 decode
    for req in running:
        need_tokens = 1   # decode step
        if blocks_available_for_next_decode(req):
            scheduled[req.id] = need_tokens
            token_budget -= 1
        else:
            # KV pool 满了, 必须抢占
            preempt(req, mode=RECOMPUTE or SWAP)

    # 2) 剩余 budget 喂 WAITING 队列里的新请求 (做 chunked prefill)
    while token_budget > 0 and waiting:
        req = waiting.pop_priority()    # SGLang: 最长前缀; vLLM: FCFS
        prefix_hit = lookup_prefix_cache(req.prompt)
        remaining = len(req.prompt) - prefix_hit
        chunk = min(remaining, token_budget, chunk_max)
        if allocate_block_for_chunk(req, chunk):
            scheduled[req.id] = chunk
            token_budget -= chunk

    # 3) 拼装本次 forward
    return scheduled
```

每一步都是一次「**预算分配**」：先保证 decode 推进（用户体验、TBT 稳定）、再用余量喂 prefill（TTFT 降低）。这就是「**stall-free batching**」的本质。

### 8.3 抢占：swap vs recompute

KV pool 满了又要分配新 block，必须从已 active 的请求里腾位置。两种策略：

| 策略 | 做什么 | 恢复代价 | 适用 |
|---|---|---|---|
| **Swap-out** | 把整个 sequence 的 KV 拷到 CPU memory | 拷回 KV 经 PCIe，恢复成本 ≈ prefill 的 1/3–1/2 | 长 prompt，重做 prefill 太贵 |
| **Recompute** | 丢弃 KV，下次重新 prefill 整个 prompt | 完整 prefill 时间 | 短 prompt，prefill 比 PCIe 还快 |

**vLLM v0** 默认混用：长 prompt → swap、短 prompt → recompute。

**vLLM V1（2025）的变化**：默认 **RECOMPUTE**。理由：

- V1 用统一调度器后 swap 路径需要额外的 CPU 端 KV 管理，复杂度高；
- chunked prefill 让重做 prefill 不再是「整块独占 GPU」——可以摊到多个 iter 里，对 TBT 影响小；
- 现代 GPU 的 prefill 速度（H100 上长 prompt ~ 10K tok/s）已经接近 PCIe gen5 的有效带宽，swap 的相对优势缩水。

**抢占对 SLO 的影响**：被抢请求的 TPOT 会出现毛刺（从 20 ms 跳到 prefill 重做时间，可能几百 ms）。压测时要关注：

- 抢占率 = 被抢请求数 / 总请求数
- 抢占后恢复 TTFT 二次值（recompute 模式下接近一次完整 TTFT）

缓解：留 KV pool reserve（不让占满）、降低 max concurrent、用 PD 分离把 decode 隔离开。

### 8.4 公平性、优先级、SLO-aware 调度

朴素 FIFO 在多租户 / 多 SLO 场景不够用：

- **租户公平**：一个用户的大并发不能挤垮另一个用户；
- **优先级**：付费高级用户的 TTFT SLO 严苛；
- **SLO-aware**：deadline 临近的请求要插队。

SGLang / TRT-LLM 引入 priority-aware scheduler，常见策略：

| 策略 | 思路 |
|---|---|
| Weighted round-robin | 每个租户分一个权重，按比例发请求 |
| EDF (Earliest Deadline First) | 按 TTFT/TPOT 剩余时间排 |
| LMP-aware（SGLang）| 命中长前缀的先调度，配合 starvation timeout |
| Goodput 优化（DistServe）| 调度目标是「满足 SLO 的请求数 / 秒」，而非 raw throughput |

### 8.5 单节点 → 多副本路由（为 §12.4 铺垫）

到了多副本（同模型在多张卡 / 多机），调度器还要选「这个请求该路由到哪个副本」：

- **Round-robin**：最简单，但忽略 prefix cache 局部性；
- **Hash-based**：按 user_id / session_id hash 到副本，多轮对话天然落同一副本 → 命中率高；
- **Cache-aware**：查每个副本的 prefix tree，选最长命中的副本（SGLang router 实现）；
- **Least-loaded**：选负载最低的副本（牺牲命中率换均衡）。

进一步演化是 §12.4 的 **PD 分离 + KV-centric 路由**（Mooncake）——把 prefill 节点和 decode 节点分开调度，按 KV cache 位置决策。

---

## 九、把开关一个个打开：累计效果对照

这是一张「从朴素到 SOTA」的能力开关表，展示每个支柱的边际贡献。设定：Llama-3-8B / H100-80G / 200 并发 / 输入 512 / 输出 256。

| # | 启用的能力 | Throughput | TTFT P50 | TPOT P50 | KV 利用率 | 备注 |
|---|---|---:|---:|---:|---:|---|
| 0 | 朴素 HF generate, static batch=4 | 600 | 4.8 s | 95 ms | 12% | 基线 |
| 1 | + Continuous Batching | 1,800 | 3.2 s | 38 ms | 28% | 3× |
| 2 | + PagedAttention | 7,500 | 1.4 s | 22 ms | 92% | KV 翻 3× → batch 翻 3× |
| 3 | + Prefix Cache（80% hit） | 11,000 | 0.4 s | 22 ms | 90% | TTFT 大降 |
| 4 | + Chunked Prefill | 12,500 | 0.5 s | 19 ms | 90% | TTFT 略升 / TBT 长尾大降 |
| 5 | + CUDA Graph | 14,000 | 0.4 s | 17 ms | 91% | decode launch 削平 |
| 6 | + RadixAttention（agent workload） | 17,000 | 0.3 s | 17 ms | 92% | 仅在前缀复用高的 workload 有效 |
| 7 | + 推测解码 + 量化（§12.3 / §12.1） | 25,000+ | 0.3 s | 8 ms | 92% | 进入 §12.3 |

要点：

- **continuous batching + PagedAttention** 贡献了大头（从 600 → 7500，约 12×）；
- prefix cache 主要降 TTFT 而非吞吐（除非 prefill 是瓶颈）；
- chunked prefill 是 **长尾稳定性** 的关键，对 P50 影响小、对 P99 改善巨大；
- CUDA Graph 的相对收益取决于 decode kernel 时间占 launch 的比例（batch 小时收益高，batch 大时收益小）；
- RadixAttention vs Prefix Cache 在简单 hash 命中已经够好的 workload 下差距不大，在前缀树状结构复杂的 agent workload 拉开差距。

把这张表反过来看就是「**性能回归的 debug 顺序**」：吞吐突然掉到 baseline 的某档，对照这张表能定位是哪一档开关失效了。

---

## 十、三大栈对比（深度版）

### 10.1 vLLM（UC Berkeley → 社区主导）

**出身**：2023.06 开源，SOSP 2023 论文，PagedAttention 的原创栈。2025 年是事实上的开源标准。

**架构关键**：
- v0 → V1（2025.01 默认）：调度器从「双队列 + 显式 prefill/decode 区分」改为「unified scheduler + token-budget 调度」；
- piecewise CUDA Graph + torch.compile 集成；
- 推测解码：内置 Eagle / Medusa / N-gram；
- 量化：AWQ / GPTQ / FP8 / NVFP4 全套；
- API：OpenAI-compatible，`vllm serve <model>` 一行启动；
- 多机：tensor / pipeline / expert parallel；PD 分离（disagg）部分支持。

**适合**：
- 通用 chat 服务（吞吐、TTFT 都要）；
- 离线批处理（throughput 最大化）；
- 大模型（70B+）/ MoE / 多机部署；
- 「不想折腾」的团队，社区最大、文档最齐、bug 修最快。

### 10.2 SGLang（LMSYS → 社区增长最快）

**出身**：2024.01 开源，团队来自 LMSYS（Chatbot Arena 背景，重点是 chat / agent）。RadixAttention + 前端 DSL 是其招牌。

**架构关键**：
- RadixAttention：显式 radix tree，cache-aware scheduling；
- 前端 DSL（控制流原语 + 受限生成）：直接表达 agent / 多轮 / tool use；
- 调度上的 Python GIL 优化 + zero-overhead scheduler；
- 推测解码：Eagle / Lookahead；
- KV：paged + radix，block_size 可以小至 1。

**适合**：
- 多轮 chat / agent / ReAct 场景（前缀复用 80%+ 时优势 30%–6×）；
- Structured generation / function calling（DSL 直接支持）；
- 重 prefix 重用的 RAG。

不适合：单轮 / 全异质 prompt / 长 context 但前缀不重叠的离线批跑（与 vLLM 接近，没有显著优势）。

### 10.3 TensorRT-LLM（NVIDIA → 商业 / 硬件天花板）

**出身**：NVIDIA 自家栈，深度绑定 Tensor Core / FP8 / NVFP4 / Hopper-Blackwell 指令集。

**架构关键**：
- in-flight batching（NVIDIA 命名的 continuous batching）+ paged KV cache；
- C++ 实现的 batch manager + Python 配置层；
- 量化最全：FP8（E4M3 / E5M2）、NVFP4、SmoothQuant、AWQ；
- CUDA Graph + 推测解码集成成熟（业界最早跑通的）；
- 配合 Triton inference server + Dynamo（生产部署框架）；
- 2025 H100/H200 benchmark：FP8 下 Llama-3-8B 可达 10K+ tok/s，TTFT < 100 ms。

**适合**：
- 全 NVIDIA 硬件、追求 throughput 天花板；
- 已有 Triton / Dynamo 部署体系；
- 商业 SLA、需要 NVIDIA 直接支持。

不适合：跨硬件（AMD / Intel）、快速迭代、研究探索（C++ 改起来不如 Python 灵活）。

### 10.4 FlashInfer：三栈共用的 Attention Engine

**FlashInfer**（MLSys 2025 best paper）是一个 attention kernel library，**vLLM、SGLang、MLC-LLM、TRT-LLM 都集成它**。它统一了：

- Paged / variable-length / chunked / 推测解码 / sliding window / MLA 各种 attention 变体；
- 支持 block-sparse KV（适配 H2O / SnapKV 等驱逐方案，§8.3）；
- 性能：ITL 比上一代 paged FlashAttention 降 **29–69%**；长 context 推理 latency 降 **28–30%**；
- 上层是统一 API、底层路由到 cuDNN / TRT-LLM / 自实现 kernel。

**含义**：上层栈（vLLM / SGLang / TRT-LLM）的 attention kernel 差距正在被 FlashInfer 抹平，未来三栈的差异会更多体现在**调度器、prefix 管理、生态**上，而不是 attention 速度。

### 10.5 选型决策树

```
你的 workload 主导特征?
├── 通用 chat / 单轮 / 离线批 ──→ vLLM
├── 多轮 chat / agent / RAG (前缀复用高) ──→ SGLang
├── 全 NVIDIA + 需要硬件天花板 + 商业 SLA ──→ TensorRT-LLM
└── 跨硬件 (AMD / Intel) 或边缘 ──→ vLLM (生态最广) 或 MLC-LLM

模型规模?
├── ≤ 8B ──→ 三栈差距明显, SGLang 优势在前缀, TRT-LLM 优势在 FP8 吞吐
├── 70B ──→ 三栈接近, 选熟悉的
└── MoE / 100B+ ──→ vLLM 多机生态 + DeepSeek 系列原生 (FP8 / EP)
```

| 维度 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 核心创新 | PagedAttention | RadixAttention + DSL | NVIDIA 全栈优化 |
| KV 管理 | paged block pool | paged + radix tree | paged (TRT-LLM PA) |
| Prefix cache | APC (hash-based) | RadixAttention | 支持 |
| Continuous batching | ✅ | ✅ | ✅ (in-flight batching) |
| Chunked prefill | ✅ (V1 默认) | ✅ | ✅ |
| CUDA Graph | piecewise (V1) | ✅ | 最成熟 |
| 推测解码 | Eagle / Medusa / N-gram | Eagle / Lookahead | Eagle / Medusa / draft model |
| 量化 | AWQ / GPTQ / FP8 / NVFP4 | AWQ / GPTQ / FP8 | 全套 + NVFP4 ⭐ |
| PD 分离 | 部分（disagg） | 部分 | 是（+ Dynamo） |
| Attention kernel | FlashInfer / FlashAttention | FlashInfer / Triton | TRT 自实现 + FlashInfer |
| 多机并行 | TP / PP / EP / DP 全套 | TP / DP / EP | TP / PP (强) |
| 易用性 | 一行启动 | DSL 灵活但需学习 | C++ 工程化 |
| 生态 | 社区最大 | 增长最快 | NVIDIA 商业支持 |

---

## 十一、关键指标体系

这一节把 serving 系统的指标定义讲严格——很多生产事故来自指标定义不一致。

### 11.1 Throughput 指标

- **Total output throughput (tok/s)**：单位时间内生成的总 output token 数。**重点**：是 output token，不是 prompt token，否则可以靠塞长 prompt 刷分。
- **Total throughput (tok/s)**：input + output 合计，反映 GPU 总算力利用。
- **Request throughput (req/s)**：单位时间完成的请求数。对短输出（如 reranker）更有意义。
- **Goodput (req/s)**：**满足 SLO 的请求数 / 秒**——超时的请求即使完成也不算。生产里 Goodput 比 raw throughput 更重要：你可以把 batch 拉到 1024 让 raw throughput 翻倍，但绝大多数请求 TTFT 都超 SLO，Goodput 反而暴跌。

### 11.2 Latency 指标：严格区分 TTFT / TPOT / ITL / TBT

容易混淆的四个名字。**正确定义**：

```
请求时间线:
  T_arr              ── 请求到达
   ↓
  T_first_tok        ── 第一个 output token 返回
   ↓
  T_2 ── T_3 ── ... ── T_N    ── 后续 output token
   ↓
  T_done             ── 最后一个 token

TTFT (Time To First Token)  = T_first_tok - T_arr
                              覆盖: 排队 + prefill + 首 token decode

TPOT (Time Per Output Token) = (T_done - T_first_tok) / (N - 1)
                              覆盖: 整个 decode 阶段的平均 per-token 时间

ITL = Inter-Token Latency = T_{i+1} - T_i  (per pair)
                              覆盖: 相邻两个 token 之间的实际间隔

TBT (Time Between Tokens) ≈ ITL  (语义同 ITL, 不同社区命名)
```

**关键差别**：

- **TPOT 是平均值**，平滑掉了所有抖动。你可以有 TPOT = 20 ms 但 P99 TBT = 800 ms（被抢占/长 prefill 干扰）——用户依然感觉卡顿。
- **TBT / ITL 是 per-token 测量**，能暴露 stall-free batching 是否真的成立。Sarathi-Serve 的关键贡献就是把 P99 TBT 压下去而不是仅压 P50 TPOT。

**用户体验关联**：

| 指标 | 用户感觉 | 工程目标 |
|---|---|---|
| TTFT 短 | 「点了就有反应」 | < 300 ms 流畅，< 1 s 可接受 |
| TPOT 短 | 「打字流畅」 | < 50 ms / token（≈ 20 tok/s，超过人类阅读速度） |
| P99 TBT 稳 | 「不卡顿」 | < 100 ms（无明显停顿） |

### 11.3 SLO 与 Goodput 设计

工业 SLO 通常是**多个百分位**联合约束：

```
SLO 示例:
  TTFT P95 < 500 ms
  AND TPOT P95 < 50 ms
  AND TBT P99 < 200 ms

Goodput = 满足以上全部约束的请求数 / 秒
```

调优时**先看 Goodput，再看 raw throughput**：

- 若 Goodput 远低于 throughput → SLO 卡在哪个百分位？
- 若 P99 TBT 飙高 → 看抢占率、chunked prefill 是否开、token budget 是否过大；
- 若 TTFT P95 飙高 → 看 prefill 排队、prefix cache 命中率、是否有热点用户。

DistServe 论文 (2024) 给出一个有用的视角：**throughput-Goodput 关系是 Pareto front**。优化 chunked prefill / PD 分离 / 推测解码本质都是在推这条前沿。

### 11.4 资源指标

```
KV cache hit rate (APC / Radix):
  = 命中的 token 数 / 总 prompt token 数
  健康: chat > 70%, agent > 80%, RAG 40–60%

KV pool utilization:
  = active KV blocks / total KV blocks
  健康: 70–90%; 持续 > 95% → 抢占多发

GPU SM occupancy:
  H100 上 decode 大 batch 应该 > 60%, 小 batch < 30% 正常
  小 batch 用 CUDA Graph + 推测解码补救

Preemption rate:
  = 被抢请求 / 总请求
  健康: < 1%; > 5% → KV pool 不足 / 并发过高
```

### 11.5 怎么压测才公平

LLM benchmark 比传统服务复杂。注意这些坑：

| 坑 | 说明 |
|---|---|
| Warmup 不足 | CUDA Graph capture + APC 预热都需要时间，前 10–30 秒数据噪声大 |
| 长度分布不真实 | 全用同长度 prompt → 调度器毫无压力，与真实 workload 偏离 |
| 并发模型错 | open-loop（按 Poisson 到达）vs closed-loop（固定 N 并发，前者更接近用户） |
| 只看 P50 | 用 P95 / P99 / max 才能发现长尾问题 |
| 不区分 cold / warm prefix | warm 命中率 80% 时 TTFT 差异巨大，要分开测 |
| 输出长度作弊 | 控制 `max_tokens` 还要看实际平均 output 长度 |

推荐工具：vLLM 自带 `benchmarks/benchmark_serving.py`、SGLang 自带 `bench_serving.py`、社区有 `llmperf`、`genai-perf`（NVIDIA）等。

---

## 十二、关键问答（深度版）

**Q1：PagedAttention 和 Continuous Batching 是不是同一回事？**

不是,且经常被混淆。Continuous batching 是**调度策略**(每个 token step 重组 batch),PagedAttention 是**内存管理机制**(分页存 KV)。两者正交叠加:

- 你可以单独有 continuous batching 而不分页(早期 TGI 就是),但 KV 碎片会限制 batch 上限;
- 你也可以单独有 PagedAttention 而不连续调度(理论上),但 batch 槽位浪费仍然存在。

**它们真正的关系是「互相赋能」**:PagedAttention 把 KV 利用率从 30% 提到 95%,使 continuous batching 的 batch size 能开到几十上百;反过来 continuous batching 让 PagedAttention 的 block 复用率最大化(请求结束就立刻释放 block)。

---

**Q2:RadixAttention 比 PagedAttention 更先进吗?是替代关系吗?**

不是替代关系,**它们处在不同抽象层**:

| 抽象层 | 解决什么 | 类比 |
|---|---|---|
| PagedAttention | KV 物理内存怎么组织 | OS 虚拟内存 / 页表 |
| RadixAttention | 多请求之间逻辑前缀怎么共享 | 文件系统的目录树 / inode 共享 |

SGLang 内部仍然有「定长 block + 引用计数」的物理 KV 池(本质上就是 PagedAttention 思想);Radix Tree 是在它**上面**多加一层索引,表达「token 序列之间任意粒度的共享」。

vLLM 的 APC(automatic prefix caching)用 block 粒度的 hash 链,本质是「**block 粒度的 trie**」,功能上是 RadixAttention 的弱化版。在前缀共享结构简单的 workload 上两者接近,在 agent / 多分支 workload 上 RadixAttention 拉开差距。

---

**Q3:Chunked prefill 为什么能降低 TTFT 长尾?它不是让 prefill 变得更慢了吗?**

是的,单个长 prompt 的 prefill **总时间**确实增加了一点(多了几次调度开销),但**TTFT 长尾**的来源是「**后排请求被长 prefill 阻塞**」而非「自己 prefill 慢」。

朴素调度下,一个 16K 长 prompt 一旦开始 prefill,会独占 GPU 直至结束——后排小请求 TTFT 被拖到秒级(P99 TTFT 飙升)。chunked prefill 把这个 prefill 切成 N 个 chunk,每个 chunk 留出 token budget 给其他请求的 decode / prefill 一起跑,长 prompt 自己的总时间几乎不变(+5% 左右),但短请求 TTFT 降到 100ms 级。**P50 略升、P99 大降**。

同时它让 decode 永远不停,TBT 抖动也消除——这是 Sarathi-Serve 论文的双重收益。

---

**Q4:CUDA Graph 会不会和推测解码冲突?怎么解?**

会冲突。推测解码每步实际接受的 token 数是动态的(1 ~ k+1),而 CUDA Graph 要求 shape 静态。常见解法:

1. **为每个可能的接受数录一个 graph**:简单粗暴,内存代价大(graph 数 × batch 桶数 × 接受数);
2. **conditional graph node**(CUDA 12+ API):允许图内分支,但生态尚浅;
3. **piecewise capture**:只把非 attention 段录 graph,attention 段动态执行(vLLM V1 的选择);
4. **推测解码关掉 graph**:接受多 token 时 launch overhead 已被均摊,关掉影响小。

TRT-LLM 在这一块工程最成熟,vLLM V1 用 piecewise 路线追赶,SGLang 类似。

---

**Q5:prefix cache 命中率怎么算?怎么调?调高会有什么风险?**

```
命中率 = 命中的前缀 token 数 / 请求总输入 token 数
```

调高方向:

1. **增大 KV pool 给 cache 的比例**(默认 30–50%,可拉到 70%);
2. **按租户 / session 做亲和路由**(同一 session 落同一副本);
3. **保留 system prompt 为 pinned cache**(永不驱逐);
4. **跨节点 prefix cache 共享**(KV transfer over RDMA,见 Mooncake §12.4)。

风险:

- cache 占 KV 多 → active batch 缩小 → 高并发下 throughput 反而下降;
- 多租户共享 cache 会有侧信道泄漏 → 用 cache salting 隔离(§4.5);
- hash 碰撞概率虽然极低,但极端情况下会让用户 A 看到用户 B 的输出 → 上 SHA-256 或强校验。

---

**Q6:抢占什么时候触发?对 SLO 影响多大?怎么缓解?**

触发条件:**KV pool 满,新 token 无 block 可分配**。这通常发生在:

- 并发数推得太高;
- 长输出请求堆积(每步都要新 block);
- 缺少 prefix cache 命中导致 KV 全是「新计算的」。

影响:被抢请求的 TPOT 从 ~20 ms 跳到 prefill 重做时间(几百 ms),P99 TBT/TPOT 出现明显尖峰。一次抢占常常击穿 P99 SLO。

缓解:

1. **降低 max_num_seqs / max_num_batched_tokens**,留 KV reserve;
2. **改 swap 为 recompute**(V1 默认),减少 swap 路径复杂度;
3. **PD 分离**(§12.4):decode 节点独享 KV pool,prefill 流量不会冲击 decode;
4. **chunked prefill**:让重做的 prefill 摊到多个 iter,降低对 TBT 的冲击;
5. **限流 / 拒绝服务**:KV 真不够时直接 503,比缓慢恶化好。

---

**Q7:SGLang 在 agent / 多轮场景常常比 vLLM 快 30–100%,为什么?**

三个机制叠加:

1. **RadixAttention 的细粒度前缀复用**:agent prompt 中 system + tool description + 历史多轮的重复率经常 > 80%,radix tree 能精确表达任意粒度的共享,命中率比 vLLM APC(block 粒度)更高;
2. **Cache-aware scheduling**:把命中长前缀的请求**连续**调度,避免命中 block 被中间请求挤掉,最大化复用窗;
3. **前端 DSL 主动暴露结构**:用户写「我这段是 system,这段是 few-shot」时调度器能更聪明地组织 cache。

在前缀不重复的 workload(每条 prompt 都不一样)上,三大栈接近;模型规模上去(70B+),attention 算子占比上升,prefix 优化的相对收益减少。

---

**Q8:为什么 vLLM V1 把抢占默认从 SWAP 改成 RECOMPUTE?这不是更慢了吗?**

理由有三:

1. **架构简化**:V1 用 unified scheduler,prefill / decode 一视同仁。swap 路径要单独维护 CPU 端 KV 管理 + PCIe 调度,与统一接口不和;
2. **现代 GPU 的 prefill 速度逼近 PCIe 带宽**:H100 上长 prompt prefill ~10K tok/s,数据量上看和 PCIe gen5 的有效带宽接近——swap 不再有显著的「时间」优势,只换来「省算力」;
3. **chunked prefill 摊掉了 recompute 的冲击**:重做 prefill 不必一次性占满 GPU,可以摊到多个 iter,对 TBT 影响有限。

所以 V1 选了**简单**(recompute 路径单一)+ **足够快**(配合 chunked prefill)的方案。如果你的 workload 是长 prompt + 频繁抢占 + 算力极贵,可以手动切回 swap 模式。

---

**Q9:vAttention 这种「不分页」方案以后会取代 PagedAttention 吗?**

可能性中等偏高。vAttention 的核心优势是 **kernel 不需要改**——这是巨大的工程胜利:

- 每个新 attention 变体(MLA、sliding window、推测解码 verifier、KV 量化变体)在 PagedAttention 世界里都要单独写 paged kernel;
- vAttention 让所有这些变体直接复用「连续 KV」的 kernel,几乎零工程成本。

劣势:依赖 CUDA Virtual Memory API,目前是 NVIDIA-only(其他厂商的 OS-level VM 支持不齐),demand-paging 的尾延迟需要 LLM-specific 优化(预分配、延迟回收)。

短期判断:vLLM / SGLang / TRT-LLM 不会立刻换栈(生态成本太大);但 vAttention 的思想会渗透——FlashInfer 已经在探索「block_size = 1 + 物理上连续」的混合方案。长期看「kernel 无感的分页」是趋势。

---

**Q10:面试一句话总结 vLLM/SGLang 为什么快?**

vLLM:**用 PagedAttention 解 KV 碎片 + Continuous Batching 解槽位空转,把 KV 利用率从 30% 推到 95%、batch size 翻 3 倍,同等显存下吞吐 10–15×**。

SGLang:**在 vLLM 思想之上,用 RadixAttention 把前缀复用做到任意粒度 + cache-aware scheduling 把命中请求连续调度,前缀重复率高的 workload 再翻 30%–6×**。

TRT-LLM:**同样的 continuous batching + paged KV,加上 NVIDIA 全栈优化(FP8 / NVFP4 / CUDA Graph / 推测解码的最成熟工程化),把 H100 / Hopper 指令集吃干**。

---

## 参考资料

### 必读论文 ⭐⭐

1. [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM, SOSP 2023 ⭐⭐
2. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — OSDI 2022(continuous batching + selective batching 奠基) ⭐⭐
3. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) — LMSYS, 2023 ⭐⭐
4. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill + stall-free batching, OSDI 2024 ⭐⭐
5. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) — MLSys 2025 best paper ⭐⭐
6. [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437) — ASPLOS 2025 ⭐

### 工程文档 / 博客

7. [vLLM V1 Architecture Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/) ⭐⭐
8. [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) — vLLM 官方深度博文 ⭐⭐
9. [vLLM Automatic Prefix Caching Implementation](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) ⭐
10. [vLLM CUDA Graphs Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/) ⭐
11. [SGLang RadixAttention Blog](https://lmsys.org/blog/2024-01-17-sglang/) ⭐⭐
12. [TensorRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/) ⭐
13. [HuggingFace TGI](https://huggingface.co/docs/text-generation-inference) — continuous batching 早期开源参考

### 调度与系统设计

14. [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) — prefill / decode 拆分思想
15. [DistServe](https://arxiv.org/abs/2401.09670) — Goodput 视角的 PD 分离
16. [Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079) ⭐⭐
17. [FastSwitch: Optimizing Context Switching Efficiency](https://arxiv.org/abs/2411.18424) — 公平性 / 抢占优化
18. [LLM Query Scheduling with Prefix Reuse and Latency Constraints](https://arxiv.org/abs/2502.04677) — 调度理论

### 指标与压测

19. [Revisiting SLO and Goodput Metrics in LLM Serving](https://arxiv.org/abs/2410.14257) ⭐
20. [On Evaluating Performance of LLM Inference Serving Systems](https://arxiv.org/abs/2507.09019)
21. [BentoML LLM Inference Metrics Handbook](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)

### 综述 / 入门

22. [LLM Inference Performance Engineering Best Practices (Databricks)](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) ⭐
23. [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)
24. [Comparing the Top 6 Inference Runtimes for LLM Serving in 2025](https://www.marktechpost.com/2025/11/07/comparing-the-top-6-inference-runtimes-for-llm-serving-in-2025/) — 横向对比
