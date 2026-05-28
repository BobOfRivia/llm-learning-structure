# 4.6 RDMA 与 GPUDirect（补充：Disaggregated Serving 的网络基石）

[← 返回框架](../../README.md) · 关联章节：[§4.3 Prefill vs Decode](./03-prefill-vs-decode.md) §6.2

> 本节是对 §4.3 §6.2 "Disaggregated Serving" 中一笔带过的 **"通过 KV-cache 迁移（RDMA）"** 的展开。要看懂 DistServe / Splitwise / Mooncake 这一类系统在工程上为什么能成立，必须先理解 RDMA 与 GPUDirect 解决的具体问题。

---

## 一、问题：为什么必须用 RDMA

Prefill 节点算完后，**KV-Cache 必须运到 Decode 节点**：

- Llama-3 70B + 4k prompt + GQA ≈ **5–10 GB** KV
- 必须在 TTFT 预算内传完（用户期望 < 500ms 看到首 token）
- 因此端到端带宽至少要 **20 GB/s 量级**

→ TCP/IP 跨机能跑到的 ~3 GB/s（25 GbE）远远不够。**RDMA 在此出场**。

---

## 二、RDMA 是什么

**R**emote **D**irect **M**emory **A**ccess —— 一台机器的网卡可以**直接读写另一台机器的应用层内存**，绕过 CPU、内核、TCP/IP 协议栈。

```
传统 TCP/IP：
  app → 系统调用 → 内核 buffer → 协议栈 → NIC → 网线
        (CPU 全程参与, 多次内存拷贝, 上下文切换)

RDMA：
  app → 用户态 verb → NIC 硬件 → 网线 → 对端 NIC → 对端 app 内存
        (CPU 几乎不参与, zero-copy)
```

三个核心特性，缺一个性能就崩：

- **Kernel bypass**：用户态库（libibverbs）直接驱动网卡，没有系统调用开销
- **Zero-copy**：网卡 DMA 直接读应用 buffer，不经过内核拷贝
- **CPU bypass（远端）**：单边操作下对端 CPU **完全不参与**

---

## 三、RDMA 的三种形态

| 协议 | 物理层 | 部署场景 | 性能 |
|------|--------|----------|------|
| **InfiniBand (IB)** | 专用网络 | NVIDIA SuperPOD、HPC 集群 | 最强，端到端延迟 ~1µs |
| **RoCE v2** | 以太网 + UDP | 公有云、企业 DC 主流 | 接近 IB，需要 PFC / DCQCN 防丢包 |
| iWARP | TCP | 几乎淘汰 | 一般 |

LLM 推理集群主流：**InfiniBand HDR / NDR (200 / 400 Gbps)** 或 **RoCE v2 (100 / 200 Gbps)**。

---

## 四、核心抽象（verb）

- **RDMA Write**：本机主动写到对端内存，对端 CPU 不感知 ⭐ KV 传输几乎都用这个
- **RDMA Read**：本机主动读对端内存
- **Send / Recv**：双边操作，传控制消息（"我准备好了"、"传完了"）

辅助概念：

- **Memory Region (MR)**：buffer 用前必须**注册**给网卡（pin 物理页 + 授权访问范围）。注册是慢操作 → 实战上预先注册一个大 MR 池循环复用。
- **Queue Pair (QP)**：一对发送 / 接收队列，等价于一条 RDMA 连接。

---

## 五、GPUDirect RDMA —— LLM 场景的关键

KV-Cache 在 **GPU HBM 里**，不在 CPU 内存里。如果走传统路径：

```
没 GPUDirect：
  GPU HBM → (DtoH 拷贝) → CPU mem → NIC → 网线
                              → 对端 NIC → 对端 CPU mem → (HtoD) → 对端 GPU HBM
```

每次 PCIe 跨越都是延迟和带宽损失。**GPUDirect RDMA** 让网卡通过 PCIe peer-to-peer **直接 DMA GPU 显存**：

```
有 GPUDirect RDMA：
  GPU HBM → NIC → 网线 → 对端 NIC → 对端 GPU HBM
            (全程不进 CPU 内存)
```

延迟降 30–50%，PCIe 带宽占用减半。**DistServe / Splitwise / Mooncake 全都强依赖 GPUDirect**。

---

## 六、在 Disaggregated Serving 里的实际数据流

```
[Prefill GPU]                          [Decode GPU]
   │                                        │
   │ 1. 算完 prefill, KV 在本地 HBM          │
   │                                        │
   │ 2. Send 控制消息: "KV ready"            │
   │ ────────────────────────────────────── ▶│
   │                                        │
   │ 3. RDMA Write (GPUDirect)              │
   │    层 0 的 K,V → 对端 HBM               │
   │ ══════════════════════════════════════ ▶│ 收到层 0 就能开始算
   │    层 1 的 K,V → 对端 HBM               │ (layer-wise pipeline)
   │ ══════════════════════════════════════ ▶│
   │    ...                                 │
```

工程优化点：

- **Layer-wise pipelining**（Mooncake 重点优化）：层级流水，Decode 收到前几层就开始算，不等全部到齐
- **预注册 MR 池**：避免热路径上注册内存
- **多 QP 并发**：单 QP 跑不满 400 Gbps，开多条并行
- **拓扑感知调度**：优先配对同一 leaf switch 下的 Prefill / Decode 节点，减少跨交换机跳数

---

## 七、带宽数字感觉

| 链路 | 单端口理论 | 实际有效 |
|------|----------|---------|
| TCP/IP over 25 GbE | 3 GB/s | ~2 GB/s |
| InfiniBand HDR 200 Gbps | 25 GB/s | ~22 GB/s |
| InfiniBand NDR 400 Gbps | 50 GB/s | ~45 GB/s |
| NVLink（节点内） | 900 GB/s | —— 远高于跨节点 |

H100 节点典型配 8 张 ConnectX-7（NDR 400），聚合 ~3.2 Tbps。**单流瓶颈通常在 PCIe Gen5 x16 ≈ 64 GB/s**，不在网卡。

---

## 关键问答

**Q1**：RDMA 跟 NVLink 是不是一回事？
- 不是。**NVLink 是节点内** GPU 互联（一台机器内 8 张卡之间）；**RDMA 是跨节点**。
- 节点内传 KV 用 NVLink（900 GB/s 量级），跨节点用 RDMA（25–50 GB/s 量级），数量级差异巨大。
- Disaggregated Serving 的瓶颈几乎总是在跨节点 RDMA 这一段。

**Q2**：RoCE 和 InfiniBand 怎么选？
- **InfiniBand**：自建集群、追求极致延迟、能接受 NVIDIA/Mellanox 单一供应商 → SuperPOD
- **RoCE v2**：复用已有以太网基础设施、公有云、要支持多租户 → 主流云厂商
- 性能差距 < 20%，但 RoCE 的拥塞控制（PFC + DCQCN）配置复杂，运维门槛高

**Q3**：为什么 Memory Region 注册是慢操作？
- 注册要 **pin 物理页**（防止 OS 把页换出），以及把虚拟地址 → 物理地址映射告诉网卡
- 涉及内核操作，延迟 ms 量级，大 buffer 可能几十 ms
- → 实战上**绝不在热路径注册**，开服务时预先分配一个大 MR 池循环用

**Q4**：单边 RDMA Write 对端怎么知道数据到了？
- RDMA Write 本身不通知。常见做法：
  - **Write with Immediate**：写完后捎带一个 32-bit 立即数，触发对端 CQE
  - **Doorbell 写**：写完后再 Write 一个标志位到对端约定地址
  - **轮询完成位**：对端轮询某个 sentinel
- DistServe / Mooncake 一般用 Write+Imm，简单且无额外往返

**Q5**：GPUDirect RDMA 对硬件有什么要求？
- GPU：NVIDIA Tesla / Datacenter 系列（V100 及以后稳定支持，消费卡不支持）
- 网卡：Mellanox ConnectX-4 及以后
- 主板拓扑：GPU 与 NIC **在同一 PCIe root complex / 同一 NUMA**，跨 NUMA 会绕路掉一半带宽
- 驱动：`nvidia_peermem`（旧 `nv_peer_mem`）必须装好

**Q6**：为什么 chunked prefill（§6.1）一般不需要 RDMA？
- chunked prefill 是**同一节点内**把 prefill 切成 chunk 与 decode 混 batch
- 没有跨节点 KV 迁移，所以不需要 RDMA
- 这也是它工程门槛比 disaggregated serving 低得多的原因

---

## 一句话总结

> **Disaggregated Serving 把 KV-Cache 在 GPU HBM 之间搬，RDMA + GPUDirect 是让"搬运"不成为新瓶颈的唯一手段**。没有它，分离方案省下来的 goodput 会被网络全吃光。

---

## 参考资料

- [Mellanox — RDMA Aware Networks Programming User Manual](https://network.nvidia.com/related-docs/prod_software/RDMA_Aware_Programming_user_manual.pdf)
- [NVIDIA — GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Zhong et al. 2024 — DistServe](https://arxiv.org/abs/2401.09670)
- [Patel et al. 2023 — Splitwise](https://arxiv.org/abs/2311.18677)
- [Qin et al. 2024 — Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [RDMA Consortium — InfiniBand vs RoCE vs iWARP](https://www.rdmaconsortium.org/)
