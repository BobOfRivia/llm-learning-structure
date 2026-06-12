# 14.4 MoE & 训练并行高频题（§9-§10）

[← 返回框架](../../README.md) · [📎 materials.md → §14.4](../../materials.md)

---

## 〇、本节回答什么

> 训练侧的硬核：MoE 路由 / 负载均衡 / EP 通信 / DP-TP-PP-SP-EP-CP 六维切分 / ZeRO / 混合精度 / PEFT。
>
> 覆盖 §9 MoE、§10 预训练 + 并行化 + PEFT。

---

## 一、MoE（→ §9）

### Q1.1 MoE 的核心思想 + 一个 forward 怎么走？ ⭐⭐

**【答 30s】**
- **核心**：把 FFN 拆成 N 个专家，每个 token 只路由到 top-k 个专家（k 通常 1 或 2）
- 训练时**总参数大**（更高容量），推理时**激活参数小**（只算选中的 k 个 expert）
- forward 步骤：
  1. token 表征 `h` 通过 router 网络得到 `logits = h · W_router`
  2. `top-k(softmax(logits))` 选 k 个 expert
  3. 每个选中 expert 算自己的 FFN
  4. 加权（softmax 权重）求和

**【加分项】**
- 经典 MoE：Switch Transformer (Fedus 2021, k=1)、GShard (Lepikhin 2020, k=2)
- DeepSeek-V3 用 k=8（细粒度），从 256 expert 选 8
- "稀疏激活"是 MoE 真正省的地方——同等 FLOPs 训出更大参数模型

**【追问】**
1. router 怎么训？→ 端到端反向传播，但 top-k 不可微，要 Straight-Through 或 aux loss 引导
2. 推理时 router 怎么并行？→ all-to-all 把 token 路由到 expert 所在 GPU
3. 总参数 vs 激活参数比例典型多少？→ DeepSeek-V3：671B 总 / 37B 激活（~5.5%）

---

### Q1.2 MoE 负载均衡：Aux Loss vs Aux-Loss-Free？ ⭐⭐⭐

**【答 30s】**
- **问题**：router 可能塌缩到几个 expert，导致大部分 expert 闲死
- **Aux Loss**（Switch / GShard）：加一个负载均衡 loss `α · sum(f_i · P_i)`，f_i 是 expert i 被选中次数比例，P_i 是平均概率
- **Aux-Loss-Free**（DeepSeek-V3）：给每个 expert 加一个 bias `b_i`，根据负载动态调整（**过载 ⇒ 减 bias，欠载 ⇒ 加 bias**），bias 不参与训练 loss
- DeepSeek 的方法避免了 aux loss 对主任务 loss 的干扰，**质量更好**

**【加分项】**
- Aux Loss 系数 α 调不好就两难：太小不均衡、太大伤主任务
- Aux-Loss-Free 调 bias 不进梯度，干净
- 还有 Expert Choice Routing（Zhou 2022）：让 expert 选 token 而非 token 选 expert，天然均衡但训推 mismatch

**【追问】**
1. expert choice 推理时怎么办？→ 推理时变成 token 选 expert，需要重训或转换
2. Aux-Loss-Free 怎么 init bias？→ 全 0，逐步动态调
3. bias 更新频率？→ 每个 micro-batch 后

---

### Q1.3 细粒度专家 + 共享专家是什么？ ⭐⭐

**【答 30s】**
- **细粒度专家**：把同一总参数量切成更多更小的 expert（256 vs 8），top-k 也更大（8 vs 2）
- **共享专家**（DeepSeek 独创）：除了 routed expert，还有 1-2 个**所有 token 都过**的共享 expert，承载"通用知识"
- 收益：路由更细 ⇒ 更好的容量利用；共享专家 ⇒ routed expert 可以更专一
- DeepSeek-MoE / V2 / V3 都用这个结构

**【追问】**
1. 共享专家有几个？→ DeepSeek-V3 1 个共享 + 256 个 routed
2. 细粒度的代价？→ all-to-all 通信开销变大，需要更精细 EP 调度
3. 为什么不能无限细？→ 单 expert 太小，FFN 矩阵乘退化为小矩阵，GPU 利用率掉

---

### Q1.4 Expert Parallelism (EP) + All-to-All 是什么？ ⭐⭐⭐

**【答 30s】**
- **EP**：把 expert 切到不同 GPU 上（不同 GPU 持有不同 expert）
- forward 时：
  1. 每张卡的 token 算 router 决定路由目标
  2. **All-to-All #1**：把 token 发到目标 expert 所在 GPU
  3. 各 GPU 算本地 expert FFN
  4. **All-to-All #2**：结果送回原 GPU
- 两次 all-to-all 是 MoE 训推的主要通信瓶颈

**【加分项】**
- DeepEP（DeepSeek 开源）：高效 all-to-all kernel，跨 NVLink/RDMA 优化
- EP 通常和 DP 组合：DP 内部 + EP 外部 = 经典 MoE 训练
- 推理时 EP 比训练更敏感——稀疏路由让通信很难均衡

**【追问】**
1. all-to-all 复杂度？→ O(N·M)，N 节点数 M 平均负载，是 bandwidth-bound
2. 怎么减少 all-to-all？→ 节点内聚合、topology-aware routing、auxiliary token grouping
3. 推理时为什么 MoE 难？→ expert imbalance 让 latency tail 长，需要 expert replication 或 dynamic batching

---

### Q1.5 MoE vs Dense 的 FLOPs / 显存 / 时延权衡？ ⭐⭐

**【答 30s】**

| 维度 | Dense | MoE |
| --- | --- | --- |
| 训练 FLOPs（同 token 数） | 高 | **低**（只算激活的 expert） |
| 参数量 | 中 | 大（5-15× 激活） |
| 显存（训练） | 中 | 大（所有 expert 都得在 GPU） |
| 推理时延（同质量） | 中 | **低**（激活参数少） |
| 推理 batching 难度 | 容易 | 难（route imbalance） |
| 部署成本 | 易 | 高（需 EP / all-to-all） |

**【加分项】**
- MoE 的 "FLOPs-equivalent" 模型质量更好——这是 MoE 真正赢的点
- DeepSeek-V3 (671B/37B) 单条 inference 大致等价 dense 37B 速度，但质量接近 dense 200B+
- Mixtral-8x7B 是早期开源 MoE 范例

**【追问】**
1. 什么时候不该用 MoE？→ 部署在小显存（<80GB）单卡场景；序列 batch 难以均衡
2. MoE 训稳定性？→ 比 Dense 难，需要 router 初始化 + 负载监控

---

## 二、并行化（→ §10）

### Q2.1 DP / TP / PP / SP / EP / CP 六维并行分别切什么？ ⭐⭐

**【答 30s】**

| 维度 | 切什么 | 通信模式 |
| --- | --- | --- |
| **DP** Data Parallel | batch 切 | all-reduce gradients |
| **TP** Tensor Parallel | 单层内的权重矩阵切 | all-reduce/all-gather |
| **PP** Pipeline Parallel | 层间切，流水 | point-to-point (micro-batch) |
| **SP** Sequence Parallel | seq 维度切（LN/dropout 段） | all-gather/reduce-scatter |
| **EP** Expert Parallel | MoE expert 切到不同 GPU | all-to-all |
| **CP** Context Parallel | 长 seq 切到不同 GPU（attention 协作） | all-gather KV / ring |

- 现代大模型训练通常 4D / 5D 组合：DP × TP × PP × SP × (EP or CP)
- DeepSeek-V3 训练用 DP × EP × PP（无 TP，因 MLA 通信负担轻）

**【加分项】**
- TP 通信量最大（单层内），一般只在 NVLink 域内（单节点 8 GPU）
- PP 容易气泡，用 1F1B / Interleaved 1F1B / Zero-Bubble 降低
- CP 是长上下文训练的关键，Megatron-LM 支持 ring attention

**【追问】**
1. TP 为什么不能跨节点？→ 单层内 all-reduce 频繁，跨节点 RDMA 太慢
2. PP 气泡怎么算？→ `bubble = (P-1) / M`，P 是 pipeline stage 数，M 是 micro-batch 数
3. Zero-Bubble PP 怎么消气泡？→ 把 backward 拆成 W (weight) 和 I (input)，更细粒度调度

---

### Q2.2 显存四块怎么算？训练 70B 全参 BF16 + Adam 需要多少？（→ §10.3） ⭐⭐⭐

**【答 30s】**

| 部分 | 公式 | 70B BF16 | 备注 |
| --- | --- | --- | --- |
| 参数 | N · 2 byte | 140 GB | 主参数 |
| 梯度 | N · 2 byte | 140 GB | 同 dtype |
| 优化器状态（Adam） | N · 12 byte | 840 GB | fp32 master + m + v |
| 激活 | 与 batch/seq 相关 | 几百 GB | gradient checkpointing 可砍 |

**合计 ~1.1 TB+**，远超单卡 80GB → 必须 TP/PP/ZeRO 切分(详见 Q2.3)。

**【加分项】**
- Adam 12 byte/param = fp32 master (4) + m (4) + v (4)
- 经典 16N 公式：BF16 (2+2) + Adam fp32 state (12) = 16 byte / param
- FP8 训练:参数+梯度 1+1 byte/p、master 仍 fp32，整体可省 ~40% (DeepSeek-V3)
- 激活省法：gradient checkpointing（recompute）、SP、CP、FlashAttention

**【追问】**
1. 为什么优化器状态最大？→ Adam 要存 fp32 master + 一阶矩 + 二阶矩
2. SGD 显存多少？→ 4N (仅 m)，但 LLM 不用 SGD,因收敛慢
3. 推理（不训）只要多少？→ 70B BF16 单纯权重 140GB；70B fp8 70GB；70B int4 ~35GB
4. QLoRA 微调 70B 凭什么单卡 24GB 能塞下？→ base NF4 量化 ~35GB 内不到一半,梯度+optimizer 只对 LoRA 参数 (~0.1N)，详见 §11.2

---

### Q2.3 ZeRO 1/2/3 区别？和 FSDP 关系？ ⭐⭐

**【答 30s】**

| ZeRO 阶段 | 切什么 | 通信 |
| --- | --- | --- |
| **ZeRO-1** | 优化器状态 | gradient all-reduce (同 DP) |
| **ZeRO-2** | + gradient | gradient reduce-scatter + all-gather |
| **ZeRO-3** | + 参数 | 每次 forward all-gather + backward all-gather |

- **FSDP** (Fully Sharded Data Parallel, PyTorch) = **ZeRO-3 的官方实现**
- ZeRO-3 显存最省，但 forward/backward 都要 all-gather 参数，通信压力大
- ZeRO-2 是性价比最优常用档

**【加分项】**
- DeepSpeed-ZeRO 是微软原始实现
- FSDP2 (PyTorch 2.4+) 是重写版本，性能更好
- ZeRO-Offload：把优化器状态/参数移到 CPU/NVMe

**【追问】**
1. ZeRO-3 适合什么场景？→ 单层不大但模型超大，单卡装不下完整副本
2. ZeRO 和 TP 组合？→ 可以，ZeRO 切跨 DP 组，TP 切单层；典型 7B 用 ZeRO，70B+ 用 ZeRO+TP
3. FSDP 在 H100 上 vs Megatron？→ FSDP 易用、Megatron 极致优化（针对 GPU 通信）；70B+ 训练 Megatron 仍是主流

---

### Q2.4 混合精度 FP32 / FP16 / BF16 / FP8 区别？ ⭐⭐

**【答 30s】**

| 格式 | 指数位 | 尾数位 | 动态范围 | 训练稳定性 |
| --- | --- | --- | --- | --- |
| FP32 | 8 | 23 | ~1e-38 ~ 1e38 | 基线 |
| **FP16** | 5 | 10 | ~6e-5 ~ 6e4 | **需 loss scaling**（容易下溢） |
| **BF16** | 8 | 7 | 同 FP32 | 不需 loss scaling，**LLM 标配** |
| **FP8 E4M3** | 4 | 3 | ~5e-2 ~ 4e2 | **forward 用**（精度高） |
| **FP8 E5M2** | 5 | 2 | ~6e-5 ~ 5e4 | **backward 用**（范围大） |

- BF16 解决了 FP16 训练崩塌问题，2022 后大模型 pretrain 全部 BF16
- FP8 (H100 Transformer Engine) 2024 起广泛使用，DeepSeek-V3 是首个 FP8 训出来的开源 SOTA

**【加分项】**
- BF16 的"指数同 FP32"是赢点：梯度量级横跨 1e-30 不会下溢
- FP8 训练要 per-tensor 或 per-block 缩放（DeepSeek-V3 用 1×128 / 128×128 block scaling）
- master weight 仍用 FP32，梯度累加用 FP32，只是矩阵乘走 FP8

**【追问】**
1. BF16 vs FP16 训练哪个更稳？→ BF16 显著更稳，工业界默认
2. FP8 训练有哪些坑？→ Loss spike、attention softmax 精度问题、累加要 FP32
3. NVFP4 / MXFP4 是什么？→ 4-bit 浮点，Blackwell 上的新格式，主要用于 inference

---

### Q2.5 LoRA / QLoRA / DoRA / AdaLoRA 区别？（PEFT,主章已搬至 §11.2） ⭐⭐

**【答 30s】**

| 方法 | 思路 |
| --- | --- |
| **LoRA** (Hu 2021) | 冻原模型，在每个线性层加 `ΔW = B·A`（低秩 r=8/16）；训练参数砍 1000× |
| **QLoRA** (Dettmers 2023) | LoRA + **base 模型量化到 NF4**；70B 单 24GB 卡就能微调 |
| **DoRA** (Liu 2024) | LoRA 分解为方向 + 幅度，质量更接近 full FT |
| **AdaLoRA** (Zhang 2023) | 动态调 rank，重要层多分配 |

**【加分项】**
- LoRA 的 `B` 初始化为 0、`A` 高斯，保证初始 ΔW=0
- α / r 缩放：实际 `ΔW = (α/r) · B · A`，α 通常等于 r 或 2r
- QLoRA 的 NF4 = NormalFloat 4-bit，针对正态分布权重设计；double quant 再压缩 scale
- DoRA 在 reasoning task 比 LoRA 涨 1-3 分

**【追问】**
1. LoRA rank r 怎么选？→ 一般 8/16/32，看任务复杂度
2. LoRA 适合 reasoning fine-tune 吗？→ 强 reasoning（如 R1 蒸馏）建议 full FT，LoRA 略弱
3. LoRA 部署时 merge 还是不 merge？→ 多 LoRA 服务用不 merge + S-LoRA；单一任务 merge 减少推理开销

---

### Q2.6 知识蒸馏分类 + 在 LLM 里怎么用？ ⭐⭐

**【答 30s】**
- **Response distillation**（黑盒）：用 teacher 生成的输出文本作 SFT 数据
- **Logit distillation**（白盒）：让 student 的 softmax 逼近 teacher 的 softmax（KL 散度）
- **Hidden state distillation**：对齐中间层表征
- **DeepSeek R1 → Qwen 蒸馏**：用 R1 生成 800K reasoning 样本 SFT 给 Qwen-7/14/32B
- **MiniCPM / Gemma-2** 都做 logit 蒸馏

**【加分项】**
- KL 蒸馏要用 reverse KL 还是 forward KL？→ DistilLLM (Ko 2024) 用 GKD（generalized JS）
- on-policy vs off-policy：on-policy 让 student 自己 sample 再算 KL（更稳定）
- 见 [00-补充知识/01-kl散度作为损失函数.md](../00-补充知识/01-kl散度作为损失函数.md)

**【追问】**
1. R1 蒸馏到小模型为什么有效？→ teacher 已把 reasoning 推开，student SFT 上模仿生成轨迹
2. logit 蒸馏在 LLM 难点？→ vocab 大、需要对齐 tokenizer
3. 蒸馏比 SFT 优势？→ 软标签信息密度高，能传"分布"而不只"答案"

---

## 三、关键问答（综合）

### Q3.1 给你 1000 张 H100 训一个 200B 模型，你怎么切？

> 典型 5D：TP=8（单节点 NVLink 内）× PP=8 × DP=16（across 128 nodes）× SP（同 TP 组内）。如果 MoE 再加 EP（替代 TP 一部分）。混合精度用 BF16 + FP8 GEMM。优化器 FP32 master + ZeRO-1 切 optimizer state。

### Q3.2 MoE 和 Dense 在工程上最大的差别？

> Dense 训推 communication 是结构化的（all-reduce、point-to-point），MoE 引入 **all-to-all** 这种和 token-level 路由相关的不规则通信，且**负载不平衡 + 推理 batching 难**。MoE 工程难度比 Dense 高一档。

### Q3.3 DeepSeek-V3 为什么训得起 671B？

> 几个关键：MLA 大砍 KV 显存 + 细粒度专家 + Aux-Loss-Free + **FP8 训练** + DualPipe 流水线 + DeepEP 高效 all-to-all + 2048 张 H800 14.8T token 共 ~2.78M H800-hours。工程上 DeepSeek 把 MoE+FP8 当作系统课题做。

### Q3.4 解释一下 Pipeline Parallel 气泡

> Pipeline 划分 P 个 stage，micro-batch M 个，气泡比例 ≈ (P-1)/(P-1+M)。如 P=8, M=8 气泡 ≈ 47%。降低办法：增大 M、Interleaved 1F1B、Zero-Bubble PP（拆 backward 成 W 和 I）。

### Q3.5 QLoRA 微调 70B 需要多少显存？

> NF4 base = 70 × 0.5 = 35GB；LoRA 参数 ~50MB；优化器 ~100MB；激活 (batch=1, seq=2K, gradient checkpoint) ~5-10GB。**单张 48GB 卡可以**（A6000、RTX 6000 Ada）。这是 QLoRA 论文的卖点。

### Q3.6 FSDP 和 DeepSpeed 我该选哪个？

> 7B-13B 推荐 FSDP（PyTorch 原生、易用）；70B+ pretrain 推荐 Megatron-LM 或 DeepSpeed（更细的 TP/PP 控制 + 通信优化）。HF Accelerate 提供 wrapper 屏蔽差异。

### Q3.7 BF16 训完 deploy fp16 / fp8 推理有什么注意？

> BF16 训出来的 weight 转 fp16 一般 ok（精度足够），但要看是否有极大/极小值（少数 attention layer 容易出 outlier）；fp8 推理需要做 calibration（per-tensor scale），DeepSeek-V3 本身就是 fp8 训 fp8 推。

---

## 四、参考资料

### MoE
- ⭐⭐ [Switch Transformer (Fedus 2021)](https://arxiv.org/abs/2101.03961)
- ⭐⭐ [GShard (Lepikhin 2020)](https://arxiv.org/abs/2006.16668)
- ⭐⭐ [DeepSeek-MoE (2024-01)](https://arxiv.org/abs/2401.06066)
- ⭐⭐ [DeepSeek-V3 Tech Report](https://arxiv.org/abs/2412.19437) — Aux-Loss-Free + 细粒度
- ⭐ [Mixtral 8x7B (Jiang 2024)](https://arxiv.org/abs/2401.04088)
- ⭐ [Expert Choice Routing (Zhou 2022)](https://arxiv.org/abs/2202.09368)
- ⭐ [DeepEP (DeepSeek 开源)](https://github.com/deepseek-ai/DeepEP)

### 并行化
- ⭐⭐ [Megatron-LM (Shoeybi 2019)](https://arxiv.org/abs/1909.08053)
- ⭐⭐ [ZeRO (Rajbhandari 2019)](https://arxiv.org/abs/1910.02054)
- ⭐⭐ [FSDP (PyTorch 2023)](https://arxiv.org/abs/2304.11277)
- ⭐⭐ [Ring Attention (Liu 2023)](https://arxiv.org/abs/2310.01889) — CP
- ⭐ [Zero-Bubble Pipeline (Qi 2023)](https://arxiv.org/abs/2401.10241)
- ⭐⭐ [HuggingFace Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — 训练并行可视化教程

### 混合精度
- ⭐⭐ [Mixed Precision Training (Micikevicius 2017)](https://arxiv.org/abs/1710.03740)
- ⭐ [BF16 vs FP16 for LLM (Kalamkar 2019)](https://arxiv.org/abs/1905.12322)
- ⭐⭐ [FP8 Formats for Deep Learning (NVIDIA 2022)](https://arxiv.org/abs/2209.05433)
- ⭐⭐ [DeepSeek FP8 Training (V3 report Section 3)](https://arxiv.org/abs/2412.19437)

### PEFT
- ⭐⭐ [LoRA (Hu 2021)](https://arxiv.org/abs/2106.09685)
- ⭐⭐ [QLoRA (Dettmers 2023 NeurIPS)](https://arxiv.org/abs/2305.14314)
- ⭐ [DoRA (Liu 2024)](https://arxiv.org/abs/2402.09353)
- ⭐ [PEFT 库（HuggingFace）](https://github.com/huggingface/peft)
- ⭐ [S-LoRA — multi-LoRA serving (Sheng 2023)](https://arxiv.org/abs/2311.03285)

### 蒸馏
- ⭐⭐ [Distilling the Knowledge in a NN (Hinton 2015)](https://arxiv.org/abs/1503.02531)
- ⭐ [MiniLLM (Gu 2023)](https://arxiv.org/abs/2306.08543)
- ⭐⭐ [DeepSeek-R1 Tech Report — distillation section](https://arxiv.org/abs/2501.12948)

### 解读
- ⭐⭐ [Megatron / Nanotron / Ultrascale Playbook (HuggingFace)](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- ⭐⭐ [Sebastian Raschka — LoRA / DoRA 解读](https://magazine.sebastianraschka.com/)

---

→ 下一节 [14.5 后训练 & Reasoning 题](./05-后训练-reasoning题.md)
