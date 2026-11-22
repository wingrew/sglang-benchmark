# XCache
**XCache** 是一套面向 MLA+DSA 注意力架构的超低比特 KV Cache 推理系统，提出了 **2-bit KV Cache 量化与热页衰退（Hot-page Decay）机制**，旨在突破传统低比特 KV Cache 量化在模型精度、推理效率及系统兼容性方面的限制，实现兼具高精度、高吞吐与生产级可用性的超低比特推理。

在基于 **GLM-5.2-FP8** 的初步评测中，XCache 展现出了接近无损的量化精度。尤其是在对模型推理能力与长程任务执行要求较高的 Coding Agent 场景中，XCache 在 **SWE-bench Verified 的 500 道测试任务中，仅比 BF16 基线少解决 1 道问题**，表明即使将 KV Cache 压缩至 2-bit，仍能够较好地保留模型复杂推理与代码生成能力。此外，针对现有多种低比特 KV Cache 量化方案（包括当前 SOTA 方法）普遍存在的**生成长度膨胀（Generation Length Inflation）**问题，XCache 在初步实验中未观察到类似现象，体现出其精度保持机制在推理行为稳定性方面的优势。

然而，强大的精度保持机制也对底层推理系统带来了显著挑战。为此，XCache 对 SGLang 推理引擎进行了系统性的协同优化。在缓存管理方面，XCache 将**混合精度 Cache Pool 与热页管理机制完整下沉至 GPU**，降低动态缓存管理开销；在计算层面，分别针对 Prefill 和 Decode 阶段设计专用的 **Fresh Attention 与 Mixed Attention 算子**，并通过旋转矩阵的权重吸收消除额外的运行时变换开销；在系统层面，重新设计与调整推理调度机制，并进一步完善了与 **Radix Cache、MTP（Multi-Token Prediction）、HiCache 及 Prefill Context Parallelism（CP）** 等关键特性的兼容与协同优化，使 2-bit KV Cache 不再局限于独立算子或特定实验场景，而是能够融入具备完整功能的生产级推理系统。

在性能方面，XCache 的优势不仅来自 KV Cache 压缩所释放的显存容量，更得益于缓存管理、Attention Kernel 与推理调度之间的深度协同优化。初步实验表明，**即使与经过充分调优的 SGLang FP8 推理配置相比，XCache 仍可实现 67% 的吞吐提升**；在常规推理配置下，其性能优势则更为显著。

总体而言，XCache 展示了 **2-bit KV Cache 在复杂 Agent 工作负载下兼顾推理精度、生成效率与系统吞吐的可行性**，并通过端到端系统设计，为超低比特 KV Cache 从研究原型走向实际大模型推理服务提供了一条可行路径。

## XCache 全量精度评测计划（TODO）

### 1. 实验目标

系统评估 XCache 与现有 KV Cache 量化方案在复杂推理、Coding Agent 和长程任务中的精度保持能力、生成长度及实际执行耗时。

| 项目 | 配置 |
|---|---|
| 主测试模型 | GLM-5.2-FP8 |
| 扩展测试模型（可选） | GLM-5.3（FP8） |
| GPU | 8 × NVIDIA H200 |
| 推理框架 | SGLang |
| Benchmark | SWE-bench Verified、DeepSWE、GPQA、AIME 2026、Terminal-Bench 2.1、longbench_v2 |
| 对比配置 | BF16、FP8、OSCAR、RAQ-MLA |
| 随机种子 | **全实验共用 3 个固定 Seed：42、123、2026** |
| 主实验规模 | 6 × 4 × 3 = **72 次评测运行** |
| 扩展实验规模 | 两个模型合计 **144 次评测运行** |

**实验原则：**

- 所有 Benchmark 和量化配置均复用相同的三个随机种子，不需要为每组实验生成新的随机种子。
- BF16、FP8 指 KV Cache 精度，所有对比配置使用相同的 FP8 模型权重。
- XCache 已有实验结果应按照相同评测口径进行对齐；若需要重新运行全部测试，每个模型额外增加 18 次评测运行。

### 2. GLM-5.2-FP8 主实验 TODO

每格对应使用 Seed `42 / 123 / 2026` 进行的三次评测。

| Benchmark | BF16 | FP8 | OSCAR | RAQ-MLA |
|---|---|---|---|---|
| SWE-bench Verified (500) | 76.4% 77.4% 80.0%* | 79.8%* 79.6%* ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| DeepSWE | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| GPQA198(100k) | 91.4% 90.4% 91.4% | 90.4% 90.9% 90.4% | ☐ ☐ ☐ | ☐ ☐ ☐ |
| AIME 2026 | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| Terminal-Bench 2.1 (Terminus-2) | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| longbench_v2 | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |

注释：新版OSCAR代码仓库采用https://github.com/FutureMLS-Lab/OSCAR/tree/sglang-rebase，可让ai照此复现，已有glm5.3的测试流程，应该也是fp8的

**主实验进度：0 / 72**

### 3. GLM-5.3（FP8）扩展实验 TODO

若模型、推理框架及量化方法兼容，则重复上述评测，并继续复用相同的三个种子。

| Benchmark | BF16 | FP8 | OSCAR | RAQ-MLA |
|---|---|---|---|---|
| SWE-bench Verified (500) | ☐ ☐ ☐ | 88.2%* 89.2%* 89.0%* | ☐ ☐ ☐ | ☐ ☐ ☐ |
| DeepSWE | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| GPQA198(100k) | 90.9% 91.4% 92.4% | 92.9% 92.4% 89.4% | ☐ ☐ ☐ | ☐ ☐ ☐ |
| AIME 2026 | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| Terminal-Bench 2.1 (Terminus-2) | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |
| longbench_v2 | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ | ☐ ☐ ☐ |

**扩展实验进度：0 / 72**

### 实验配置说明

1.SWE-bench Verified

当前使用 SWE-bench Verified 的完整 test 集，共 500 题。Agent 生成补丁，再由官方 SWE-bench harness 执行测试评分。
| 项目 | 实际配置 |
|---|---|
| 推理框架 | SGLang 0.5.16|
| PyTorch / Transformers | 2.11.0+cu129 / 5.12.1 |
| Agent | mini-swe-agent==2.4.6 |
| 评分框架 | swebench==5.0.2 |
| API 客户端 | litellm==1.80.0，openai==2.6.1 |
| 数据加载 | datasets==5.0.0 |
| 服务镜像 | lmsysorg/sglang:v0.5.16-cu129 | 
| 上下文窗口，含输入与输出 | 409600 tokens | 
| 单轮最大输出 | 32768 tokens | 
| 采样 | temperature=1, top_p=1，top_k使用sglang默认值| 
Agent 与评分约束：

- 每题最多 250 步；单次模型请求超时 7200 秒，单条 shell 命令超时 120 秒。
- 每题使用独立 SWE-bench Docker 环境，在 /testbed 中修改代码、测试并提交补丁。
- 评分并发 4，每题测试超时 1800 秒；生成结束后停止模型、释放 GPU，再运行 CPU 评分。
- 最终分数为 resolved instances ÷ 500，另行统计空补丁和评测错误。

2.DeepSWE

建立 Datacurve DeepSWE v1.1 评测环境。

- 113 题已下载，使用独立 Python 3.12、Pier 0.3.1、mini-swe-agent 2.4.6。
- 单轮最大32K 输出。

- Agent 解题上限：3 小时/题。
- 评分上限：30 分钟/题。
- mini-swe-agent 默认 step_limit: 250轮

3.GPQA198

关键配置如下：
| 项目 | 配置 |
| GPQA 数量 | 全量 198 题，NUM_EXAMPLES 不设置 |
| 最大生成长度 | 跑一遍32k tokens，再跑一遍100k tokens|
| Temperature | 1.0 |
| Top-p | 0.95 |

需要统计各测试下的平均输出长度

4.AIME 2026

此前 AIME 2026 的 160K 测试使用以下配置，XCache 跑了 10 轮，BF16 跑了 1 轮对照。

| 项目 | 配置 |
|---|---|
| 服务环境 | glm52-oscar；Python 3.11.15、PyTorch 2.11.0+cu129、Transformers 5.12.1 |
| 并行配置 | 8 卡 TP，4 并发 |
| 上下文窗口 | 524288 tokens |
| 单题生成上限 | 163840 tokens，即 160 × 1024 |
| 采样参数 | temperature=1、top_p=1；未显式传入 top_k |
| 请求超时 | 单次 API 请求 4 小时 |

5.Terminal-Bench2.1

待补充

6.longbench_v2

待补充

### 4. 核心评测指标

| 优先级 | 指标 | 具体内容 | 状态 |
|---|---|---|---|
| P0 | **Accuracy / Pass Rate / Resolved Rate** | 汇总各 Benchmark 的正确率、通过率及任务解决率 | ☐ |
| P0 | **Average Generated Tokens** | 对比各 KV Cache 量化配置下的平均生成 Token 数，检查 Generation Length Inflation | ☐ |
| P0 | **Execution Time** | 统计任务平均端到端执行耗时及完整 Benchmark 总运行时间 | ☐ |

#### 统计与分析任务

- [ ] 汇总三个固定 Seed 的结果，计算 Mean ± Std
- [ ] 计算各方案相对于 BF16 的精度变化
- [ ] 计算各方案相对于 BF16 的平均生成 Token 增长比例
- [ ] 分别统计全部任务与成功任务的平均生成 Token 数
- [ ] 对 Coding Agent 统计整个任务中所有 LLM 调用累计生成的 Token 数
- [ ] 统计各 Benchmark 的平均任务执行耗时
- [ ] 统一采样参数、Agent 步数上限、Token 上限及任务超时配置
- [ ] 记录模型版本、SGLang 版本及评测环境

### 5. 实验准备 TODO

| 任务 | 状态 |
|---|---|
| 配置 GLM-5.2-FP8 的 BF16 KV Cache 基线 | ☐ |
| 配置 GLM-5.2-FP8 的 FP8 KV Cache 基线 | ☐ |
| 集成 OSCAR，验证 MLA+DSA 兼容性 | ☐ |
| 集成 RAQ-MLA，验证 MLA+DSA 兼容性 | ☐ |
| 对齐 XCache 的评测配置 | ☐ |
| 构建 SWE-bench Verified 评测环境 | ☐ |
| 构建 DeepSWE 评测环境 | ☐ |
| 构建 GPQA、AIME 2026 和 longbench_v2 评测环境 | ☐ |
| 构建 Terminal-Bench 2.1（Terminus-2）环境 | ☐ |
| 固定并统一使用三个 Seed：42、123、2026 | ☐ |
| 实现 Accuracy、Generated Tokens、Execution Time 自动统计 | ☐ |
| 汇总最终结果并生成论文对比图表 | ☐ |

### 6. 8 × H200 运行 GLM-5.2-FP8 的常用配置

#### 基础配置

| 参数 | 参考配置 |
|---|---|
| GPU | 8 × H200（141 GB/GPU） |
| Model | `zai-org/GLM-5.2-FP8` |
| Tensor Parallelism | `--tp 8` |
| BF16 KV Cache | `--kv-cache-dtype bf16` |
| FP8 KV Cache | `--kv-cache-dtype fp8_e4m3` |
| Memory Fraction | `0.80–0.85` |
| Chunked Prefill | `32768` |
| Max Running Requests | `80–256`，按负载调整 |
| MoE Communication | DeepEP（高并发配置） |
| Attention DP | `--attn-dp-size 8`（高吞吐配置） |
| Radix Cache | 各对比方案保持一致 |
| Prefill CP | 按需单独验证 |

#### BF16 KV Cache 基础评测配置

```bash
python -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.80 \
  --host 0.0.0.0 \
  --port 30000
```

FP8 KV Cache 基线将 `--kv-cache-dtype` 替换为 `fp8_e4m3`。

#### FP8 KV Cache 高吞吐参考配置

```bash
python -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --tp 8 \
  --kv-cache-dtype fp8_e4m3 \
  --attn-dp-size 8 \
  --moe-a2a-backend deepep \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 32768 \
  --max-running-requests 256 \
  --host 0.0.0.0 \
  --port 30000
```

该配置仅作为高吞吐测试参考起点，需根据实际 SGLang 版本和内核兼容性进行调整。

### 7. 最终论文实验产出

- [ ] 各 Benchmark 的 Accuracy / Pass Rate / Resolved Rate 对比表
- [ ] 不同量化方案的平均生成 Token 数对比图
- [ ] Generation Length Inflation 相对 BF16 的变化图
- [ ] 各 Benchmark 的平均任务执行耗时对比
- [ ] 三个固定 Seed 的统计结果与波动分析
- [ ] XCache 与 OSCAR、RAQ-MLA 的精度对比分析
- [ ] GLM-5.2 与 GLM-5.3 的跨模型验证结果
- [ ] XCache 精度保持机制与执行开销之间的权衡分析
