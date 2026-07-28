# vLLM + Qwen3-30B-A3B 基准测试报告

## 硬件环境

| 项目 | 规格 |
|------|------|
| CPU | AMD Ryzen AI MAX+ 395 (16 核 32 线程) |
| GPU | AMD Radeon 8060S Graphics (gfx1151 / RDNA 3.5) |
| 内存 | 96 GB LPDDR5X (共享，GPU/CPU 统一寻址) |
| 存储 | Lexar NM790 NVMe SSD 2TB |
| 内核 | Linux 7.0.0-28-generic (HWE) |
| OS | Ubuntu 24.04.4 LTS |

## 软件栈

| 组件 | 版本 |
|------|------|
| ROCm | 7.14.0 (HIP 7.14.60850) |
| PyTorch | 2.11.0+rocm7.14.0 |
| vLLM | 0.23.1.dev1 |
| Docker | 29.1.3 |
| Docker 镜像 | `rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0` |

## 模型信息

| 项目 | 数值 |
|------|------|
| 模型 | Qwen3-30B-A3B (MoE) |
| 量化 | BF16 (无量化) |
| 总参数 | 30.5B |
| 激活参数 | 3.08B (8/128 专家 + 共享专家) |
| 架构 | Qwen3MoeForCausalLM |
| 模型大小 | 57 GB (16 个 safetensors 分片) |
| 上下文长度 | 40,960 tokens |
| KV cache | 28.38 GiB (309,952 tokens) |

## vLLM 启动参数

```bash
vllm serve /app/models/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --chat-template chat_template_no_think.jinja \
  --port 8000
```

## 量化方案对比

### BF16 vs FP8 (Qwen3-32B)

| 指标 | BF16 (Qwen3-32B Dense) | FP8 (Qwen3-32B Dense) |
|------|----------------------|---------------------|
| 模型大小 | 62 GB | 32 GB |
| 推理速度 | **3.5 tok/s** | 0.5 tok/s |
| GPU 利用率 | 100% | 100% |
| 内存带宽效率 | **84%** (214 GB/s) | 6% (~15 GB/s) |
| 计算路径 | rocBLAS (原生 AOT 内核) | Triton JIT (未调优) |
| 启动时间 | ~10 分钟 | ~40 分钟 (含 JIT 编译) |

### 关键发现

- gfx1151 (RDNA 3.5) **没有 FP8 矩阵核心**，FP8 走 Triton JIT 软件反量化路径，效率极低
- BF16 走 rocBLAS，有 gfx1151 专属预编译内核，带宽效率 84%
- GPTQ-Int4 在 gfx1151 上报 `qzeros shape mismatch` 错误
- AWQ 在 gfx1151 上有段错误 bug (vLLM issue #37151)

## Dense vs MoE 对比

| 指标 | Qwen3-32B (Dense BF16) | Qwen3-30B-A3B (MoE BF16) |
|------|----------------------|------------------------|
| 总参数 | 32B | 30.5B |
| 激活参数 | 32B | 3.08B |
| 每 token 读取 | 62 GB | ~6 GB |
| 推理速度 (单请求) | 3.5 tok/s | **14.4 tok/s** |
| 启动时间 | ~10 分钟 | ~5 分钟 |
| JIT 编译 | 不需要 | 首次请求 (已缓存) |
| MoE 内核 | N/A | Triton fused-MoE |

## AIPerf 基准测试结果

### 测试配置

| 项目 | 数值 |
|------|------|
| 工具 | NVIDIA AIPerf 0.11.0 |
| 数据集 | agentic_coding (20 个对话, 2014 轮次) |
| 并发数 | 4 |
| 请求数 | 20 |
| 流式 | 是 |
| GPU 遥测 | amdsmi |
| 测试时长 | 85.62 秒 |

### LLM 性能指标

| 指标 | 平均值 | p50 | p90 | p99 | 最大 |
|------|--------|-----|-----|-----|------|
| 首 Token 延迟 (TTFT) | 1,184 ms | 868 ms | 2,804 ms | 2,804 ms | 2,804 ms |
| Token 间延迟 (ITL) | 152 ms | 156 ms | 161 ms | 164 ms | 164 ms |
| 请求延迟 | 16,162 ms | 15,375 ms | 21,907 ms | 22,241 ms | 22,316 ms |
| 每用户吞吐量 | 6.60 tok/s | 6.42 tok/s | 6.96 tok/s | - | 8.18 tok/s |
| 端到端吞吐量 | 6.18 tok/s | 6.06 tok/s | 6.41 tok/s | - | 7.57 tok/s |
| 总吞吐量 | 23.35 tok/s | - | - | - | - |
| 请求吞吐量 | 0.23 req/s | - | - | - | - |
| 输出长度 | 99.95 tokens | 93 | 131 | 163 | 169 |
| 输入长度 | 1,805 tokens | 1,826 | 2,189 | 2,732 | 2,836 |

### GPU 遥测指标

| 指标 | 平均值 | p50 | 最大 |
|------|--------|-----|------|
| GPU 功耗 | 91.9 W | 92 W | 129 W |
| GPU 利用率 | 98.5% | 100% | 100% |
| 显存使用 | 94.84 GB | 94.84 GB | 94.84 GB |
| ECC 错误 | 0 | - | - |

### 最大并发能力分析

```
KV cache 容量: 309,952 tokens
模型权重: 57 GB
```

| 每请求上下文 | 最大并发数 | 适用场景 |
|------------|-----------|---------|
| 40,960 tokens | 7 | 满上下文 Agent |
| 8,192 tokens | 37 | 中等对话 |
| 4,096 tokens | 75 | 短对话/代码 |

## Batch Size 对比测试

使用相同的 agentic_coding 数据集 (20 个对话, 2014 轮次)，vLLM 设 `--max-num-seqs 32`，通过 AIPerf `--concurrency` 参数测试不同并发数 (1/4/8/12/16/24/32)。

### 完整性能对比

| Batch | 总吞吐 (tok/s) | 每用户 (tok/s) | TTFT (ms) | ITL (ms) | 请求延迟 (ms) | GPU利用率 (%) | 功耗 (W) |
|-------|---------------|---------------|----------|---------|-------------|-------------|---------|
| 1 | 12.8 | 12.7 | 796 | 70 | 8,011 | 98.2 | 87.9 |
| 4 | 23.4 | 6.3 | 1,057 | 152 | 17,416 | 98.4 | 90.9 |
| 8 | 32.0 | 4.9 | 2,479 | 191 | 26,079 | 98.1 | 90.9 |
| 12 | 44.2 | 4.3 | 2,890 | 209 | 26,684 | 97.1 | 92.4 |
| 16 | 44.7 | 3.6 | 7,935 | 222 | 32,084 | 96.9 | 99.6 |
| 24 | 59.7 | 3.4 | 7,860 | 220 | 31,355 | 95.9 | 89.1 |
| 32 | 69.2 | 4.4 | 907 | 219 | 24,821 | 95.4 | 79.1 |

### 趋势分析

- **总吞吐量随并发线性增长**：从 batch=1 的 12.8 tok/s 到 batch=32 的 69.2 tok/s，增长 5.4 倍
- **每用户速度递减**：batch=1 时 12.7 tok/s，batch=24 时降至 3.4 tok/s，batch=32 回升至 4.4 tok/s（调度优化）
- **TTFT 在 batch=16/24 时剧增**：prefill 排队严重，达 7-8 秒；batch=32 时因 continuous batching 优化反而降至 907ms
- **ITL 趋于稳定**：batch>=8 后 ITL 稳定在 190-220ms，说明 decode 阶段 GPU 时间分摊均衡
- **GPU 利用率始终 >95%**：所有并发下 GPU 都是瓶颈，无空闲
- **功耗稳定**：87-100W 范围，batch=16 峰值 99.6W

### 场景推荐

| 场景 | 推荐 batch size | 理由 |
|------|----------------|------|
| 单用户交互 (Agent) | 1 | TTFT 0.8s，每用户 12.7 tok/s，体验最佳 |
| 多用户 Agent (延迟敏感) | 4-8 | TTFT 1-2.5s，每用户 5-6 tok/s |
| 多用户 Agent (吞吐优先) | 12 | 总吞吐 44 tok/s，TTFT 2.9s 可接受 |
| 高并发 API 服务 | 24-32 | 总吞吐 60-69 tok/s，TTFT 波动大 |
| 纯压测 | 32 | 最大总吞吐 69.2 tok/s |

## 结论

1. **BF16 是 gfx1151 上的最优推理路径**——rocBLAS 有专属调优内核，带宽效率 84%
2. **MoE 模型比 Dense 模型快 4 倍**——Qwen3-30B-A3B 仅激活 3B 参数，每 token 读取量减少 10 倍
3. **总吞吐量随并发增长**：batch=1 时 12.8 tok/s → batch=32 时 69.2 tok/s（5.4 倍提升）
4. **GPU 利用率始终 >95%**，所有并发下 GPU 都是瓶颈
5. **所有低精度量化路径在 gfx1151 上均不可用**（FP8/INT4/AWQ/GPTQ），原因是 RDNA 3.5 缺乏对应矩阵核心
6. vLLM 的 MoE Triton 内核未为 Radeon 8060S 调优，实际性能约为理论值的 55-70%
7. **Agent 场景推荐 batch=4-8**（兼顾延迟和吞吐），高并发 API 推荐 batch=24-32

## 结果文件

| 文件 | 说明 |
|------|------|
| `results_batch1/` | batch=1 测试结果 |
| `results_batch4/` | batch=4 测试结果 |
| `results_batch8/` | batch=8 测试结果 |
| `results_batch12/` | batch=12 测试结果 |
| `results_batch16/` | batch=16 测试结果 |
| `results_batch24/` | batch=24 测试结果 |
| `results_batch32/` | batch=32 测试结果 |

每个目录包含：`profile_export_aiperf.csv/json`（LLM 指标）、`gpu_telemetry_export.jsonl`（GPU 遥测）、`server_metrics_export.csv/json`（服务端指标）

---

*测试日期: 2026-07-28*
*硬件: AMD Ryzen AI MAX+ 395 / Strix Halo*
*软件: ROCm 7.14.0 + vLLM 0.23.1 + AIPerf 0.11.0*
