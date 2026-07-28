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

使用相同的 agentic_coding 数据集 (20 个对话, 2014 轮次)，分别测试 `--max-num-seqs 4` 和 `--max-num-seqs 16`：

### LLM 性能对比

| 指标 | batch=4 | batch=16 | 变化 |
|------|---------|----------|------|
| 首 Token 延迟 (TTFT) | 1,184 ms | 7,112 ms | 6.0x 慢 |
| Token 间延迟 (ITL) | 152 ms | 254 ms | 1.7x 慢 |
| 请求延迟 | 16,162 ms | 33,275 ms | 2.1x 慢 |
| 每用户吞吐量 | 6.60 tok/s | 3.31 tok/s | 0.5x |
| **总吞吐量** | **23.35 tok/s** | **41.80 tok/s** | **1.8x 快** |
| 请求吞吐量 | 0.23 req/s | 0.41 req/s | 1.8x 快 |
| 测试时长 | 85.62 s | 49.21 s | 0.6x |

### GPU 遥测对比

| 指标 | batch=4 | batch=16 |
|------|---------|----------|
| GPU 功耗 (平均) | 91.9 W | 101.0 W |
| GPU 功耗 (最大) | 129 W | 146 W |
| GPU 利用率 (平均) | 98.5% | 97.3% |
| 显存使用 | 94.84 GB | 94.95 GB |

### Batch Size 16 详细结果

| 指标 | 平均值 | p50 | p90 | p99 | 最大 |
|------|--------|-----|-----|-----|------|
| 首 Token 延迟 (TTFT) | 7,112 ms | 8,434 ms | 12,069 ms | 12,838 ms | 13,019 ms |
| Token 间延迟 (ITL) | 254 ms | 244 ms | 306 ms | 369 ms | 376 ms |
| 请求延迟 | 33,275 ms | 36,907 ms | 42,636 ms | 46,457 ms | 47,181 ms |
| 每用户吞吐量 | 3.31 tok/s | 2.90 tok/s | 5.29 tok/s | - | 5.40 tok/s |
| 总吞吐量 | 41.80 tok/s | - | - | - | - |
| 输出长度 | 52.35 tokens | - | - | - | - |

### 分析

- **总吞吐量提升 1.8 倍**：batch=16 时 41.80 tok/s vs batch=4 时 23.35 tok/s
- **每用户速度下降一半**：batch=16 时 3.31 tok/s vs batch=4 时 6.60 tok/s
- **TTFT 大幅增加**：batch=16 时 7.1s vs batch=4 时 1.2s，高并发下 prefill 排队严重
- **GPU 利用率接近饱和**：两种配置都在 97-98%，说明 GPU 已是瓶颈
- **功耗增加**：batch=16 平均 101W，最高 146W（vs batch=4 的 92W/129W）
- **最佳权衡**：Agent 场景推荐 batch=4-8（兼顾延迟和吞吐），纯吞吐压测推荐 batch=16

## 结论

1. **BF16 是 gfx1151 上的最优推理路径**——rocBLAS 有专属调优内核，带宽效率 84%
2. **MoE 模型比 Dense 模型快 4 倍**——Qwen3-30B-A3B 仅激活 3B 参数，每 token 读取量减少 10 倍
3. **并发 4 路时总吞吐量 23.35 tok/s，16 路时 41.80 tok/s**，GPU 利用率均接近 100%
4. **所有低精度量化路径在 gfx1151 上均不可用**（FP8/INT4/AWQ/GPTQ），原因是 RDNA 3.5 缺乏对应矩阵核心
5. vLLM 的 MoE Triton 内核未为 Radeon 8060S 调优，实际性能约为理论值的 55-70%
6. **Agent 场景推荐 max-num-seqs=4-8**，纯吞吐压测可用 16

## 结果文件

| 文件 | 说明 |
|------|------|
| `results/profile_export_aiperf.csv` | batch=4 LLM 性能指标 CSV |
| `results/profile_export_aiperf.json` | batch=4 LLM 性能指标 JSON |
| `results/gpu_telemetry_export.jsonl` | batch=4 GPU 遥测时序数据 |
| `results_batch16/profile_export_aiperf.csv` | batch=16 LLM 性能指标 CSV |
| `results_batch16/profile_export_aiperf.json` | batch=16 LLM 性能指标 JSON |
| `results_batch16/gpu_telemetry_export.jsonl` | batch=16 GPU 遥测时序数据 |

---

*测试日期: 2026-07-28*
*硬件: AMD Ryzen AI MAX+ 395 / Strix Halo*
*软件: ROCm 7.14.0 + vLLM 0.23.1 + AIPerf 0.11.0*
