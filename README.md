# AMD Strix Halo (Ryzen AI Max+ 395)

本仓库记录 AMD Ryzen AI Max+ 395（Strix Halo）开发机的架构资料、配置笔记和实验记录。

## 硬件概览

| 项目 | 规格 |
|------|------|
| 处理器 | AMD Ryzen AI Max+ 395 (Strix Halo) |
| 统一内存 | 128 GB LPDDR5x |
| iGPU | Radeon 8060S (RDNA 3.5, 60 FP16 TFLOPS) |
| NPU | XDNA 2 / AIE2P (50 INT8 TOPS) |
| GPU VRAM | 96 GB (BIOS 预留) |
| 操作系统 | Ubuntu 24.04 LTS |

## 文档目录

- [GPU 与 NPU 架构说明](docs/gpu-npu-architecture.md) — 硬件实测数据、架构对照、为什么 NPU 跑小模型的分析

## 目标

- GPU (iGPU) 跑大模型 (Qwen2.5-72B) — llama.cpp + Vulkan
- NPU 跑小模型 (Phi-4-mini) — Ryzen AI SW + OGA NPU-only flow
- 两者真正并行调度
