# AMD Ryzen AI Max+ 395 — GPU 与 NPU 架构说明

> 本文档基于本机（AMD Ryzen AI Max+ 395 / Strix Halo，主机名 `AI-Station-395-Max`）的实测数据，结合 AMD 官方公布的规格整理而成。
> 记录日期：2026-07-27

---

## 一、硬件身份（实测确认）

| 项目 | 实测值 | 来源 |
|------|--------|------|
| 型号 | AMD Ryzen AI Max+ 395（代号 Strix Halo） | `hostname` = `AI-Station-395-Max` |
| 统一内存 | 128 GB LPDDR5x（其中 96 GB 划给 iGPU 当 VRAM） | `mem_info_vram_total` + BIOS 配置 |
| 操作系统 | Ubuntu 24.04.4 LTS，Wayland 会话 | `lsb_release -a` |
| 内核 | 6.10+（`amdxdna` 主线驱动要求 v6.10 以上） | `uname -r` |
| GPU PCI ID | `1022:1586` (rev c1)，Display controller | `lspci` |
| GPU 驱动 | `amdgpu` | `lsmod` |
| NPU PCI ID | `1022:17f0` (rev 11)，Signal processing controller | `lspci` |
| NPU 驱动 | `amdxdna` 0.7.0（主线内核内置，`intree: Y`） | `modinfo amdxdna` |
| NPU 固件 | `amdnpu/17f0_11/npu.sbin` | `journalctl -k` |
| NPU 设备节点 | `/dev/accel/accel0`（major 261, minor 0） | `ls /dev/accel/` |
| NPU 内部代号 | `aie2p`，拓扑 `6×8` | AMD XRT 文档 |

### 内核启动日志（NPU 初始化，无报错）

```
amdxdna 0000:c6:00.1: [drm] Load firmware amdnpu/17f0_11/npu.sbin
amdxdna 0000:c6:00.1: enabling device (0000 -> 0002)
[drm] Initialized amdxdna_accel_driver 0.7.0 for 0000:c6:00.1 on minor 0
```

---

## 二、GPU 详解（Radeon 8060S iGPU）

### 基本规格

| 项目 | 规格 |
|------|------|
| 架构代次 | RDNA 3.5（gfx1151） |
| 计算单元（CU） | 40 个 CU |
| 峰值算力 | **60 FP16 TFLOPS**（AMD 官方） |
| 显存 | 共享 LPDDR5X，BIOS 预留 96 GB 窗口 |
| 驱动 | `amdgpu`（主线内核） |
| 编程模型 | Vulkan / ROCm / HIP / OpenCL（通用 GPGPU） |
| LLM 推理路径 | llama.cpp（Vulkan 后端）或 ROCm |

### 设计定位

GPU 是**通用并行计算**单元：大量 SIMT 流处理器，可运行任意并行 workload——图形渲染、GPGPU 计算、LLM 推理、视频编解码。灵活性强，软件生态成熟。

### 在本机的角色

- 跑**大模型**（Qwen2.5-72B-Int4，权重约 36 GB）
- 96 GB VRAM 装下 72B 权重 + KV cache 绰绰有余
- 用 llama.cpp + Vulkan 后端（gfx1151 的 ROCm 支持仍在完善，Vulkan 更稳）

---

## 三、NPU 详解（XDNA 2 / AIE2P）

### 基本规格

| 项目 | 规格 |
|------|------|
| 架构代次 | XDNA 2（AIE2P，第二代 AI Engine） |
| 拓扑 | 6×8 = 48 个 AIE tile |
| 峰值算力 | **50 TOPS**（INT8，AMD 官方） |
| 显存 | 无独立显存窗口，从系统内存经 DMA 分配 |
| 驱动 | `amdxdna`（主线内核）+ XRT 用户态 |
| 编程模型 | ONNX Runtime + VitisAI Execution Provider |
| LLM 推理路径 | ONNX Runtime GenAI (OGA) NPU-only flow |
| 支持模型 | 仅 AMD 预优化的 ONNX 模型（Phi-3/4、Qwen、Llama 等） |

### 设计定位

NPU 是**空间数据流阵列**：48 个 AIE tile 构成可重构的数据流图，模型被编译成静态计算图加载到 tile 上执行。

- 强项：固定结构的推理（CNN、小 Transformer），**低功耗高效率**
- 弱项：动态控制流（LLM 的变长 attention、KV cache 动态管理、采样逻辑）
- AMD 官方明确：*"NPU is not designed for ML training"*——也隐含了它对动态/大模型 workload 的局限

### 在本机的角色

- 跑**小模型**（Phi-4-mini），做标题生成、会话摘要等轻量任务
- 用 Ryzen AI SW 1.8.0 + OGA NPU-only flow
- AMD 官方实测（Phi-3.5-mini on NPU）：prefill 753 tokens/s，生成 49 tokens/s

### Linux 支持现状（重要）

Ryzen AI Software 1.8.0 在 Linux 上：

| 层 | 状态 | 说明 |
|----|------|------|
| ① 内核驱动 `amdxdna` | ✅ 已就绪 | 主线内核内置，固件加载成功 |
| ② XRT 用户态 runtime | ✅ 官方 deb | `xrt_*_24.04-amd64-*.deb` + `xrt_plugin.*-amdxdna.deb` |
| ③ Ryzen AI SW 1.8.0 | ✅ 官方 tgz | `ryzen_ai-1.8.0.tgz`，装到 Python venv |
| ④ 预优化 NPU 模型 | ✅ HuggingFace | `amd/Phi-3.5-mini-instruct_rai_1.8.0_npu_4K` 等 |
| ⑤ OpenAI 兼容 server | ⚠️ 需自建 | Lemonade Server 仅有 Windows 版；Linux 用 OGA Python API 包一层 |

**Linux 限制：**
- 仅支持 NPU-only flow，**不支持 Hybrid**（NPU+iGPU 混合）
- 不支持 Model Generate（不能自己转模型，只能用预优化版）
- 需要 `memlock unlimited` 配置

---

## 四、GPU vs NPU 架构对照

| 维度 | GPU（Radeon 8060S） | NPU（XDNA 2 / AIE2P） |
|------|----------------------|------------------------|
| **计算范式** | SIMT 流处理器，通用并行 | 空间数据流阵列，可重构计算图 |
| **架构代次** | RDNA 3.5（gfx1151） | XDNA 2（AIE2P） |
| **核心单元** | 40 个 Compute Unit | 48 个 AIE tile（6×8） |
| **峰值算力** | 60 **FP16** TFLOPS | 50 **INT8** TOPS |
| **数据类型** | FP32/FP16/INT8/INT4，全精度灵活 | INT8/INT16/BF16，面向推理 |
| **编程模型** | Vulkan / ROCm / HIP / OpenCL | ONNX Runtime + VitisAI EP |
| **内存访问** | 共享 LPDDR5X（96 GB 预留窗口），高带宽 | 共享 LPDDR5X，经 DMA buffer 访问 |
| **调度独立性** | 独立硅块，`amdgpu` 驱动 | 独立硅块，`amdxdna` 驱动 |
| **设计目标** | 通用并行计算、高吞吐 LLM 推理 | 低功耗、固定结构 AI 推理 |
| **LLM 路径** | llama.cpp（Vulkan） | OGA NPU-only flow |
| **模型来源** | 任意 GGUF/HF 模型 | 仅 AMD 预优化 ONNX 模型 |
| **大模型支持** | ✅ 72B 直接跑 | ❌ 无 72B 预优化版 |
| **训练支持** | ✅ | ❌ |

---

## 五、为什么 NPU 跑小模型？（算力分析）

从 TOPS 数字看，NPU（50 TOPS）似乎可与 GPU（60 TFLOPS）媲美。但"峰值算力相近"不等于"跑 LLM 的实际能力相近"。差距来自以下非算力因素：

### 1. LLM 推理的瓶颈是带宽，不是算力

LLM 生成阶段是 **memory-bound**：每生成一个 token，都要把整个模型权重从内存读一遍。这个阶段算力基本闲置，比的是**内存带宽**。

- **GPU**：96 GB 预留 VRAM 窗口，经 amdgpu 直连 LPDDR5X 高带宽通道，llama.cpp 优化过权重预取
- **NPU**：经 DMA buffer object 访问同一 LPDDR5X，多一层拷贝/调度，有效带宽更低

**证据**：AMD 官方实测 Phi-3.5-mini 在 NPU 上生成 49 tokens/s。同样模型在 iGPU 上用 llama.cpp 能轻松过百——差距就是带宽和软件优化，不是算力。

### 2. TOPS 数字的陷阱

| | GPU 60 | NPU 50 |
|---|---|---|
| 精度 | **FP16** TFLOPS | **INT8** TOPS |
| 换算同精度 | INT8 下远高于 60 | FP16 下远低于 50 |
| 性质 | 峰值，可持续接近 | 峰值，受功耗墙限制持续打折 |

50 INT8 TOPS ≠ 50 FP16 TFLOPS。NPU 的 50 是 INT8 峰值，跑 BF16 会大幅下降；GPU 的 60 是 FP16。同精度下 GPU 算力其实高得多。

### 3. NPU 架构天生不适合大 LLM

NPU 是**空间数据流阵列**（6×8 AIE tile）：
- 强项：固定结构计算图（CNN、小 Transformer），低功耗高效率
- 弱项：LLM 的动态控制流——attention 变长序列、KV cache 动态管理、采样逻辑，不适合静态数据流图

AMD 文档原话：*"NPU is not designed for ML training"*，也隐含了它对动态、大模型 workload 的局限。NPU 设计目标是**低功耗推理**，不是高吞吐。

### 4. 软件栈成熟度（最现实的约束）

| | GPU | NPU |
|---|---|---|
| 模型来源 | 任意 GGUF/HF 模型，llama.cpp 直接跑 | **只能用 AMD 预优化的 ONNX 模型** |
| 72B 支持 | ✅ llama.cpp 直接跑 | ❌ **没有 72B 的 NPU 预优化版** |
| 自己转模型 | ✅ `convert_hf_to_gguf.py` | ❌ Linux 不支持 Model Generate |
| 上下文长度 | 任意（VRAM 够就行） | 4K / 16K（受限于 NPU overlay） |
| batch / 流式 | ✅ 完整支持 | 仅 batch=1 |

**就算想把 72B 塞给 NPU，也根本没有可用的模型和软件路径。** 不是算力不够，是生态没准备好。

### 5. 显存窗口

- **GPU**：BIOS 预留 96 GB，装 72B-Int4（~36 GB）绰绰有余
- **NPU**：没有独立显存窗口，从系统内存（本机仅剩 ~31 GB）分配——72B 权重都装不下

### 结论

NPU 的 50 TOPS 在跑 LLM 时**发挥不出来**，因为：
1. LLM 是带宽瓶颈，NPU 有效带宽不如 GPU
2. 50 是 INT8 峰值，同精度下算力低于 GPU
3. 数据流阵列不适合 LLM 动态控制流
4. 没有 72B 的预优化模型，软件栈也不支持
5. 没有独立显存窗口，装不下大模型

**NPU 的真正价值**：低功耗、固定结构的小模型推理（CNN、ASR、小 LLM 做摘要/标题）。让它跑小模型不是"浪费算力"，而是"用对了地方"。GPU 跑大模型 + NPU 跑小模型，正是 AMD 这套硬件的设计意图。

---

## 六、并行调度可行性

**架构上完全支持，且是设计意图。**

1. **物理独立**：GPU 和 NPU 是 Strix Halo 封装内两块独立的计算单元，由各自内核驱动独立管理，计算资源互不抢占
2. **共享内存不共享算力**：两者都通过 LPDDR5X 统一内存访问数据，但计算单元完全隔离——能真正并行执行
3. **唯一共享瓶颈是带宽**：LPDDR5X 带宽被两者分摊。但小模型后台偶发跑，对大模型的带宽影响可忽略
4. **AMD 官方明确**：Ryzen AI SW 1.8 文档把 NPU-Only 模式的用途写成 *"Maximum NPU utilization while preserving iGPU for parallel workloads"*——也就是"NPU 跑小模型 + iGPU 跑大模型"

### 目标调度架构

```
opencode ──model(大)────► GPU 服务 :8001 ──► Radeon 8060S (Qwen2.5-72B-Int4)
        └─small_model(小)─► NPU 服务 :8002 ──► XDNA 2 NPU  (Phi-4-mini)
```

两个服务都暴露 OpenAI 兼容 API，opencode 的 `model` / `small_model` 各指向一个端口。

---

## 七、官方文档清单

| 主题 | 链接 |
|------|------|
| Ryzen AI Halo 用户手册 (PDF) | https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/halo/amd-ryzen-ai-halo-user-manual.pdf |
| Ryzen AI Max+ 395 规格页 | https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html |
| Ryzen AI Halo 总览页 | https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo.html |
| Ryzen AI Software 1.8 文档首页 | https://ryzenai.docs.amd.com/en/latest/ |
| Linux 安装指南 | https://ryzenai.docs.amd.com/en/latest/linux.html |
| Linux 上跑 LLM 指南 | https://ryzenai.docs.amd.com/en/latest/llm_linux.html |
| LLM 执行模式对比 | https://ryzenai.docs.amd.com/en/latest/llm/overview.html |
| 1.8 Release Notes | https://ryzenai.docs.amd.com/en/latest/relnotes.html |
| Lemonade Server (OpenAI API) | https://ryzenai.docs.amd.com/en/latest/llm/server_interface.html |
| amdxdna 内核驱动仓库 | https://github.com/amd/xdna-driver |
| AMD AI Engine 技术页 | https://www.amd.com/en/products/adaptive-socs-and-fpgas/technologies/ai-engine.html |
| ROCm 文档 | https://rocm.docs.amd.com/en/latest/ |
| AMD 技术文档库 | https://docs.amd.com/ |

---

## 八、本机实测命令参考

### 确认 NPU 驱动状态

```bash
# 模块信息
modinfo amdxdna

# PCI 设备与驱动绑定
lspci -k -s c6:00.1

# 设备节点
ls -la /dev/accel/accel0

# 内核启动日志
sudo journalctl -k | grep -iE "amdxdna|xdna|17f0"
```

### 确认 GPU VRAM

```bash
cat /sys/class/drm/card*/device/mem_info_vram_total
# 输出 103079215104 字节 ≈ 96 GB
```

### 确认内存配置

```bash
free -h
# total 30Gi (系统侧) + 96Gi (GPU 预留) = 128GB 物理内存
```

### NPU 安装后验证（待执行）

```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
# 期望输出: NPU Strix / aie2p / 6x8
```
