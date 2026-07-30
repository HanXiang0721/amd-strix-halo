# AMD 架构与 ROCm 参考文档索引

## 本地已下载文档

以下文档已下载到本目录，可直接阅读：

| 文件 | 内容 | 来源 |
|------|------|------|
| `rocm-precision-support.txt` | ROCm 精度支持矩阵——各架构（RDNA/CDNA）支持的矩阵核心精度（BF16/FP16/INT8/FP8/INT4） | [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) |
| `rocblas-datatypes.txt` | rocBLAS 数据类型支持——GEMM 等算子在不同架构上支持哪些精度 | [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/data-type-support.html) |
| `rocm-compatibility.txt` | ROCm 兼容性矩阵——各 ROCm 版本支持的 GPU 架构和操作系统 | [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) |
| `strix-halo-optimization.txt` | Strix Halo (RDNA 3.5) 系统优化指南——UMA 内存调优、BIOS 配置、GTT/TTM 设置 | [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html) |

## 在线文档（需要浏览器访问）

以下白皮书需要 JavaScript 渲染，无法直接下载，请在浏览器中访问：

### AMD 架构白皮书

| 文档 | 链接 | 说明 |
|------|------|------|
| RDNA 3 白皮书 | https://www.amd.com/en/products/graphics/radeon-rx-7000-series.html | RDNA 3 架构详情（Strix Halo 基于 RDNA 3.5） |
| CDNA 3 白皮书 | https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html | MI300X 架构详情 |
| CDNA 2 白皮书 | https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html | MI250 架构详情 |

### ROCm 官方文档

| 文档 | 链接 | 说明 |
|------|------|------|
| ROCm 文档首页 | https://rocm.docs.amd.com/ | ROCm 完整文档 |
| 精度支持 | https://rocm.docs.amd.com/en/latest/reference/precision-support.html | **最重要的文档**——决定什么量化方案能用 |
| 兼容性矩阵 | https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html | 各 ROCm 版本支持的 GPU |
| Strix Halo 优化 | https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html | APU 推理优化指南 |
| vLLM on ROCm | https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html | vLLM AMD 部署指南 |
| rocBLAS 数据类型 | https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/data-type-support.html | rocBLAS 支持的精度 |

### AMD Ryzen AI 文档

| 文档 | 链接 | 说明 |
|------|------|------|
| Ryzen AI 文档 | https://ryzenai.docs.amd.com/ | NPU (XDNA2) 开发指南 |
| Lemonade Server | https://lemonade-server.ai/ | 本地 AI 服务框架 |

## 阅读建议

### 第一优先：精度支持文档 (`rocm-precision-support.txt`)

这是最关键的文档。从中可以学到：
- RDNA 3.5 (gfx1151) 支持 BF16/INT8 矩阵核心，**不支持 FP8/INT4 矩阵核心**
- CDNA 3 (MI300X) 支持 FP8 (FNUZ)、BF16、INT8
- 为什么 FP8 在我们的机器上慢——走 Triton 软件路径而非硬件矩阵核心
- 不同架构的矩阵核心吞吐量对比

### 第二优先：Strix Halo 优化 (`strix-halo-optimization.txt`)

从中可以学到：
- APU 共享内存架构（UMA）的特殊性
- BIOS VRAM carve-out 配置对推理性能的影响
- GTT/TTM 内存限制调整方法
- 内核版本要求（≥6.18.4）

### 第三优先：rocBLAS 数据类型 (`rocblas-datatypes.txt`)

从中可以学到：
- rocBLAS 的 GEMM 算子在哪些架构上调优了
- 为什么 BF16 走 rocBLAS 快、FP8 不走 rocBLAS
- AOT 预编译内核 vs JIT 编译的路径选择

### 第四优先：兼容性矩阵 (`rocm-compatibility.txt`)

从中可以学到：
- ROCm 7.x 支持哪些 GPU 架构
- gfx1151 在哪个 ROCm 版本开始支持
- Ubuntu/kernel 版本要求

---

*下载日期: 2026-07-30*
*机器: AMD Ryzen AI MAX+ 395 (Strix Halo) / gfx1151 / RDNA 3.5*
