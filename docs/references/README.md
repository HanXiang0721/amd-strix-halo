# AMD 架构与 ROCm 参考文档

## 本地文档

以下文档已整理为 markdown 格式，可直接阅读：

| 文件 | 内容 | 说明 |
|------|------|------|
| [precision-support-matrix.md](precision-support-matrix.md) | **精度支持矩阵** | 各架构矩阵核心精度支持，gfx1151 完整对比，量化方案选择依据 |
| [rocblas-datatypes.md](rocblas-datatypes.md) | rocBLAS 数据类型支持 | GEMM 算子在各架构上的精度支持，rocBLAS vs Triton 路径对比 |
| [rocm-compatibility.md](rocm-compatibility.md) | ROCm 兼容性矩阵 | 各 ROCm 版本支持的 GPU 架构、OS、内核要求 |
| [strix-halo-optimization.md](strix-halo-optimization.md) | Strix Halo 系统优化 | UMA 内存架构、GTT/TTM 调优、BIOS 配置、内核要求 |

## 在线文档

以下白皮书需浏览器访问（需 JavaScript 渲染，无法直接下载）：

### AMD 架构白皮书

- [RDNA 3 白皮书](https://www.amd.com/en/products/graphics/radeon-rx-7000-series.html)
- [CDNA 3 白皮书（MI300X）](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [CDNA 4 白皮书（MI350X）](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)

### ROCm 官方文档

- [精度支持](https://rocm.docs.amd.com/en/latest/reference/precision-support.html) — **最重要**
- [兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [Strix Halo 优化](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [vLLM on ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
- [rocBLAS 数据类型](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/data-type-support.html)

### AMD Ryzen AI（NPU）

- [Ryzen AI 文档](https://ryzenai.docs.amd.com/)
- [Lemonade Server](https://lemonade-server.ai/)

## 阅读顺序

1. **precision-support-matrix.md** — 理解为什么 BF16 最优、FP8 不行
2. **strix-halo-optimization.md** — 理解 APU 共享内存架构和调优方法
3. **rocblas-datatypes.md** — 理解 rocBLAS vs Triton 的性能差异根因
4. **rocm-compatibility.md** — 理解 ROCm 版本和硬件的对应关系

---

*整理日期: 2026-07-30*
