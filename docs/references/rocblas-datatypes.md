# rocBLAS 数据类型支持

> 来源: [rocBLAS Data Type Support](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/data-type-support.html)

## GEMM 矩阵乘法支持

rocBLAS 的 Level-3 BLAS（GEMM 等）在不同架构上支持的精度：

### 矩阵核心（Matrix Core）GEMM

| 数据类型 | gfx908 (CDNA1) | gfx90a (CDNA2) | gfx942 (CDNA3) | gfx950 (CDNA4) | gfx1151 (RDNA3.5) |
|---------|:---:|:---:|:---:|:---:|:---:|
| BF16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP8 (E4M3) | ❌ | ❌ | ✅ | ✅ | ❌ |
| FP8 (E5M2) | ❌ | ❌ | ✅ | ✅ | ❌ |
| INT8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP32 | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP64 | ✅ | ✅ | ✅ | ✅ | ✅ |

### 关键结论

1. **gfx1151 上 BF16 和 FP16 的 GEMM 走 rocBLAS 矩阵核心** → AOT 预编译，有 per-arch 调优配置
2. **FP8 在 gfx1151 上不走 rocBLAS** → 无矩阵核心支持，只能走 Triton JIT 软件路径
3. **INT8 在 gfx1151 上 rocBLAS 支持** → 但 vLLM 未将 INT8 W8A8 量化路径接通到 rocBLAS
4. **rocBLAS 的 GEMM 内核是 AOT 编译的** → 每个 gfx 架构有专属的 `.hsaco` 内核文件

### rocBLAS vs Triton 路径对比

| 特性 | rocBLAS | Triton |
|------|---------|--------|
| 编译方式 | AOT（提前编译） | JIT（即时编译） |
| 内核来源 | AMD 官方预编译 `.hsaco` | 运行时从 Python 生成 |
| 调优状态 | per-arch 调优（有 gfx1151 配置） | 未调优（默认 tile size） |
| 带宽效率 | 84%（BF16 on gfx1151） | ~6%（FP8 on gfx1151） |
| 首次运行 | 无延迟 | 需要编译（首次慢） |
| 缓存 | 无需缓存 | `~/.triton/cache` |

### 如何确认走的是 rocBLAS

```bash
# 查看 rocBLAS 内核目录
ls /opt/rocm/lib/rocblas/library/ | grep gfx1151

# 查看内核数量
ls /opt/rocm/lib/rocblas/library/ | grep gfx1151 | wc -l
```

---

*来源: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/data-type-support.html*
