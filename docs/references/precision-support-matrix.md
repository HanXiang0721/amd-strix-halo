# AMD GPU 矩阵核心精度支持矩阵

> 来源: [ROCm Precision Support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
> 提取日期: 2026-07-30

## 矩阵核心（Matrix Core）精度支持

这是决定量化方案选择的最关键表格。✅=原生硬件支持，❌=不支持（只能走软件路径）。

### 整数类型

| 类型 | CDNA1 (MI100) | CDNA2 (MI250) | CDNA3 (MI300X) | CDNA4 (MI350X) | RDNA2 | RDNA3 | **RDNA3.5 (gfx1151)** | RDNA4 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| int8 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| int16 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| int32 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| int64 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 低精度浮点类型

| 类型 | CDNA1 | CDNA2 | CDNA3 (MI300X) | CDNA4 (MI350X) | RDNA2 | RDNA3 | **RDNA3.5 (gfx1151)** | RDNA4 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| float4 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| float6 (E2M3) | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| float6 (E3M2) | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **float8 (E4M3)** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | **❌** | ✅ |
| **float8 (E5M2)** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | **❌** | ✅ |

### 高精度浮点类型

| 类型 | CDNA1 | CDNA2 | CDNA3 | CDNA4 | RDNA2 | RDNA3 | **RDNA3.5** | RDNA4 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| bfloat16 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| float16 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| tensorfloat32 | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| float32 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## 关键结论（针对我们的机器 gfx1151 / RDNA 3.5）

1. **BF16: ✅ 原生支持** → 走 rocBLAS AOT 预编译内核，带宽效率 84%
2. **FP8 (E4M3/E5M2): ❌ 不支持** → 走 Triton JIT 软件反量化路径，性能极差（0.5 tok/s）
3. **INT8: ✅ 原生支持** → 硬件支持但 vLLM 未接通 rocBLAS INT8 路径
4. **INT4: ❌ 不支持** → GPTQ/AWQ 无法利用矩阵核心
5. **float4: ❌ 不支持** → 未来量化也无法使用

## 与数据中心 GPU 对比

| 精度 | MI300X (CDNA3) | gfx1151 (RDNA3.5) | 影响 |
|------|:---:|:---:|------|
| BF16 | ✅ | ✅ | 两者都能高效推理 |
| FP8 | ✅ | ❌ | MI300X 能用 FP8 加速，我们不能 |
| INT8 | ✅ | ✅ | 两者都支持，但软件栈适配不同 |
| INT4 | ❌ | ❌ | 两者都不支持矩阵核心 |
| float4 | ❌ | ❌ | 两者都不支持 |

## 对量化方案选择的影响

| 量化方案 | 硬件支持 | 软件路径 | 实际性能 |
|---------|---------|---------|---------|
| BF16 (无量化) | ✅ 矩阵核心 | rocBLAS AOT 内核 | 14.4 tok/s (最优) |
| FP8 (E4M3) | ❌ 无矩阵核心 | Triton JIT 软件反量化 | 0.5 tok/s (极慢) |
| GPTQ-Int4 | ❌ 无矩阵核心 | Triton W4A16 (未调优) | 失败 (qzeros 错误) |
| AWQ-Int4 | ❌ 无矩阵核心 | Triton (segfault) | 失败 (bug #37151) |
| INT8 (W8A8) | ✅ 矩阵核心 | vLLM 未接通 | 不可用 (需开发) |

## FP8 格式说明

- **CDNA3 FP8**: 使用 FNUZ 格式（Flush-to-zero, unsigned zero），与 NVIDIA H100 的 FP8 格式不同
- **float8 (E4M3)**: 4位指数 + 3位尾数，精度较高
- **float8 (E5M2)**: 5位指数 + 2位尾数，动态范围较大
- **RDNA 3.5**: 虽然不支持 FP8 矩阵核心，但 HIP 运行时支持 FP8 数据类型存储和转换（软件路径）

---

*来源: https://rocm.docs.amd.com/en/latest/reference/precision-support.html*
*提取日期: 2026-07-30*
