# 软硬协同设计：C++ 硬件亲和编程技巧——以 vLLM/ROCm 源码为例

> 从 vLLM 实际源码中提取的硬件亲和（hardware-aware）C++ 编程技巧。
> 这些技巧不是通用 C++ 语法，而是**利用 C++ 特性让代码贴合 GPU 硬件行为**，实现软硬协同优化。
> 所有代码来自 vLLM `csrc/` 目录，标注了文件路径。

## 技巧 1：编译期 GPU 架构检测——同一份代码适配多种 GPU

### 原理

AMD GPU 有多种架构（gfx908=MI100, gfx942=MI300X, gfx1151=RDNA3.5），不同架构的行为差异很大（FP8 格式、warp size、内存一致性模型）。用 C++ 预处理宏在编译期检测架构，为每种架构生成最优代码。

### vLLM 代码：按架构选择 FP8 格式

```cpp
// 文件: csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:65

__device__ __forceinline__ uint8_t rocm_cvt_float_to_fp8_e4m3(float val) {
  // gfx942 (MI300X) 使用 FNUZ 格式
  #if defined(__gfx942__)
    __hip_fp8_e4m3_fnuz fp8_val(val);
  #else
    // 其他架构使用 OCP 标准格式
    __hip_fp8_e4m3 fp8_val(val);
  #endif
  return reinterpret_cast<uint8_t&>(fp8_val);
}
```

```cpp
// 文件: csrc/libtorch_stable/cache_kernels.cu:27

// FNUZ 格式的 FP8 最大值是 224.0，OCP 格式是 448.0
#if defined(__gfx942__)
  constexpr float kFp8ScaleDivisor = 224.f;
#else
  constexpr float kFp8ScaleDivisor = 448.f;
#endif
```

### 为什么是软硬协同

```
gfx942 (MI300X, CDNA3):
  - 硬件支持 FP8 FNUZ 格式（Flush-to-zero, unsigned zero）
  - 最大值 224.0
  - 有 FP8 矩阵核心

gfx1151 (RDNA 3.5, 我们的机器):
  - 硬件不支持 FP8 矩阵核心
  - 使用 OCP 标准格式
  - 最大值 448.0
  - 走 Triton 软件路径
```

编译时 `__gfx942__` 宏由 HIP 编译器根据目标架构自动定义。同一份源码编译为 MI300X 版本时走 FNUZ 路径，编译为 gfx1151 版本时走 OCP 路径——**零运行时开销的硬件适配**。

---

## 技巧 2：运行时架构检测——动态选择 FP8 格式

### 原理

有些架构信息在编译时不知道（比如同一份二进制需要在多种 GPU 上跑），需要运行时检测。用 C++ 的 `gcnArchName` 字符串匹配做动态判断。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/common.cuh:25

static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;  // NVIDIA 只有一种 FP8 格式
#else
  auto* dprops = at::cuda::getCurrentDeviceProperties();
  std::string device_arch = dprops->gcnArchName;
  // gfx94x (MI300X/MI325X) 使用 FNUZ 格式，不是 OCP
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}
```

### 为什么是软硬协同

```
编译时：不知道最终跑在什么 GPU 上
运行时：查询 gcnArchName → 判断是 gfx942 还是 gfx1151
       → 选择正确的 FP8 格式 → 调用对应的内核

这就是为什么 vLLM 的 Docker 镜像能在多种 AMD GPU 上运行：
  rocm/vllm:rocm7.14.0_rdna_  → 编译为 RDNA 版本
  rocm/vllm:rocm7.14.0_cdna_  → 编译为 CDNA 版本
```

---

## 技巧 3：warp size 自适应——NVIDIA 32 vs AMD 64

### 原理

NVIDIA GPU 的 warp 是 32 线程，AMD GPU 的 wavefront 是 64 线程。warp-level reduction 的代码完全不同。vLLM 用 C++ 模板 + `if constexpr` 让同一份代码适配两种 warp size。

### vLLM 代码：编译期和运行时双重检测

```cpp
// 文件: csrc/cuda_compat.h

#ifdef USE_ROCM
struct Utils {
  // 运行时检测（用于 host 代码）
  static __host__ int get_warp_size() {
    int device_id;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&device_id, deviceProp);
    cudaGetDeviceProperties(&deviceProp, device_id);
    return deviceProp.warpSize;  // AMD 返回 64，NVIDIA 返回 32
  }

  // 编译期检测（用于 device 代码）
  static __device__ constexpr int get_warp_size() {
  #ifdef __GFX9__
    return 64;   // AMD CDNA 架构
  #else
    return 32;   // 其他架构
  #endif
  }
};
  #define WARP_SIZE Utils::get_warp_size()
#else
  #define WARP_SIZE 32  // NVIDIA 固定 32
#endif
```

### vLLM 代码：warp reduction 按架构展开

```cpp
// 文件: csrc/libtorch_stable/quantization/fused_kernels/layernorm_utils.cuh:53

template <typename T>
__device__ __forceinline__ void warpMaxReduce(T* val, int tid,
                                              int64_t reduced_elems) {
  static_assert(WARP_SIZE == 32 || WARP_SIZE == 64);
  
  // AMD wavefront=64 时，多一步 64 线程的 reduction
  if constexpr (WARP_SIZE == 64) {
    if (thread_in_warp + 64 < reduced_elems)
      val[tid] = fmaxf(val[tid], val[tid + 64]);
  }
  
  // 以下是 32 线程共有的部分
  if (thread_in_warp + 32 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 32]);
  if (thread_in_warp + 16 < reduced_elems)
    val[tid] = fmaxf(val[tid], val[tid + 16]);
  // ... 继续展开到 8, 4, 2, 1
}
```

### 为什么是软硬协同

```
NVIDIA (warp=32):
  reduction 步骤: 32 → 16 → 8 → 4 → 2 → 1  (5 步)

AMD (wavefront=64):
  reduction 步骤: 64 → 32 → 16 → 8 → 4 → 2 → 1  (6 步)

if constexpr 在编译期消除不匹配的步骤：
  - 编译为 NVIDIA 版本：WARP_SIZE=32，64 那步不存在
  - 编译为 AMD 版本：WARP_SIZE=64，64 那步存在

static_assert 编译期保证：只允许 32 或 64，其他值编译报错
```

---

## 技巧 4：硬件原子指令差异——shfl 指令适配

### 原理

NVIDIA 的 warp shuffle 指令（`__shfl_xor_sync`）需要 sync mask 参数，AMD 的（`__shfl_xor`）不需要。vLLM 用宏抹平这个差异。

### vLLM 代码

```cpp
// 文件: csrc/cuda_compat.h

#ifndef USE_ROCM
  // NVIDIA: 需要 sync mask
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
  // AMD: 不需要 sync mask
  #define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif
```

### vLLM 代码：ROCm 旧版本的 warp 同步

```cpp
// 文件: csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:44

#if defined(HIP_VERSION) && HIP_VERSION < 70000000
// ROCm 7.0 之前没有 __syncwarp，手动实现
__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#endif

// AMD 的 FINAL_MASK 是 64 位（64 线程 wavefront）
#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL
#else
  #define FINAL_MASK 0xffffffff  // NVIDIA 32 位
#endif
```

### 为什么是软硬协同

```
__builtin_amdgcn_fence  — AMD GCN 架构的内存屏障指令
__builtin_amdgcn_wave_barrier — AMD wavefront 级别的同步指令

这两个是 AMD 独有的编译器内建函数（builtin），直接映射到 GCN ISA 指令。
NVIDIA 没有等价物——NVIDIA 用 __syncwarp()。

vLLM 的做法：
  1. 检测 ROCm 版本 < 7.0 → 手动实现 syncwarp（用 amdgcn 内建函数）
  2. ROCm >= 7.0 → 用官方 __syncwarp()
  3. NVIDIA → 直接用 __syncwarp()

这是纯硬件亲和代码——直接调用 GPU 架构的底层指令。
```

---

## 技巧 5：纹理缓存读取——`__ldg` vs 直接读取

### 原理

NVIDIA GPU 有只读数据缓存（texture cache），`__ldg` 指令通过这个缓存读取数据，带宽更高。AMD 没有等价指令，直接读取。vLLM 用宏适配。

### vLLM 代码

```cpp
// 文件: csrc/cuda_compat.h

#ifndef USE_ROCM
  #define VLLM_LDG(arg) __ldg(arg)    // NVIDIA: 走纹理缓存
#else
  #define VLLM_LDG(arg) *(arg)        // AMD: 普通读取
#endif
```

### 实际使用

```cpp
// 文件: csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:255

float cos_val = CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
float sin_val = CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));

// 文件: csrc/libtorch_stable/quantization/activation_kernels.cu:81
const int4 x_128bit = VLLM_LDG(&x_128bit_ptr[vec_idx]);
```

### 为什么是软硬协同

```
NVIDIA __ldg：
  - 数据通过 L1 texture cache 读取
  - 不污染 L1 data cache（数据 cache 给计算用）
  - 带宽更高，延迟更低（对于只读数据）

AMD 普通读取：
  - 走标准 L1 cache
  - 没有 separate texture cache
  - 但 L1 cache 更大（补偿了没有 texture cache 的劣势）

vLLM 用 VLLM_LDG 宏让同一份代码在两种架构上都走最优路径。
```

---

## 技巧 6：`__launch_bounds__` + 编译期 occupancy 计算

### 原理

GPU 的 occupancy（每个 SM 上同时跑多少个 block）取决于寄存器使用量。vLLM 用编译期宏自动计算最优 occupancy，不需要手动猜。

### vLLM 代码：自动计算 blocks per SM

```cpp
// 文件: csrc/libtorch_stable/launch_bounds_utils.h

// 从设备属性获取最大线程数/SM（编译期常量）
// NVIDIA: 通常是 2048，AMD: 通常是 1024

#define VLLM_BLOCKS_DIV(VAL) (VLLM_MAX_THREADS_PER_SM / (VAL))
#define VLLM_CLAMP_BLOCKS_PER_SM(VAL) \
  (((VAL) <= 0) ? 1 \
   : (((VAL) < VLLM_LAUNCH_BLOCKS_CAP) ? (VAL) : VLLM_LAUNCH_BLOCKS_CAP))
#define VLLM_BLOCKS_PER_SM(BLOCK_THREADS) \
  VLLM_CLAMP_BLOCKS_PER_SM(VLLM_BLOCKS_DIV(BLOCK_THREADS))
```

### 实际使用

```cpp
// 文件: csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu:39

__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
void cvt_fp16_to_fp4(...) {
  // 512 线程/block
  // VLLM_BLOCKS_PER_SM(512) = MAX_THREADS / 512
  //   NVIDIA (2048): 2048/512 = 4 blocks/SM
  //   AMD (1024):   1024/512 = 2 blocks/SM
}
```

### 运行时版本

```cpp
// 文件: csrc/libtorch_stable/launch_bounds_utils.h

static inline int vllm_runtime_blocks_per_sm(int block_threads) {
  int device = -1;
  cudaGetDevice(&device);
  int max_threads_per_sm = VLLM_MAX_THREADS_PER_SM;
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor, device);
  int blocks = (block_threads > 0) ? (max_threads_per_sm / block_threads) : 1;
  return VLLM_CLAMP_BLOCKS_PER_SM(blocks);
}
```

### 为什么是软硬协同

```
NVIDIA SM: 最多 2048 线程 → 512线程/block → 4 blocks/SM
AMD CU:    最多 1024 线程 → 512线程/block → 2 blocks/SM

VLLM_BLOCKS_PER_SM 宏在编译期自动算出这个比值
→ __launch_bounds__ 告诉编译器"至少要能跑 X 个 block"
→ 编译器调整寄存器分配满足这个约束
→ 最大化 occupancy，隐藏内存延迟
```

这就是我们之前分析 FP8 Triton 内核 occupancy 低时的核心概念——vLLM 用这套机制主动控制，而 Triton 内核没有这个控制。

---

## 技巧 7：共享内存 union——多种用途复用同一块内存

### 原理

GPU 的共享内存（shared memory）是稀缺资源（通常 48-64KB/SM）。同一块内存在不同执行阶段可以用于不同目的。用 C++ 的 `union` 让多种数据结构复用同一块共享内存。

### vLLM 代码

```cpp
// 文件: csrc/libtorch_stable/sampler.cu:362

__shared__ union {
  FinalItems items;           // 排序阶段用
  FinalSortTempStorage finalSort;  // 最终排序用
  Histogram histo;            // 直方图阶段用
} smemFinal;

// 阶段1: 用 smemFinal.histo 做直方图统计
// 阶段2: 用 smemFinal.items 存排序结果
// 阶段3: 用 smemFinal.finalSort 做最终排序
// 三个阶段复用同一块共享内存，不冲突（不同时使用）
```

### 为什么是软硬协同

```
不用 union:
  __shared__ FinalItems items;        // 4KB
  __shared__ FinalSortTempStorage sort; // 8KB
  __shared__ Histogram histo;         // 4KB
  总共: 16KB → 可能超过共享内存限制

用 union:
  __shared__ union { ... } smemFinal; // max(4KB, 8KB, 4KB) = 8KB
  总共: 8KB → 省了一半

共享内存是 GPU 上最快的内存（比 L1 cache 快 10-20 倍），
省下来的空间可以用来提高 tile size 或增加 block 数量。
```

---

## 技巧 8：CDNA 架构的内存一致性模型——`MUBUF_ACQUIRE`

### 原理

不同 AMD 架构的内存一致性指令不同。CDNA3（gfx942）用 scope bits，CDNA1/2 用 glc bit。vLLM 的 quickreduce 库用预处理宏适配。

### vLLM 代码

```cpp
// 文件: csrc/quickreduce/base.h:22

// 设置向量内存读取（mubuf 指令）的 acquire-release 语义
#if defined(__gfx942__)
  // CDNA3 (MI300X): 用 scope bits sc0, sc1
  #define MUBUF_ACQUIRE 16
  #define MUBUF_RELEASE 16
#elif (defined(__gfx908__) || defined(__gfx90a__))
  // CDNA1 (MI100) 和 CDNA2 (MI250): 用 glc bit
  #define MUBUF_ACQUIRE 1
  #define MUBUF_RELEASE 0
#endif
```

### 为什么是软硬协同

```
MUBUF 是 AMD GCN ISA 的内存读取指令（Memory UnBuffered）。
不同架构对内存一致性的控制方式不同：

gfx942 (CDNA3):
  - 新增了 scope bits (sc0, sc1) 控制内存操作的范围
  - 值 16 = acquire-release 语义
  - 更精细的内存一致性控制

gfx908/gfx90a (CDNA1/2):
  - 只有 glc (globally coherent) 位
  - 值 1 = acquire，0 = release
  - 较粗糙的内存一致性控制

这是最底层的硬件亲和代码——直接操作 GPU ISA 指令的 bit 字段。
```

---

## 技巧 9：AMD 硬件 FP8 转换指令——`__hip_cvt`

### 原理

AMD GPU 有硬件级别的 FP8 转换指令（`__hip_cvt_float_to_fp8` 等），一条指令完成 float→FP8 转换。C++ 代码直接调用这些内建函数，映射到单条 GPU 指令。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/amd/quant_utils.cuh

// float → FP8（单条硬件指令）
__device__ __forceinline__ fp8_type cvt_c10(float const r) {
  // __hip_cvt_float_to_fp8 是 AMD 硬件内建函数
  // 编译为单条 V_CVT_F32_TO_FP8 GCN 指令
  return __hip_cvt_float_to_fp8(r, 
                                fp8_type::__default_saturation,
                                fp8_type::__default_interpret);
}

// float2 → FP8x2（一次转两个，也是单条指令）
__inline__ __device__ uint16_t
vec_conversion<uint16_t, float2>(const float2& a) {
  return __hip_cvt_float2_to_fp8x2(a / scale, 
                                   fp8_type::__default_saturation,
                                   fp8_type::__default_interpret);
}

// FP8 → half（硬件指令）
__inline__ __device__ __half_raw
vec_conversion(const uint8_t& a) {
  return __hip_cvt_fp8_to_halfraw(a, fp8_type::__default_interpret);
}
```

### 为什么是软硬协同

```
软件方式（慢）:
  1. float → 提取符号位、指数、尾数
  2. 按 FP8 格式重新组合
  3. 处理溢出、饱和
  → ~20 条指令

硬件方式（快）:
  __hip_cvt_float_to_fp8(r, ...)
  → 1 条 V_CVT_F32_TO_FP8 指令
  → GPU 的转换单元硬件完成所有步骤

这就是为什么 vLLM 在支持 FP8 的 GPU (MI300X) 上性能好：
  软件调用 __hip_cvt → 编译为硬件指令 → 1 周期完成转换

而在我们的 gfx1151 上：
  没有FP8矩阵核心 → __hip_cvt 可能仍然可用（数据类型转换）
  但矩阵乘法走 Triton 软件路径 → 性能差
```

---

## 技巧 10：`reinterpret_cast` + `__restrict__`——零拷贝 + 别名消除

### 原理

`reinterpret_cast` 告诉编译器"这个地址换个类型看"（零拷贝）。`__restrict__` 告诉编译器"这个指针没有别名"（允许更激进的优化）。两者结合实现零拷贝 + 最大优化。

### vLLM 代码

```cpp
// 文件: csrc/cpu/micro_gemm/cpu_micro_gemm_neon.hpp:275

auto* __restrict__ out = reinterpret_cast<bfloat16_t*>(a_packed);
//                     ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                     别名消除        零拷贝类型重解读
```

### 为什么是软硬协同

```
不用 __restrict__:
  编译器假设 out 可能和其他指针别名（指向同一块内存）
  → 每次写 out 后，之前读的数据可能失效
  → 必须重新从内存加载 → 性能差

用 __restrict__:
  编译器知道 out 不和其他指针别名
  → 之前读的数据不会因为写 out 而失效
  → 可以把数据保持在寄存器中 → 性能好

这在 NEON SIMD 代码中特别重要：
  reinterpret_cast 让 void* 变成 float16_t* 给 SIMD 指令用
  __restrict__ 让编译器可以放心地把数据放在 NEON 寄存器中
```

---

## 技巧 11：编译期 FP8 最大值常量——`quant_type_max_v`

### 原理

不同 FP8 格式的最大值不同（E4M3=448, FNUZ E4M3=224, E5M2=57344）。用 C++ 模板 + `constexpr` 在编译期确定正确的最大值。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/common.cuh:68

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  // quant_type_max_v<fp8_type> 在编译期求值
  // E4M3: 448.0f
  // FNUZ E4M3: 224.0f
  // E5M2: 57344.0f
  float r = fmaxf(-quant_type_max_v<fp8_type>,
                  fminf(x, quant_type_max_v<fp8_type>));
  
  // 硬件转换指令
  return fp8::cvt_c10<fp8_type>(r);
}
```

### 为什么是软硬协同

```
编译期确定最大值 → fmaxf 的操作数是立即数 → 单条指令
运行时确定最大值 → fmaxf 需要先加载常量 → 多一条 load 指令

在 GPU 上，寄存器是稀缺资源。
编译期常量不占寄存器，运行时变量占寄存器。
→ 编译期常量 = 更少寄存器 = 更高 occupancy = 更好性能
```

---

## 技巧 12：动态共享内存——`extern __shared__`

### 原理

GPU 的共享内存可以静态分配（`__shared__ float data[1024]`）或动态分配（`extern __shared__ char raw[]`）。动态分配的大小在 kernel 启动时指定，可以根据输入大小灵活调整。

### vLLM 代码

```cpp
// 文件: csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:339

extern __shared__ char smem_storage[];
// 大小在 launch 时通过 <<<grid, block, shared_mem_size>>> 指定

// 文件: csrc/libtorch_stable/cooperative_topk.cuh:265
extern __shared__ uint8_t smem_raw[];
// 动态共享内存，用 uint8_t 原始字节表示

// 文件: csrc/libtorch_stable/sampler.cu:371
extern __shared__ int32_t smemOutput[];
// 动态共享内存，按 int32_t 使用
```

### 为什么是软硬协同

```
静态共享内存:
  __shared__ float data[1024];  // 大小固定
  → 如果实际只需要 256 个 float，浪费 3KB
  → 如果需要 2048 个，编译报错

动态共享内存:
  extern __shared__ char raw[];  // 大小运行时决定
  // launch: kernel<<<grid, block, needed_bytes>>>
  → 按需分配，不浪费
  → 可以用满共享内存上限（比如 64KB → 提高 tile size）

vLLM 还用 cuda_compat.h 适配了设置动态共享内存大小的 API：
  NVIDIA: cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
  AMD:    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
```

---

## 总结：软硬协同的核心模式

| 模式 | C++ 技巧 | 硬件亲和点 | vLLM 中的场景 |
|------|---------|-----------|--------------|
| 编译期架构检测 | `#ifdef __gfx942__` | FP8 格式、内存一致性指令 | FP8 量化、quickreduce |
| 运行时架构检测 | `gcnArchName` 字符串匹配 | FP8 格式选择 | `is_fp8_ocp()` |
| warp size 适配 | `if constexpr (WARP_SIZE==64)` | NVIDIA 32 vs AMD 64 | warp reduction |
| 指令差异适配 | 宏 `#ifdef USE_ROCM` | shfl/ldg/syncwarp | 全部跨平台代码 |
| 硬件指令调用 | `__hip_cvt_float_to_fp8` | AMD FP8 转换单元 | FP8 量化内核 |
| occupancy 控制 | `__launch_bounds__` + 编译期计算 | SM 寄存器分配 | 所有关键 kernel |
| 共享内存复用 | `__shared__ union` | 共享内存是稀缺资源 | sampler、topk |
| 内存一致性 | `MUBUF_ACQUIRE` bit | CDNA 架构的 scope bits | quickreduce |
| 纹理缓存 | `__ldg` vs `*` | NVIDIA texture cache | 只读数据加载 |
| 编译期常量 | `constexpr` + 模板 | FP8 最大值、tile size | clamp 操作 |
| 零拷贝 + 别名消除 | `reinterpret_cast` + `__restrict__` | 寄存器保持数据 | NEON SIMD |
| 动态共享内存 | `extern __shared__` | 按需分配共享内存 | attention、sampler |

### 软硬协同的思维方式

```
普通 C++ 程序员:
  "这段代码逻辑对不对？"

软硬协同工程师:
  "这段代码逻辑对不对？
   它跑在什么架构上？
   编译器会生成什么指令？
   寄存器够不够？occupancy 高不高？
   内存访问模式是否 coalesced？
   有没有 warp divergence？
   共享内存用了多少？能复用吗？
   走的是硬件指令还是软件模拟？"
```

**这就是软硬协同的本质——写代码时不只考虑逻辑正确，还考虑每一行代码在硬件上的执行行为。**

---

*文档日期: 2026-07-31*
*代码来源: vLLM 项目 csrc/ 目录*
*机器: AMD Ryzen AI MAX+ 395 (Strix Halo) / gfx1151 / RDNA 3.5*
