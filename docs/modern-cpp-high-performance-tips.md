# 现代 C++ 高性能编程技巧——以 vLLM 源码为例

> 从 vLLM 实际源码中提取的高性能 C++ 技巧，每个技巧配有真实代码、原理解释和性能影响分析。
> 目标：读完能理解 vLLM 量化内核为什么这样写，并能应用到自己写的 GPU kernel 中。

## 技巧 1：union 类型双关——零拷贝类型转换

### 原理

FP8 数据在内存中是 1 字节，但 GPU 的类型转换指令需要特定类型输入。用 `union` 让同一块内存按不同类型解读，**不产生任何内存拷贝**。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/amd/quant_utils.cuh

// FP8 -> float2 转换：两个 FP8 打包在 uint16_t 里
template <>
__inline__ __device__ float2
vec_conversion<float2, uint16_t>(const uint16_t& a) {
  fp8x2_type f8x2;
  f8x2.__x = a;        // 直接把 uint16_t 的 bit pattern 塞进 fp8x2
  return static_cast<float2>(f8x2);  // 硬件指令转换
}

// half2 -> fp8x2 转换：用 union 做 bit-level reinterpret
template <>
__inline__ __device__ uint16_t
vec_conversion<uint16_t, uint32_t>(const uint32_t& a) {
  union {
    uint32_t ui32;
    __half2_raw h2r;
  } tmp;
  tmp.ui32 = a;        // uint32_t 的 bit 直接当 half2_raw 用
  return __hip_cvt_halfraw2_to_fp8x2(tmp.h2r, ...);
}
```

### 为什么高性能

```cpp
// 慢的写法：拷贝内存
uint32_t a = ...;
__half2_raw h2r;
memcpy(&h2r, &a, sizeof(uint32_t));  // 产生一次内存拷贝
```

```cpp
// 快的写法：union 类型双关，零拷贝
union { uint32_t ui32; __half2_raw h2r; } tmp;
tmp.ui32 = a;  // 编译器知道同一块内存，不产生拷贝
```

union 的两个成员**共享同一块内存**，赋值 `ui32` 后 `h2r` 自动指向相同数据。编译器优化为零开销。

### 实际应用场景

FP8 量化时需要大量类型转换（FP8↔FP16↔BF16↔FP32），每次转换如果产生内存拷贝会在 GPU 上累积成巨大开销。vLLM 的 FP8 内核中**每个转换都用 union**，这是 FP8 量化能高效运行的基础。

---

## 技巧 2：向量化类型转换——一次处理多个元素

### 原理

GPU 的内存带宽是瓶颈。一次读 1 个 FP8（1 字节）和一次读 4 个 FP8（4 字节，打包在 uint32_t 里）的访存延迟几乎一样。把数据打包成更宽的类型，**用更少的指令处理更多的数据**。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/amd/quant_utils.cuh

// 1个 FP8 -> 1个 float
template <>
__inline__ __device__ float vec_conversion<float, uint8_t>(const uint8_t& a) {
  fp8_type f8;
  f8.__x = a;
  return static_cast<float>(f8);
}

// 4个 FP8 -> 1个 float4（一次处理 4 个元素）
template <>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(const uint32_t& a) {
  Float4_ tmp = vec_conversion<Float4_, uint32_t>(a);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

// 8个 FP8 -> 1个 Float8_（一次处理 8 个元素）
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(const uint2& a) {
  Float4_ tmp1, tmp2;
  tmp1 = vec_conversion<Float4_, uint32_t>(a.x);  // 前4个
  tmp2 = vec_conversion<Float4_, uint32_t>(a.y);  // 后4个
  Float8_ res;
  res.x = tmp1.x; res.y = tmp1.y;
  res.z = tmp2.x; res.w = tmp2.y;
  return res;
}
```

### 为什么高性能

```
标量方式：8 次 1 字节读取 = 8 次内存事务
向量化：1 次 8 字节读取 = 1 次内存事务

GPU 内存事务的最小粒度通常是 32 字节（一个 warp 的一次访存）
8 个 FP8 = 8 字节，远小于 32 字节
→ 标量方式浪费 75% 的带宽
→ 向量化方式接近 100% 利用
```

vLLM 用模板特化实现：同一个 `vec_conversion` 函数名，根据输入类型自动选择处理 1/2/4/8 个元素。这就是模板在 GPU 编程中的实际价值——**零开销的代码复用**。

---

## 技巧 3：`if constexpr` 消除运行时分支

### 原理

普通 `if` 在 GPU 上有代价：两个分支的代码都会生成，占用寄存器和指令缓存，分支预测失败时还有流水线气泡。`if constexpr` 在编译期消除分支，不满足的分支**完全不生成机器码**。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/common.cuh:61

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;   // 乘法
  } else {
    x = val / scale;   // 除法（比乘法慢 3-5 倍）
  }
  // ...
}
```

### 为什么高性能

```cpp
// 普通 if：两个分支都编译进二进制
if (is_scale_inverted) { x = val * scale; }    // 生成乘法指令
else                    { x = val / scale; }    // 生成除法指令
// 运行时：分支预测 + 可能的流水线气泡
// 寄存器：两个分支的临时变量都占寄存器

// if constexpr：只编译一个分支
if constexpr (is_scale_inverted) { x = val * scale; }  // 只有乘法
else                             { x = val / scale; }  // 完全不生成
// 无分支预测开销，无寄存器浪费
```

GPU 的分支粒度是 warp（32 线程 / AMD 64 线程），warp 内部分歧（divergence）会导致串行执行。`if constexpr` 在编译期消除分支，**从根源上消灭 warp divergence**。

### 更多 vLLM 例子

```cpp
// 文件: csrc/quantization/w8a8/fp8/amd/quant_utils.cuh:619
// FP8 KV Cache 按精度格式走不同转换路径

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale);
  }
  // 如果不是 E4M3，这行代码根本不存在
}
```

---

## 技巧 4：`__launch_bounds__` 控制 occupancy

### 原理

GPU 的 occupancy（占用率）= 每个计算单元（SM/CU）上同时能跑多少个 block。occupancy 太低→内存延迟无法隐藏→性能差。太高→寄存器不够→溢出到慢内存。`__launch_bounds__` 让程序员精确控制这个平衡。

### vLLM 代码

```cpp
// 文件: csrc/libtorch_stable/sampler.cu:545

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock)
void topKPerRowPrefill(...) {
  static constexpr int kNumBins = 2048;
  // ...
}
```

```cpp
// 文件: csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu:39

__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
void nvfp4_experts_quant_kernel(...) {
```

### 参数解释

```cpp
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
```

| 参数 | 含义 | 作用 |
|------|------|------|
| `maxThreadsPerBlock` | 每个 block 最多多少线程 | 编译器据此分配寄存器 |
| `minBlocksPerMultiprocessor` | 每个 SM 至少要跑几个 block | 编译器会减少寄存器使用量来满足 |

### 为什么高性能

```
不设 launch_bounds：
  编译器默认每个线程用很多寄存器 → 每个 SM 只能跑 2 个 block → occupancy 低

设 launch_bounds(512, VLLM_BLOCKS_PER_SM(512))：
  告诉编译器"每个 SM 至少要跑 X 个 block"
  → 编译器减少每线程寄存器使用 → 更多 block 能同时跑 → occupancy 提高
  → 内存延迟被更多并行计算隐藏 → 性能提升
```

这就是我们之前分析 FP8 Triton 内核 occupancy 低时的核心概念——vLLM 的 CUDA 内核用 `__launch_bounds__` 主动控制这个平衡。

---

## 技巧 5：`#pragma unroll` 强制循环展开

### 原理

GPU 没有 branch predictor，循环的每次迭代都要判断条件 + 跳转。`#pragma unroll` 让编译器把循环体复制 N 次，**消除循环开销，暴露指令级并行**。

### vLLM 代码

```cpp
// 文件: csrc/libtorch_stable/fused_qknorm_rope_kernel.cu:76

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}
```

```cpp
// 文件: csrc/cpu/mla_decode.cpp:64

#pragma unroll
for (int unroll = 0; unroll < HEAD_UNROLL; ++unroll) {
  // MLA attention 的核心计算循环
  sum_exp[unroll] += val;
}
```

### 为什么高性能

```cpp
// 不展开：4 次迭代，每次有条件判断 + 跳转
for (int i = 0; i < 4; i++) {
  val += data[i];  // 4 次条件判断 + 4 次加法
}

// 展开后：4 次加法，无条件判断
val += data[0];
val += data[1];
val += data[2];
val += data[3];
// 编译器还能做指令调度：让 4 次加法并行执行
```

warpReduceSum 的展开特别重要：`mask` 从 16 递减到 1，只有 5 次迭代（16→8→4→2→1），展开后是 5 条 `__shfl_xor_sync` 指令，没有循环开销。这是 GPU 上 warp-level reduction 的标准写法。

---

## 技巧 6：branchless 编程——用数学函数替代 if

### 原理

GPU 的 warp divergence 代价很高——同一个 warp 里如果一半线程走 if、一半走 else，两组会**串行执行**。用 `fmin/fmax` 等数学函数替代 if，所有线程走同一条路径。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/common.cuh:68

float r = fmaxf(-quant_type_max_v<fp8_type>,
                fminf(x, quant_type_max_v<fp8_type>));
```

### 为什么高性能

```cpp
// 有分支：warp divergence
if (x > max_val) {
  x = max_val;
} else if (x < -max_val) {
  x = -max_val;
}
// warp 内不同线程走不同分支 → 串行执行

// branchless：所有线程走同一条路径
float r = fmaxf(-max_val, fminf(x, max_val));
// fminf/fmaxf 是单条 GPU 指令，无分支
```

FP8 量化必须做 clamp（把超出范围的值截断到 [-448, 448]），这个操作在每个 token 的每个元素上都要做。用 branchless 写法避免 warp divergence，在高吞吐场景下性能差异可达 2-3 倍。

---

## 技巧 7：`atomicMaxFloat`——用整数原子操作实现浮点原子

### 原理

GPU 原子操作只支持整数类型（`atomicAdd`, `atomicMax` 等），不支持浮点。但浮点数的 bit pattern 和整数有对应关系（正浮点数的 bit 排序和数值排序一致），可以用整数原子操作模拟浮点原子。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/common.cuh:43

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}
```

### 为什么高性能

```cpp
// 正浮点数：bit pattern 和数值排序一致
// float 1.0 = 0x3F800000
// float 2.0 = 0x40000000
// → 直接用 atomicMax 比较整数就等于比较浮点数

// 负浮点数：bit pattern 和数值排序相反
// float -1.0 = 0xBF800000
// float -2.0 = 0xC0000000
// → 用 atomicMin（符号位反转后大小关系变了）
```

`__float_as_int` 和 `__int_as_float` 是**零开销的 bit reinterpret**（编译成 0 条指令，只是告诉编译器换一种类型解读寄存器里的值）。这是 GPU 编程中常用的"类型双关"技巧，比 memcpy 快。

---

## 技巧 8：`std::move` 避免大对象拷贝

### 原理

`std::move` 把对象的所有权"移动"而不是拷贝。对于 `torch::Tensor` 这样的大对象，移动只是传指针，拷贝要复制整个数据。

### vLLM 代码

```cpp
// 文件: csrc/cpu/sgl-kernels/gemm_int4.cpp:755

return std::make_tuple(
    std::move(blocked_weight),    // 移动 tensor，不拷贝数据
    std::move(blocked_scales),
    std::move(blocked_qzeros)
);
```

### 为什么高性能

```cpp
// 不用 move：返回时拷贝 3 个 tensor 的数据
return std::make_tuple(blocked_weight, blocked_scales, blocked_qzeros);
// 每个 tensor 可能几百 MB → 3 次大拷贝

// 用 move：只移动 tensor 的元数据（指针、shape、stride）
return std::make_tuple(std::move(blocked_weight), ...);
// 几十字节的指针传递，零数据拷贝
```

在 INT4 量化预处理中，weight tensor 可能有几百 MB。`std::move` 避免了返回时的数据拷贝，这是 C++11 move 语义在高性能计算中的直接价值。

---

## 技巧 9：编译期断言 `static_assert`

### 原理

在编译时检查条件，不满足直接编译失败。比运行时 assert 早发现问题，零运行时开销。

### vLLM 代码

```cpp
// 文件: csrc/cpu/micro_gemm/cpu_micro_gemm_impl.hpp:43

static_assert(n_size % 16 == 0);
// 如果 n_size 不是 16 的倍数，编译直接报错
// 不会等到运行时才发现问题

// 文件: csrc/cpu/cpu_attn_rvv.hpp:278
static_assert(HeadDim % HeadDimAlignment == 0);
// 确保 head dimension 满足对齐要求
```

### 为什么高性能

```cpp
// 运行时检查：有性能开销
assert(n_size % 16 == 0);  // 运行时判断 + 可能的跳转

// 编译期检查：零开销
static_assert(n_size % 16 == 0);  // 编译时就确定了，运行时不存在
```

在 GPU 编程中，很多约束（对齐、维度整除、常量范围）在编译期就能验证。`static_assert` 把错误提前到编译阶段，同时不产生任何运行时代码。

---

## 技巧 10：`__builtin_prefetch` 预取数据

### 原理

CPU 计算和内存访问可以并行。提前发出预取指令，让数据在需要时已经在 cache 里，**隐藏内存延迟**。

### vLLM 代码

```cpp
// 文件: csrc/cpu/cpu_types_scalar.hpp:480

inline void prefetch(const void* addr) {
  __builtin_prefetch(addr, 0, 3);
  // 参数：地址, 读/写(0=读), 局部性级别(0-3, 3=尽量留在 cache)
}

// 文件: csrc/cpu/micro_gemm/cpu_micro_gemm_amx.hpp:111
// 注释说明预取策略：
// TILE 0, (1): load A matrix, use extra 1 tile for prefetch
// TILE 2, 3, (4, 5): load B matrix, use extra 2 tiles for prefetch
```

### 为什么高性能

```
不预取：
  计算A → 等数据B加载 → 计算B → 等数据C加载 → 计算C
  GPU/CPU 大量空闲等待

预取：
  计算A + 预取B → 计算B + 预取C → 计算C + 预取D
  计算和访存重叠，吞吐量翻倍
```

vLLM 的 AMX（Intel 高级矩阵扩展）GEMM 内核用预取实现计算和访存的流水线重叠——在计算当前 tile 的同时预取下一个 tile 的数据。这是 CPU 高性能 GEMM 的标准优化技巧。

---

## 技巧 11：跨平台编译期分支——`#ifdef` + 模板

### 原理

vLLM 需要同时支持 NVIDIA CUDA 和 AMD ROCm。用预处理宏在编译期选择平台特定的代码路径，零运行时开销。

### vLLM 代码

```cpp
// 文件: csrc/quantization/w8a8/fp8/common.cuh

#ifndef USE_ROCM
  #include "nvidia/quant_utils.cuh"   // NVIDIA 路径
#else
  #include "amd/quant_utils.cuh"      // AMD 路径
#endif

// 运行时检测 FP8 格式（ROCm 有两种 FP8）
static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;  // NVIDIA 只有一种 FP8
#else
  auto* dprops = at::cuda::getCurrentDeviceProperties();
  std::string device_arch = dprops->gcnArchName;
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;  // gfx94x 用 FNUZ 格式
#endif
}
```

```cpp
// 文件: csrc/cuda_utils.h

#if defined(__HIPCC__)
  #define HOST_DEVICE_INLINE __host__ __device__
  #define DEVICE_INLINE __device__
#elif defined(__CUDACC__)
  #define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
  #define DEVICE_INLINE __device__ __forceinline__
#else
  #define HOST_DEVICE_INLINE inline
  #define DEVICE_INLINE inline
#endif
```

### 为什么高性能

- `#ifdef` 在预处理阶段就删除了不用的代码——AMD GPU 上完全看不到 NVIDIA 的代码，反之亦然
- `is_fp8_ocp()` 的 `#ifndef USE_ROCM` 分支在 NVIDIA 上编译为 `return true;`（一行指令），在 AMD 上编译为运行时检测
- 这就是为什么 vLLM 能用同一份代码库支持两种 GPU——**编译期分支，零运行时开销**

---

## 技巧 12：`reinterpret_cast` 零拷贝类型重解读

### 原理

和 union 类似，`reinterpret_cast` 告诉编译器"这块内存换个类型看"，不产生任何拷贝。常用于 SIMD 内 Intrinsics 函数需要特定类型指针的场景。

### vLLM 代码

```cpp
// 文件: csrc/cpu/micro_gemm/cpu_micro_gemm_int8_neon.hpp:34

// NEON SIMD 指令需要 float16_t* 但 input 是 void*
const auto input_vec = vld1q_f16(reinterpret_cast<const float16_t*>(input));

// 文件: csrc/cpu/micro_gemm/cpu_micro_gemm_neon.hpp:275
auto* __restrict__ out = reinterpret_cast<bfloat16_t*>(a_packed);
```

### 为什么高性能

```cpp
// 慢：拷贝到新类型
float16_t* buf = new float16_t[n];
memcpy(buf, input, n * sizeof(float16_t));
vld1q_f16(buf);

// 快：reinterpret_cast，零拷贝
vld1q_f16(reinterpret_cast<const float16_t*>(input));
// 只是告诉编译器把 input 的地址当 float16_t* 用
```

`__restrict__` 关键字也很重要——告诉编译器"这个指针指向的内存没有别的指针别名"，编译器可以更激进地优化（不用担心别名导致的问题）。

---

## 技巧 13：`constexpr` 编译期计算消除运行时开销

### 原理

`constexpr` 函数在编译期求值，结果直接嵌入到机器码中作为常量。运行时零计算。

### vLLM 代码

```cpp
// 文件: csrc/cuda_utils.h

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}
```

```cpp
// 文件: csrc/libtorch_stable/sampler.cu

static constexpr int kNumBins = 2048;  // 编译期常量，不占运行时内存
```

### 为什么高性能

```cpp
// 运行时计算：每次调用都算
int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

// 编译期计算：结果直接嵌入机器码
constexpr int num_tiles = ceil_div(N, TILE_SIZE);
// 编译后变成：mov r0, #42  （直接是常量）
```

在 GPU 内核中，tile size、bin 数量、warp size 等参数在编译时就确定了。用 `constexpr` 让这些值成为立即数，不占用寄存器，不产生运行时计算。

---

## 总结：技巧与性能影响

| 技巧 | 性能影响 | 适用场景 | vLLM 使用频率 |
|------|---------|---------|:---:|
| union 类型双关 | 消除内存拷贝 | 类型转换（FP8/FP16/BF16） | 极高 |
| 向量化转换 | 带宽利用率提升 4-8 倍 | 批量数据处理 | 极高 |
| `if constexpr` | 消除分支 + 减少寄存器占用 | 按类型/精度走不同路径 | 1012 处 |
| `__launch_bounds__` | 控制 occupancy | 所有性能关键 kernel | 关键 kernel 全用 |
| `#pragma unroll` | 消除循环开销 + 指令并行 | 小循环（已知次数） | 高频 |
| branchless | 消除 warp divergence | clamp/select 操作 | 量化内核 |
| 原子操作技巧 | 实现浮点原子 | 并行 reduction | 中频 |
| `std::move` | 避免大对象拷贝 | Tensor 传递 | 中频 |
| `static_assert` | 编译期验证，零开销 | 维度/对齐约束 | 高频 |
| 预取 | 隐藏内存延迟 | CPU GEMM | CPU 路径 |
| `#ifdef` 跨平台 | 编译期分支 | CUDA/ROCm 兼容 | 贯穿全部 |
| `reinterpret_cast` | 零拷贝类型重解读 | SIMD intrinsics | CPU 路径 |
| `constexpr` | 编译期计算 | 常量定义 | 2985 处 |

### 学习建议

1. **先理解 union 类型双关和向量化转换**——这是 FP8 量化内核的基础，也是 GPU 编程和 CPU 编程通用的技巧
2. **然后学 `if constexpr` 和 `__launch_bounds__`**——这是控制 GPU 内核行为的核心手段
3. **最后学 branchless 和 `#pragma unroll`**——这是微优化层面的技巧

每个技巧都能在我们这台机器上验证：比如 union 在 FP8 转换中的效果、`if constexpr` 在 BF16 vs FP8 路径选择中的效果、`__launch_bounds__` 对 occupancy 的影响。

---

*文档日期: 2026-07-31*
*代码来源: vLLM 项目 csrc/ 目录*
*机器: AMD Ryzen AI MAX+ 395 (Strix Halo) / gfx1151 / RDNA 3.5*
