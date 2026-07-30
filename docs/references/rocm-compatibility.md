# ROCm 兼容性矩阵

> 来源: [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

## ROCm 7.x 支持的 GPU 架构

### Instinct (数据中心)

| GPU | gfx 架构 | ROCm 7.14 支持 |
|-----|---------|:---:|
| MI350X / MI355X | gfx950 (CDNA4) | ✅ |
| MI325X | gfx942 (CDNA3) | ✅ |
| MI300X | gfx942 (CDNA3) | ✅ |
| MI300A | gfx942 (CDNA3) | ✅ |
| MI250 / MI250X | gfx90a (CDNA2) | ⚠️ (实验性) |
| MI100 | gfx908 (CDNA1) | ❌ (7.x 不支持) |

### Radeon (消费级/工作站)

| GPU | gfx 架构 | ROCm 7.14 支持 |
|-----|---------|:---:|
| RX 9070 / 9070XT | gfx1201 (RDNA4) | ✅ |
| RX 7900XT / 7900XTX | gfx1100 (RDNA3) | ✅ |
| RX 7800/7700/7600 | gfx1101/gfx1102 (RDNA3) | ✅ |

### Ryzen APU

| APU | gfx 架构 | ROCm 7.14 支持 |
|-----|---------|:---:|
| Ryzen AI MAX+ 395 (Strix Halo) | **gfx1151 (RDNA3.5)** | ✅ |
| Ryzen AI MAX 390 (Strix Halo) | gfx1151 (RDNA3.5) | ✅ |
| Ryzen AI 9 HX 370 (Strix Point) | gfx1150 (RDNA3.5) | ✅ |
| Ryzen AI 7 350 (Kraken Point) | gfx1152 (RDNA3.5) | ✅ |

## 操作系统支持

| OS | ROCm 7.14 支持 |
|----|:---:|
| Ubuntu 24.04 | ✅ |
| Ubuntu 22.04 | ✅ |
| RHEL 9 | ✅ |
| SLES 15 | ✅ |

## 内核要求

| 硬件 | 内核要求 |
|------|---------|
| Instinct (MI300+) | 标准内核 |
| RDNA 3.5 APU (Strix Halo) | **≥6.18.4** 或 Ubuntu HWE ≥6.17.0-19 |
| RDNA 3/4 | 标准内核 |

## ROCm 版本与 gfx1151 支持

| ROCm 版本 | gfx1151 支持 | 说明 |
|-----------|:---:|------|
| 6.4.x | ❌ | 不支持 |
| 7.1.x | ❌ | 仅 hipBLAS 支持 |
| 7.2.0+ | ⚠️ | 首次平台级支持 |
| 7.14.0 | ✅ | 完整支持（当前使用） |

---

*来源: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html*
