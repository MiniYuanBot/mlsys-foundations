# 脚手架验证范围

本页记录**脚手架**（提供的代码、测试与 benchmark）在什么环境下验证过。它不是学生完成 lab 的证明。

验证方法：作者先在仓库外写出完整的参考实现，确认全部测试通过、性能达到 README 中写的数字，再把学生部分替换成抛出 `TODO ... is not implemented yet` 的桩，并确认此时测试全部失败。参考实现不随仓库分发。

## lab0（2026-10-06）

| 检查 | 环境 | 结果 |
|---|---|---|
| 构建 | Windows 11，MinGW-w64 GCC 13.2，CMake 3.29，`-O3 -march=native -fopenmp-simd` | 无警告 |
| 参考实现 + 测试 | 同上 | partA/partB/partC 全部通过 |
| 学生桩 + 测试 | 同上 | partA/partB/partC 全部失败（只有 `B0_naive_reference_sanity` 通过，这是预期的） |
| `bench_gemm` 参考实现 | Intel i9 Alder Lake（24 线程），n=1024 | naive 0.85，reordered 31，blocked 47，packed 113，parallel(24t) 369；n=2048 时 parallel 605 GFLOP/s |
| BLAS 基线 | NumPy 2.5 + MKL 2025 | 1 线程 129，24 线程 650 GFLOP/s |
| `bench_fusion` 参考实现 | 同上，n=2^24 | affine：unfused 340 ms，tiled 37 ms；gelu：unfused 2.7 s，tiled 0.79 s |
| 参考实现 + 测试 | WSL2 Ubuntu 24.04，GCC 13.3，CMake 3.28 | 全部通过；`bench_gemm` n=1024：packed 123，parallel(24t) 494 GFLOP/s |
| `MT_SANITIZE=ON`（ASan + UBSan） | 同上 | 全部通过，无报告 |

**未验证**：MSVC、Apple clang（arm64/NEON）。如果你在这些平台上遇到问题，请记录编译器版本与报错。

## 尚未重构的 lab

`legacy/` 下是重构前的 lab1–lab7，只用于迁移参考，不保证可以运行。
