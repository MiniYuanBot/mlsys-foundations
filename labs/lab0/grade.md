# Lab 0 验收清单

自动测试必须全部通过；性能和报告项逐条自查。全部打勾才算完成本 lab。

## 自动测试

```bash
cmake --build build/lab0 -j && ctest --test-dir build/lab0 --output-on-failure
```

- [ ] `partA`：`test_a_tensor` 全部 PASS
- [ ] `partB`：`test_b_gemm` 全部 PASS
- [ ] `partC`：`test_c_lazy` 全部 PASS
- [ ] 只修改了 `src/*_student.cpp`（以及可选地修改了 `gemm.hpp` 中的分块常量）

## 性能（n = 1024，Release 构建，插电，关闭其他重负载程序）

- [ ] B3 `packed` 单线程 ≥ BLAS 单线程的 50%（挑战：≥ 80%）
- [ ] B4 `parallel` ≥ BLAS 同线程数的 50%
- [ ] `bench_fusion` 中，`affine` 和 `gelu` 两个负载的 `tiled` 都比 `unfused` 快 ≥ 3×
- [ ] 每个数字至少跑 3 次，报告中位数和范围

## 报告 `reports/lab0.md`

- [ ] 环境信息：CPU 型号、核数、缓存大小、编译器版本与参数
- [ ] 写出单核 FP32 峰值的计算过程，并给出 B3 占峰值的百分比
- [ ] 一张 GFLOP/s 对矩阵规模的曲线图（`plot_gemm.py` 生成），含 BLAS 基线和峰值线
- [ ] 为 B1、B2、B3、B4 各写一句话，解释“为什么这一步变快了”，要具体到缓存或寄存器层面
- [ ] Part C 手算流量表，并与 `analyze()` 的结果一致
- [ ] 回答 Part C 的 3 个问题
- [ ] 写出“卡住最久的地方”和最终的解决办法

## 可选挑战（做了就在报告里单列一节）

- [ ] B3d：用 intrinsics 手写微内核
- [ ] B3e：MC/KC 扫描热力图，并用缓存大小做出解释
- [ ] C5：生成 C++ 代码，编译后用 dlopen 加载执行
- [ ] C6：让逐元素运算跨过归约融合
