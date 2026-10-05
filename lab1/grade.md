# Lab 1 验收清单

自动测试必须全部通过；性能和报告项逐条自查。全部打勾才算完成本 lab。

## 自动测试（WSL2）

```bash
cmake --build build/lab1 -j && ctest --test-dir build/lab1 --output-on-failure
for t in test_a_basics test_b_sgemm test_c_hgemm; do compute-sanitizer build/lab1/$t; done
compute-sanitizer --tool racecheck build/lab1/test_b_sgemm B2
```

- [ ] `partA`、`partB`、`partC` 全部 PASS
- [ ] `compute-sanitizer`（memcheck）对三个测试程序都报告 `ERROR SUMMARY: 0 errors`
- [ ] `racecheck` 对 B2 和 C2 报告 0 hazards
- [ ] 只修改了 `src/*_student.cu`

## 性能（插电、高性能模式；以 n = 2048 为准，至少跑 3 次，取中位数）

- [ ] `vector_add`、`reduce_sum` ≥ cudaMemcpy 带宽的 80%
- [ ] `transpose_padded` > `transpose_tiled` > `transpose_naive`（在 `--side 2048` 下）
- [ ] B4 `blocktile_2d` ≥ cuBLAS SGEMM 的 70%（挑战：≥ 90%）
- [ ] C2 `mma` ≥ cuBLAS HGEMM 的 60%（挑战：≥ 80%）

参考实现在 RTX 4060 Laptop 上的结果：B4 约 100%，C2 约 85%。如果你用的是其他 GPU，目标按“占 cuBLAS 的百分比”计算，不按绝对的 TFLOP/s。

## 报告 `reports/lab1.md`

- [ ] 环境：GPU 型号、SM 数、L2 大小、显存带宽、驱动和 CUDA 版本，以及满载时 `nvidia-smi` 看到的频率与功耗
- [ ] 写出 FP32 峰值和 FP16 Tensor 峰值的计算过程，并给出 B4、C2 各自占峰值的百分比
- [ ] 阶梯图：`plot_gemm.py` 生成的 TFLOP/s 曲线和 % of cuBLAS 柱状图
- [ ] 为 B1、B2、B3、B4、C1、C2 各写一句话，解释“为什么这一步变快了（或者为什么没快多少）”，要用到“每次 FMA 读几次内存”或具体的硬件资源
- [ ] 解释 transpose 在 `--side 8192` 和 `--side 2048` 下表现不同的原因
- [ ] 至少一个 Nsight Compute 观察（bank conflict、occupancy、寄存器数等），附截图或数字
- [ ] C2：实测把 STAGES 改为 2 的结果，以及去掉 padding（`+8`）的结果，并给出解释
- [ ] 写出“卡住最久的地方”和最终的解决办法

## 可选挑战

- [ ] C3：用 XOR swizzle 代替 padding
- [ ] C4：任意形状（predicated load 或边界 kernel）
- [ ] C5：在 H100 上实现 TMA + WGMMA 版本
- [ ] C6：提交到 LeetGPU 或 GPU MODE 排行榜
