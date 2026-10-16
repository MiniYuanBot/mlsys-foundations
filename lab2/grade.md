# Lab 2 验收清单

## 自动测试（WSL2，`bash tools/check_env.sh --lab 2` 全部 OK）

```bash
cmake --build build/lab2 -j && python -m pytest lab2/tests -q
compute-sanitizer --tool memcheck python -m pytest lab2/tests/test_c_cuda.py -q
```

- [ ] `lab2/tests` 全部 PASS（共 96 项）
- [ ] C1 的 memcheck 为 0 errors
- [ ] 只修改了 `*_student.*`，并把各 `*_DONE` 设为 True

## 性能（插电、高性能模式；每个数字取中位数，bench 已内置重复）

- [ ] A1 ≥ `torch.softmax` 带宽的 80%，A2 ≥ 70%（4096×4096）
- [ ] B2 ≥ SDPA FlashAttention 后端的 70%（D=64，S=4096）
- [ ] B3 causal 耗时 ≤ 非 causal 的 60%
- [ ] B4 D=128 ≥ SDPA 的 90%（已写入 `_block_sizes`）
- [ ] C1 ≥ SDPA 的 70%（D=64，S=4096）
- [ ] C2 自动切分比 `num_splits=1` 快 ≥ 1.5×（B=1，H=8，S=32768），带宽 ≥ memcpy 的 80%

参考实现（RTX 4060 Laptop）：A1 94%、A2 97%、B2 95%、B3 54%、B4 98%、C1 94%、C2 2.3× / 91%。

## 报告 `reports/lab2.md`

- [ ] 环境（`bash tools/check_env.sh --lab 2` 的输出）
- [ ] 手推一遍 online softmax 的正确性证明，即讲义 §2 的自测题 2.2
- [ ] 计算并说明 S=4096、D=64 时，标准 attention 与 FlashAttention 各自的 HBM 读写字节数
- [ ] softmax、attention、decode 三张结果表，以及每部分 README 中提出的问题
- [ ] B4 扫描结果表，并解释最优配置
- [ ] C1：解释“P 不经过 shared memory”具体省下了什么，并给出你的 kernel 的寄存器数（`-DL2_PTXAS_VERBOSE=ON`）
- [ ] 卡住最久的地方

## 可选挑战

- [ ] C3 Triton FA2 反向传播
- [ ] C4 C1 的 cp.async 双缓冲
- [ ] C5 paged decode
