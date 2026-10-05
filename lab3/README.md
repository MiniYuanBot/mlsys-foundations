# Lab 3：tinycomp —— 一个会生成 CUDA 代码的张量编译器，以及 mini-Mirage

> 配套讲义：[L3 深度学习编译器与超优化](../../mlsys-research-course/lectures/L3-compilers.md)
> 预计用时：2 周 × 约 7 小时　·　语言：C++17（生成 CUDA C++，由 NVRTC 在运行时编译）　·　硬件：RTX 4060（WSL2）

lab0 Part C 里，你用解释器执行了融合后的计算图。真实的编译器（XLA、TorchInductor、TVM、tinygrad）会更进一步：**为每个融合后的 kernel 生成源代码，编译后在 GPU 上运行**。这个 lab 把这条链路完整走一遍，最后再往前走一步，做超优化器 Mirage 所做的事：不只按规则融合，而是**改写计算图本身**，验证改写后的图与原图等价，再从所有候选中选出最快的。

```
Graph --A1 toposort--> --A2 evaluate (CPU, fp64: the oracle)
      --B1 partition--> Plan (kernels) --A3/B2/B3 codegen--> CUDA source --NVRTC--> GPU
      --C1 rewrite--> candidates --C2 verify--> --C3 cost model + search--> best Plan
```

| 部分 | 你要写什么 | 难度梯度 | 对应的真实系统 |
|---|---|---|---|
| **A** 前端与参考执行 | 拓扑排序、fp64 解释器（广播、归约、matmul）、逐元素代码生成 | ① 填空，熟悉 IR | 所有编译器的 IR 与参考执行器 |
| **B** 融合与代码生成 | 融合划分器（逐元素 / 行归约 / matmul 序言与尾声）、行归约 kernel 生成、GEMM 模板的钩子 | ② 核心实现 + 性能目标 | TorchInductor 的调度器、XLA fusion、TVM 的 epilogue fusion |
| **C** mini-Mirage | 改写规则、基于随机测试的概率等价验证、代价模型、广度优先搜索 | ③ 前沿精简复现 | [Mirage](https://arxiv.org/abs/2405.05751)（OSDI'25）、TASO/PET 等超优化器 |

做完后你应当能回答：

- 融合省下的是什么？什么样的节点**不能**融合？
- 为什么规则融合做不出 FlashAttention，而搜索可以？
- 代价模型与实测排序为什么会不一致？

---

## 0. 构建

这个 lab 是纯 C++，不需要 nvcc：生成的 CUDA 代码由 NVRTC 在运行时编译（`tools/check_env.sh` 已经确认过 NVRTC 可用）。

```bash
cmake -S lab3 -B build/lab3 && cmake --build build/lab3 -j
ctest --test-dir build/lab3 --output-on-failure      # all fail at first
build/lab3/test_b B2                                 # only tests whose name contains B2
build/lab3/tc_dump softmax                           # graph, plans and generated CUDA source
build/lab3/tc_dump softmax_matmul --search           # Part C: every candidate with its cost
```

生成的代码如果编译失败，NVRTC 的错误信息和完整源码会一起作为异常抛出，像看普通编译错误一样读它就行。

## 1. 代码地图

```
lab3/
+-- include/tc/
|   +-- ir.hpp             Graph, Node, Op, Shape (2-D tensors, 2-D broadcasting)
|   +-- compiler.hpp       the whole pipeline: read this first
|   +-- codegen_detail.hpp helpers shared by provided and student codegen
|   +-- models.hpp         softmax, layernorm, rmsnorm, matmul_bias_gelu, rmsnorm_matmul, softmax_matmul
+-- src/
|   +-- ir.cpp plan_core.cpp codegen_core.cpp runtime.cpp     provided
|   +-- interp_student.cpp    TODO(A1, A2)
|   +-- codegen_student.cpp   TODO(A3a, A3b, B2, B3)
|   +-- fusion_student.cpp    TODO(B1a, B1b)
|   +-- search_student.cpp    TODO(C1, C2, C3a, C3b)
+-- tests/  tools/tc_dump.cpp  bench/bench_fusion.cpp  bench/bench_search.cpp
```

---

## Part A：前端与参考执行（第 1 周前半，约 2 小时）

| TODO | 内容 | 要点 |
|---|---|---|
| A1 | `toposort` | 与 lab0 C1 相同 |
| A2 | `evaluate` | fp64 精度；二维广播规则是“大小为 1 的维度取下标 0” |
| A3a | `emit_op` | 每个运算对应一个 C 表达式；二元运算加括号 |
| A3b | `emit_load` | 按操作数形状生成带广播的下标表达式 |

A3 完成后，`plan_unfused`（每个节点一个 kernel）就能在 GPU 上跑通。`test_a A3_unfused_plans_run_on_gpu` 会把 GPU 结果与你的 A2 解释器对比。所以 A2 必须是正确的，**后面所有部分都把它当作“标准答案”**。

## Part B：融合与代码生成（第 1 周后半 + 第 2 周前半，约 6 小时）

读讲义 §2–§4。三类 kernel：

| 类型 | 执行方式 | 能装进什么 |
|---|---|---|
| Elementwise | 每个线程算一个输出元素 | 一串逐元素运算 |
| Row | 每个 block 处理一行；列方向用循环；归约用 `tc_block_sum/max` | 逐元素运算 + 若干行归约（softmax、layernorm 都能装进**一个** kernel） |
| MatMul | 提供的 64×64 GEMM 模板，加上你生成的三个钩子 | 一个 matmul + A 一侧的逐元素序言 + 输出一侧的逐元素尾声 |

| TODO | 内容 | 要点 |
|---|---|---|
| B1a | `can_absorb` | 融合规则：判断节点 n 能否加入它的消费者所在的 kernel |
| B1b | `plan_fused` | 按逆拓扑序“吸收”到消费者所在的 kernel；最后按创建顺序的逆序输出 |
| B2 | `codegen_row` | 每个归约一个列循环；循环里需要的“整行”值就地重算 |
| B3 | `matmul_hooks` | 生成 `tc_load_a`、`tc_load_b`、`tc_epilogue` 三个函数体 |

调试顺序：先用 `tc_dump` 看 plan 是否符合预期（B1），再看生成的源码（B2、B3）。

```bash
build/lab3/bench_fusion > results/lab3/fusion.csv
```

**通过标准**

- `test_a`、`test_b` 全部 PASS。
- softmax、layernorm、rmsnorm 各被融合成 1 个 kernel，8192×4096 时有效带宽 ≥ **180 GB/s**（参考实现 227–239 GB/s，unfused 为 36–63 GB/s）。
- matmul_bias_gelu 融合后比 unfused 快 ≥ **2×**（参考实现 2.3×）。

## Part C：mini-Mirage（第 2 周后半，约 5 小时）

读讲义 §5。三条改写规则，其中一条**故意写错**：

| 规则 | 改写 | 是否正确 |
|---|---|---|
| R1 | `MatMul(a * r, B) -> MatMul(a, B) * r`，对 `/ r` 同样适用；r 的形状为 [M, 1] | 正确：按行缩放与矩阵乘可交换 |
| R2 | 被多处使用的逐元素节点，给其中一个使用者一份独立的拷贝（反 CSE） | 正确，但单看只会增加计算 |
| R3 | `MatMul(a * s, B) -> MatMul(a, B) * s`，s 的形状为 [1, K] | **错误**；形状碰巧对得上（K == N）时才会被提出，C2 必须把它拒掉 |

| TODO | 内容 |
|---|---|
| C1 | `rewrite_once`：对每个节点尝试三条规则（`rebuild` 和 `clone_operand` 两个辅助函数已提供） |
| C2 | `probably_equivalent`：在随机输入上比较两个图（Schwartz-Zippel 引理） |
| C3a | `estimate`：每个 kernel 的耗时取 max(访存 / 带宽, 运算 / 算力)，再加启动开销 |
| C3b | `search`：广度优先搜索，按 canonical 形式去重，验证等价后保留，按代价排序 |

```bash
build/lab3/tc_dump softmax_matmul --search
build/lab3/bench_search > results/lab3/search.csv
```

**通过标准**

- `test_c` 全部 PASS。
- 对 `softmax_matmul`（形状与 attention 相同：8192 个 query、8192 个 key、d=64），搜索找到的最优方案**不物化任何 [M, K] 张量**，即 R2 + 融合，实测比 unfused 快 ≥ **1.5×**（参考实现 11.9 → 6.6 ms）。
- 报告中讨论 `bench_search` 的结果：参考实现中，代价模型排第一的候选，实测只排第二；实测最快的那个，模型排第四。为什么？代价模型漏掉了什么？提示：GEMM 模板中 A 的序言要对每个 N-tile 重算一遍；行内核对同一行要读多遍。

**可选挑战**

- **C4 Tensor Core 模板**：把 GEMM 模板换成 lab1 C2 的 `mma.sync` 版本（需要 fp16 输入）。
- **C5 改进代价模型**：把上面提示的两个因素加进 `estimate`，看排序是否与实测一致。
- **C6 真正的 FlashAttention**：让 Row kernel 和 MatMul kernel 融合成一个（需要 online softmax，也就是 lab2 B2 的做法）。思考：需要新增什么样的改写规则和 kernel 模板？这正是 Mirage 在 thread-block 层级上搜索出来的东西。

---

## 提交物与参考

报告写在 `reports/lab3.md`（[模板](../templates/report.md)），验收清单见 [grade.md](grade.md)。报告要包括：`tc_dump softmax` 生成的源码（附上你的注释）、融合前后的带宽表、`bench_search` 的完整输出与分析、卡住最久的地方。

- 讲义 [L3](../../mlsys-research-course/lectures/L3-compilers.md)
- [Mirage 论文](https://arxiv.org/abs/2405.05751) · [代码](https://github.com/mirage-project/mirage) · [MPK](https://arxiv.org/abs/2512.22219)
- [TorchInductor 设计](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)、[tinygrad](https://github.com/tinygrad/tinygrad)（运行时加 `DEBUG=4` 可以看到生成的代码）、[TVM](https://arxiv.org/abs/1802.04799)
- [NVRTC 文档](https://docs.nvidia.com/cuda/nvrtc/index.html)
