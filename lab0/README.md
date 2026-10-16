# Lab 0：minitensor —— 用 C++ 写一个张量运行时

> 配套讲义：[L0 张量、内存与 CPU 性能](../../mlsys-research-course/lectures/L0-tensor-runtime.md)（先读第 1–2 节再开始 Part A）
> 预计用时：2 周 × 约 7 小时实验　·　语言：C++17　·　硬件：任意 CPU（Windows / WSL / Mac 均可）

PyTorch 的 `Tensor` 背后其实只有一块平坦的内存，再加上 shape、stride 和 offset 三样元数据；`x.T`、`x[:, 1:3]`、广播都不复制数据，只是改了这几样元数据。`x @ y` 最终会调用一个 BLAS 库。这个库写得好坏，速度能差 100 倍以上。`torch.compile` 会把一串逐元素算子融合成一个 kernel，好处是省掉中间结果来回读写内存。

这个 lab 让你用 C++ 依次亲手实现这三件事：

| 部分                        | 你要写什么                                                                           | 难度梯度                       | 对应的真实系统                                                                  |
| --------------------------- | ------------------------------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------- |
| **A** 张量核心              | stride/offset 寻址、transpose/slice/broadcast 视图、contiguous、广播逐元素运算、归约 | ① 代码填空（每个函数 5–20 行） | `torch.Tensor`、NumPy `ndarray`、[CMU Needle NDArray](https://dlsyscourse.org/) |
| **B** GEMM 优化阶梯         | 从 1 GFLOP/s 的三重循环，一步步做到接近 BLAS                                         | ② 核心代码实现 + 性能目标      | OpenBLAS、BLIS、Intel MKL                                                       |
| **C** Lazy 计算图与算子融合 | 拓扑排序、融合调度、表达式编译、分块解释器                                           | ③ 论文/系统设计的精简复现      | [tinygrad](https://github.com/tinygrad/tinygrad)、TorchInductor、XLA            |

做完后你应当能回答：

- 为什么 `x.T.contiguous()` 会复制数据，而 `x.T` 不会？
- 为什么同样是 2n³ 次浮点运算，不同的循环顺序能差 30 倍？
- 算子融合到底省掉了什么？什么时候融合也救不了你？

---

## 0. 环境与构建（约 30 分钟）

需要 C++17 编译器、CMake ≥ 3.18，以及一个装了 numpy 和 matplotlib 的 Python 环境（只用来跑基线和画图）；不需要 GPU。WSL、Linux、macOS 和 Windows 原生都可以做。建议用 WSL：从 lab1 开始所有实验都在 WSL 里做，这样环境只用配一次。在仓库根目录下执行：

```bash
# WSL / Linux / macOS
bash tools/setup_env.sh --lab 0     # g++, cmake, direnv, ~/venvs/mlsys with numpy/matplotlib/pytest
# open a NEW terminal and cd back into the repo: direnv now activates the venv
bash tools/check_env.sh --lab 0     # every line must be OK
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 0
# open a NEW PowerShell window, cd back into the repo, then activate the venv:
mlsys
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 0
```

之后在仓库里直接用 `python`（或 `python3`），用的就是课程环境里的版本。如果看到 `No module named 'numpy'`，说明当前终端没有激活课程环境，详见 [setup.md](../setup.md#虚拟环境放在哪里怎么生效)。

构建和测试（Windows 原生要在第一行末尾加 `-G Ninja`）：

```bash
cmake -S lab0 -B build/lab0 -DCMAKE_BUILD_TYPE=Release
cmake --build build/lab0 -j
ctest --test-dir build/lab0 --output-on-failure                # all 3 fail at first: expected
```

常用命令：

```bash
build/lab0/test_a_tensor          # all Part A tests
build/lab0/test_a_tensor A5       # only tests whose name contains "A5"
ctest --test-dir build/lab0 -L partB --output-on-failure
```

每个测试会打印 `[ PASS ]` 或 `[ FAIL ]`，失败时还会给出文件名、行号和失败的表达式。还没实现的函数会抛出 `TODO Ax ... is not implemented yet`。

**调试内存错误（推荐在 WSL/Linux/Mac 上用）**：`cmake -S lab0 -B build/lab0-asan -DMT_SANITIZE=ON`，越界读写会被 AddressSanitizer 当场捕获，而不是悄悄算出错误结果。

## 1. 代码地图

```
lab0/
+-- include/mt/
|   +-- tensor.hpp        Tensor class + ops (read it first: comments are the spec)
|   +-- gemm.hpp          GEMM API, blocking knobs MR/NR/MC/KC/NC, packing layout
|   +-- lazy.hpp          lazy graph, Kernel, Program (postfix code)
+-- src/
|   +-- *_core.cpp        provided: read them, normally no changes needed
|   +-- *_student.cpp     YOUR code: every TODO(A*/B*/C*) lives in these 3 files
+-- tests/                tests (check.hpp is a 60-line test framework)
+-- bench/                bench_gemm, bench_fusion
+-- scripts/              blas_baseline.py (strong baseline), plot_gemm.py (plots)
```

规则：**只改 `src/*_student.cpp`**。`gemm.hpp` 中的分块常量可以调，调完必须重跑测试。其他文件不要改；如果你认为某个测试本身有错，写进报告里。

---

## Part A：张量核心（第 1 周前半，约 3 小时）

先读讲义 L0 §1（内存布局与 stride）和 §2（视图与广播），然后打开 `src/tensor_student.cpp`。每个 TODO 上方都有规格说明和手算例子。建议**按编号顺序**完成：后面的函数会用到前面的。

| TODO | 函数                                | 要点                                                             | 测试   |
| ---- | ----------------------------------- | ---------------------------------------------------------------- | ------ |
| A1   | `contiguous_strides`                | 最后一维变化最快                                                 | `A1_*` |
| A2   | `offset_of`                         | 先检查边界，再算 `offset + Σ idx·stride`                         | `A2_*` |
| A3   | `is_contiguous`                     | 大小为 1 的维度，其 stride 可以是任意值                          | `A3_*` |
| A4   | `permute` / `transpose`             | 只交换元数据                                                     | `A4_*` |
| A5   | `slice`                             | 新 offset = 旧 offset + start·stride；新 stride = 旧 stride·step | `A5_*` |
| A6   | `broadcast_shapes` / `broadcast_to` | 从右往左对齐；被拉伸的维度 stride 设为 0                         | `A6_*` |
| A7   | `contiguous`                        | 唯一真正复制数据的操作                                           | `A7_*` |
| A8   | `zip`                               | 先广播成视图，再逐元素计算                                       | `A8_*` |
| A9   | `sum`                               | 用 `keepdim` 的技巧可以少写很多代码                              | `A9_*` |

**动手画图（强烈推荐）**：每完成一个视图函数，就用 `describe()` 打印结果，并和 [Stride Visualizer](https://ezyang.github.io/stride-visualizer/index.html) 的结果对照。例如：

```cpp
auto t = mt::Tensor::arange(12).view({3, 4});
std::cout << t.slice(1, 1, 4, 2).describe();
// shape=[3, 2] strides=[4, 2] offset=1 contiguous=no
// logical -> storage index
//    1   3
//    5   7
//    9  11
// storage: [.0 #1 .2 #3 .4 #5 .6 #7 .8 #9 .10 #11]
```

（可以在 `tests/` 里新建一个 scratch.cpp 自己玩，也可以临时写进某个测试函数。）

**通过标准**：`test_a_tensor` 全部 PASS，包括最后的 `A_capstone_softmax_from_views`（它只用你写的算子实现了一个 softmax）。

---

## Part B：GEMM 优化阶梯（第 1 周后半 + 第 2 周前半，约 6 小时）

先读讲义 L0 §3（缓存层次、Roofline、Goto 算法），然后打开 `src/gemm_student.cpp`。

### B0 先测出你机器的上限（30 分钟）

1. 查 CPU 型号与主频，按讲义 §3.1 的公式算出**单核 FP32 峰值** GFLOP/s。例如 Alder Lake P 核：4.7 GHz × 2 个 FMA 单元 × 8 lanes × 2 FLOP ≈ 150 GFLOP/s。
2. 跑强基线（BLAS 的单线程和多线程版本）：
   ```bash
   mkdir -p results/lab0
   python lab0/scripts/blas_baseline.py --threads 1 > results/lab0/blas_1t.csv
   python lab0/scripts/blas_baseline.py --threads 8 > results/lab0/blas_8t.csv   # use your core count
   ```
3. 跑一次 `build/lab0/bench_gemm`，现在只会有 naive 一行数据。把它和 BLAS 比一比，差距通常有 30–100 倍。

### B1–B4 逐级优化

| 步骤 | TODO                             | 做什么                                                      | 参考实现在作者机器上的结果（i9 Alder Lake, GCC 13, n=1024） |
| ---- | -------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| B0   | 已提供                           | i-j-k 三重循环                                              | ~1 GFLOP/s（n=1024 时 B 的列访问次次 cache miss）           |
| B1   | `gemm_reordered`                 | 交换成 i-k-j 循环                                           | ~30                                                         |
| B2   | `gemm_blocked`                   | 在 B1 外面套三层分块循环                                    | ~47                                                         |
| B3   | `pack_A` `pack_B` `micro_kernel` | Goto 算法：打包 + 6×16 寄存器微内核（5 层循环的驱动已提供） | ~113（MKL 单线程 129）                                      |
| B4   | `gemm_parallel`                  | 按行切块，交给 `std::thread` 并行                           | ~605 / 24 线程（MKL 650）                                   |

做每一步时都按这个循环走：**写代码 → `ctest -L partB` → `bench_gemm` → 记录数字 → 解释原因**。B3 拆成三个可以单独测试的函数，`B3a/B3b/B3c` 三个测试分别验证打包布局和微内核，逐个调通后再跑 `B3_packed`。

> ⚠️ **向量化很脆弱**。用 GCC 13 实测：微内核里的累加器写成二维数组 `float acc[6][16]`，或者漏掉 `#pragma omp simd`，单核会从约 130 GFLOP/s 掉到约 3 GFLOP/s，因为累加器被溢出到了栈上。改完一定要跑 benchmark。想看汇编可以这样做：
>
> ```bash
> g++ -std=c++17 -O3 -march=native -fopenmp-simd -Ilab0/include -S lab0/src/gemm_student.cpp -o - | grep -c vfmadd
> ```
>
> 如果数到的 `vfmadd…ps %ymm` 指令很少，说明微内核没有被向量化。

### B5 测量与报告

```bash
build/lab0/bench_gemm --sizes 128,256,512,1024,2048 > results/lab0/gemm.csv
python lab0/scripts/plot_gemm.py results/lab0/gemm.csv results/lab0/blas_1t.csv results/lab0/blas_8t.csv \
    --peak <your-1-core-peak> --out results/lab0/gemm.png
```

**通过标准**

- `test_b_gemm` 全部 PASS。
- n=1024 时，B3 单线程 ≥ BLAS 单线程的 **50%**；B4 ≥ BLAS 同线程数的 **50%**。
- 挑战线：B3 ≥ 80%。

**可选挑战**

- **B3d**：用 AVX2 intrinsics（`_mm256_fmadd_ps`）或 NEON（`vfmaq_f32`）手写微内核，参考 [BLISlab](https://github.com/flame/blislab) 和 [Algorithmica: Matrix Multiplication](https://en.algorithmica.org/hpc/algorithms/matmul/)。
- **B3e**：扫描 MC/KC，画出 GFLOP/s 热力图，找出最优点，并用你 CPU 的 L1/L2 大小解释它。

---

## Part C：Lazy 计算图与算子融合（第 2 周后半，约 4 小时）

先读讲义 L0 §4（从 eager 到 graph：融合省掉了什么）。`lazy.hpp` 顶部的注释是这部分的总设计，务必读完。

`LazyTensor` 上的运算不会立即计算，只会构建一张 DAG。调用 `realize_*()` 后，才依次执行：拓扑排序 → 调度（决定哪些节点合并到同一个 kernel）→ 把每个 kernel 编译成后缀指令 → 解释执行。

| TODO | 函数             | 要点                                                       | 测试   |
| ---- | ---------------- | ---------------------------------------------------------- | ------ |
| C1   | `toposort`       | 深度优先后序遍历；每个节点只出现一次                       | `C1_*` |
| C2   | `schedule_fused` | 只有 3 类节点需要物化：根节点、SumLast、被使用 ≥2 次的节点 | `C2_*` |
| C3   | `compile`        | 生成后缀（栈机）指令；同一个 buffer 只分配一个编号         | `C3_*` |
| C4   | `eval_tile`      | 每条指令处理一整个 tile（512 个元素），而不是一个元素      | `C4_*` |

**手算练习（写进报告）**：对 `y = relu(x * w + b)`，`x`、`w`、`b` 各有 N 个 float，分别算出不融合和融合时的 kernel 数、读字节数、写字节数和中间 buffer 字节数。答案就在 `C1_unfused_schedule_counts` 和 `C2_fused_chain_is_one_kernel` 两个测试里，先自己算再去看。

**测量**：

```bash
build/lab0/bench_fusion > results/lab0/fusion.csv
```

输出里有两个负载：`affine` 是访存密集型（几乎不做计算），`gelu` 里有 tanh，是计算密集型。每个负载各有五个变体：unfused、fused、tiled、手写 C++ 循环、memcpy 带宽基线。

**通过标准**

- `test_c_lazy` 全部 PASS。
- `bench_fusion` 中，两个负载的 `tiled` 都比 `unfused` 快 **≥ 3×**。
- 报告里回答下面 3 个问题：
  1. 对 GELU 而言，融合把内存流量减少到了原来的几分之一？实际加速比是多少？为什么二者对不上？（提示：比较 `effective_GBps` 和 `copy` 那一行。）
  2. `fused` 和 `unfused` 走的都是逐元素的标量解释器，`tiled` 为什么能快好几倍？这和 GPU 上“每个线程处理一个元素，但一个 warp 执行同一条指令”有什么相似之处？
  3. 在作者的机器上，affine 负载的 `tiled` 解释器居然比手写循环还快（37 ms 对 80 ms）。在你的机器上是否也这样？提出至少一个可检验的假设。（这是一道开放题，作者还没有确定原因；一个思路是看手写循环是否同时打开了太多内存流。）

**可选挑战（向前沿靠近）**

- **C5 代码生成**：把 `Program` 翻译成一段 C++ 源码字符串，写入文件，调用编译器编译成动态库，再用 `dlopen` 或 `LoadLibrary` 加载执行。这就是 tinygrad、TorchInductor 的 C++ 后端在做的事情；到 L3 你会把它换成生成 CUDA 代码、用 NVRTC 编译。
- **C6 跨归约融合**：让 `exp(x).sum_last()` 后面接的逐元素运算也能并入同一个 kernel（类似 softmax 中的 “reduce → broadcast → elementwise”）。思考：为什么这一步比融合纯逐元素链难得多？这正是 L2 FlashAttention 要解决的问题。

---

## 提交物

在 `results/lab0/`（git 不跟踪）下保存原始 CSV 和图，在 `reports/lab0.md` 中写报告（模板见 [templates/report.md](../templates/report.md)）。报告应包含：

1. 环境：CPU 型号、核数、L1/L2/L3 大小、编译器版本与编译参数。
2. Part B：GFLOP/s 曲线图；每一步优化的加速比，并用一句话解释原因；单核峰值的计算过程；与 BLAS 的比例。
3. Part C：手算的流量表；bench 结果表；上面 3 个问题的回答。
4. 一段“我卡住最久的地方”。这段对你自己最有用。

验收清单见 [grade.md](grade.md)。

## 参考资料

- 讲义：[L0 张量、内存与 CPU 性能](../../mlsys-research-course/lectures/L0-tensor-runtime.md)
- Edward Z. Yang, [PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/)：stride、storage、view 的图解
- [CMU 10-414 Deep Learning Systems](https://dlsyscourse.org/)：HW3 NDArray 的 C++/CUDA 后端，本 lab 的 Part A 借鉴了它的设计
- Goto & van de Geijn, [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/users/flame/pubs/GotoTOMS_final.pdf) (TOMS 2008)：Part B 的原始论文
- [BLISlab](https://github.com/flame/blislab)、[Algorithmica: Matrix Multiplication](https://en.algorithmica.org/hpc/algorithms/matmul/)、[Stanford CS149 asst1](https://github.com/stanford-cs149/asst1)
- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)：Part C 的直觉来源
- [tinygrad](https://github.com/tinygrad/tinygrad)：运行时加 `DEBUG=4` 可以看到它的调度与生成的代码
