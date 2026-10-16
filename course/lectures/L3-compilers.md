---
title: L3 深度学习编译器与超优化
lab: lab3
weeks: 7-8
tags: [lecture, compiler, fusion, codegen, superoptimization, mirage, megakernel]
---

# L3 深度学习编译器与超优化

[← 课程大纲](../syllabus.md) · 上一讲：[L2 Attention](L2-attention.md) · 配套实验：[lab3](../../mlsys-research-labs/lab3/README.md) · 下一讲：[L4 LLM 推理系统](L4-serving.md)

> [!abstract] 这一讲要解决的三个问题
> 1. **编译器怎么把一张计算图变成 GPU kernel？** 中间经过哪些步骤，每一步做了什么决定？
> 2. **融合的规则是什么，边界在哪里？** 为什么基于规则的融合永远写不出 FlashAttention？
> 3. **超优化器（TASO、PET、Mirage）比规则多做了什么？** 它们靠什么保证改写后的程序仍然正确？megakernel 又把“融合”推到了什么程度？

L0 Part C 用解释器执行融合后的计算图，L1、L2 是手写 kernel。本讲把两者连起来：**让程序替你写 kernel**。这也是苗旭鹏老师组 Mirage 和 MPK 的研究方向。

| 节 | 内容 | 对应 lab |
|---|---|---|
| §1 | 编译器流水线：IR、参考执行、代码生成 | Part A |
| §2 | 融合：三类 kernel 与吸收规则 | Part B1 |
| §3 | 行归约 kernel 的代码生成：重算还是物化 | Part B2 |
| §4 | GEMM 模板与序言/尾声融合 | Part B3 |
| §5 | 超优化：改写、验证、搜索 | Part C |
| §6 | 演进脉络：Halide → TVM → Triton → Inductor → Mirage → megakernel | 选题参考 |

---

## §1 编译器流水线

```
 user code           graph capture           IR (a DAG)               lowering
 y = softmax(x)  ->  trace / Dynamo  ->  x -> RowMax -> Sub -> Exp  -> fusion groups -> kernel source -> binary
                                                       \-> RowSum -/        (lab3 B1)      (A3, B2, B3)    (NVRTC)
```

lab3 实现的就是从 “IR” 到 “binary” 这一段：

- **IR**：一张 DAG。每个节点是一个算子，带有形状。lab 只用二维张量和二维广播，概念上与 TorchInductor 的 IR、XLA 的 HLO 相同。
- **参考执行器**（lab A2）：用 fp64 在 CPU 上逐节点直接计算。它很慢，但**显然正确**，后面所有的生成代码和改写都以它作为判断对错的标准。
- **代码生成**：为每个融合组生成一段 CUDA C++ 源码。lab 用的是 SSA 风格，每个中间值是一个 `const float vN = ...;`。
- **运行时编译**：NVRTC 把源码字符串编译成 GPU 二进制，driver API 再把它加载并启动。

> [!tip] 为什么要在运行时编译？
> 形状、融合方式都要等程序运行起来才知道。PyTorch 的 `torch.compile` 用 Triton 生成并编译 kernel，tinygrad 用 clang 或 NVRTC，lab3 用 NVRTC。代价是第一次运行很慢（编译需要几十到几百毫秒），所以要按源码缓存编译结果；lab 的 `Runtime` 就是以源码文本为键做缓存的。

---

## §2 融合：三类 kernel 与吸收规则

### 2.1 三类 kernel

| 类型 | 并行方式 | 代表运算 |
|---|---|---|
| Elementwise | 每个线程一个输出元素 | add、mul、exp、gelu |
| Row（归约） | 每个 block 一行，列方向用循环 | softmax、layernorm、rmsnorm |
| MatMul | 分块 GEMM 模板 | 线性层 + bias + 激活函数 |

### 2.2 吸收规则（lab B1）

按**逆拓扑序**（先访问消费者）遍历每个节点 n：如果 n 的所有消费者都在同一个 kernel K 中，并且 n 可以加入 K，就把 n 并进 K。这样 n 永远不会被写到显存里。否则 n 自成一个新 kernel，它的结果要**物化**到显存。

以 softmax 为例（x 为 [R, C]）：

```
reverse topo order        decision
 Div  (output)        ->  new kernel K0 (Elementwise)
 RowSum(e)            ->  consumers {Div} all in K0, reduction of width C -> joins, K0 becomes Row
 Exp  = e             ->  consumers {RowSum, Div} both in K0 -> joins
 Sub                  ->  joins
 RowMax(x)            ->  consumers {Sub} in K0, same width C -> joins
=> ONE kernel: read x, write y.  Unfused: 5 kernels, 4 temporaries of [R, C]
```

不能融合的典型情况：

1. **消费者分散在两个 kernel 中**：比如 `t = exp(x)` 同时被 matmul 的 A 操作数和 matmul 之后的加法使用（lab 测试 `B1_node_feeding_two_kernels_is_materialised`）。t 必须写进显存，供两个 kernel 分别读取。
2. **行宽不一致**：一个 16 列的循环里，没法同时归约一行 32 列的数据。
3. **matmul 的 B 操作数**：GEMM 模板对 B 的加载方式固定，lab 只允许把逐元素运算融合到 A 一侧。
4. **归约与 matmul**：一个 GEMM tile 只看到一行的一部分（BK 列），无法完成整行归约。

第 4 条正是 attention 的情况：softmax(x)·V 里，softmax 需要整行归约，而 matmul 只能逐块地看到这一行。lab 中规则融合的结果是 3 个 kernel，并且要把 [M, K] 的 exp 结果写入显存一次、再读两次。§5 会说明搜索是怎样绕开它的。

---

## §3 行归约 kernel：重算还是物化（lab B2）

Row kernel 中，每遇到一个归约就要扫一遍整行。softmax 需要三遍：求 max、求 sum、写输出。每一遍都会用到 `x - m`、`exp(...)` 这类“整行”中间值。它们放不进寄存器（每行有几千个元素），有两种处理办法：

| 做法 | 代价 |
|---|---|
| 写入 shared memory 或显存，下一遍再读 | 额外的访存；行太长时 shared memory 装不下 |
| **每一遍都重新计算** | 多做几次 exp 等运算 |

lab 选择重算：生成的代码里，`x - m` 在第二遍和第三遍各算一次。这在访存受限的场景中划算，背后是 L2 FlashAttention 的同一个道理：**在访存受限时，用计算换带宽**。lab 实测：融合后的 softmax 达到 236 GB/s，unfused 版本只有 55 GB/s。

lab 中 `tc_dump softmax` 生成的代码结构如下（已简化）：

```
tc_kernel(in0, out, R, C) {
  i = blockIdx.x
  acc2 = -inf
  for j in row:  acc2 = max(acc2, in0[i*C+j])                 // pass 1: RowMax
  v2 = block_max(acc2)
  acc5 = 0
  for j in row:  v3 = in0[i*C+j] - v2; v4 = expf(v3); acc5 += v4   // pass 2: RowSum, recompute
  v5 = block_sum(acc5)
  for j in row:  v3 = ...; v4 = expf(v3); out[i*C+j] = v4 / v5    // pass 3: output, recompute
}
```

---

## §4 GEMM 模板与序言/尾声融合（lab B3）

GEMM 不适合从零开始生成，它的性能来自 L1 中那些精细的分块。编译器的通常做法是：**保留一个手写的 GEMM 模板，只把三个“钩子”函数交给代码生成**：

```
template (provided):                       hooks (generated, lab B3):
  for k-tile:                                tc_load_a(i, j)   = prologue ops on A(i, j)
    As[..] = tc_load_a(i, j)       <------   tc_load_b(i, j)   = B(i, j)
    Bs[..] = tc_load_b(i, j)                 tc_epilogue(acc, i, j) = epilogue ops on acc
    acc += As * Bs
  out[i, j] = tc_epilogue(acc, i, j)
```

- **尾声**（epilogue）：matmul 之后的逐元素运算，比如 `+ bias`、`gelu`、`+ residual`，在写回之前就地完成。
- **序言**（prologue）：A 操作数上的逐元素运算，比如 `x * gamma`，在把 A 读进 shared memory 时完成。

CUTLASS 的 epilogue visitor、cuBLASLt 的 epilogue、TorchInductor 的 “template + epilogue fusion” 都是这种模式。lab 实测：matmul_bias_gelu 融合后，从 11 个 kernel、15.2 ms 降到 1 个 kernel、6.5 ms。

> [!warning] 序言的隐藏代价
> 尾声对每个输出元素只执行一次。序言则不同：A 的每个元素会被**每一个 N 方向的 tile** 各读一次，序言也就跟着被重复计算 N/BN 次。N 较大时，复杂的序言可能比单独物化 A 更慢。§5 的代价模型没有考虑这一点，这正是 lab C 中模型与实测不一致的原因之一。

---

## §5 超优化：改写、验证、搜索（lab C）

### 5.1 规则融合的局限

规则融合只能在**给定的图**上做决定。但很多更好的方案需要**先把图改一改**：

| 改写 | 效果 |
|---|---|
| `MatMul(a * r, B) = MatMul(a, B) * r`（r 是按行的缩放） | 把 rmsnorm 的缩放从 A 一侧（序言，每个 N-tile 都要重算）挪到输出一侧（尾声，只算一次） |
| 反 CSE：给被共享的节点每个使用者一份拷贝 | 结合融合，可以消除一个必须物化的大张量 |

softmax·V 的例子：e = exp(x − m) 被 RowSum 和 matmul 共用，规则融合只能把它物化。如果给 matmul 一份独立的 e 拷贝（反 CSE），那么 RowSum 那一份可以融合进 Row kernel，matmul 那一份可以融合进 GEMM 的序言，**[M, K] 的 e 就再也不需要写入显存了**。代价是 exp 要算两遍。这正是 FlashAttention 的核心思想。lab 的搜索会自己找到这个方案：实测从 11.9 ms 降到 6.6 ms。

### 5.2 怎么保证改写正确：验证

自动生成的改写可能是错的。lab 的 R3 就故意写错：`MatMul(a * s, B) = MatMul(a, B) * s`，其中 s 按列缩放。只有当 K == N 时，这个式子的形状才对得上，但它的值一般是错的。

验证方法有两类：

| 方法 | 代表系统 | 优缺点 |
|---|---|---|
| 形式化证明（SMT 求解器） | [TASO](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf)（SOSP'19） | 严格，但要为每个算子写规格说明，求解器可能很慢 |
| **随机测试** | Mirage、lab C2 | 简单，适用于任意算子；有极小的误判概率 |

随机测试为什么可信？如果两个函数都是 d 次多项式，并且不相等，那么在一个随机点上它们取值相同的概率不超过 d/|域的大小|（**Schwartz–Zippel 引理**）。Mirage 在有限域上做随机测试，没有舍入误差，测几次就能把误判概率压到可以忽略。lab 在 fp64 实数上测 3 次，容差为 1e-9。

[PET](https://www.usenix.org/conference/osdi21/presentation/wang)（OSDI'21）更进一步：允许“部分等价”的改写，也就是只在部分输出位置上相等，然后自动生成修正代码，弥补不相等的部分。

### 5.3 代价模型与搜索

候选图可能有几十到上百万个，不可能每一个都实际运行一遍。所以需要一个**代价模型**快速估算。lab 用的是一个 Roofline 风格的模型：

$$t_{\text{kernel}} = \max\left(\frac{\text{bytes}}{\text{bandwidth}}, \frac{\text{FLOPs}}{\text{FLOP rate}}\right) + t_{\text{launch}}$$

搜索采用广度优先，以 canonical 形式去重，只保留验证通过的候选。

lab 的 `bench_search` 对比了模型的排序与实测的排序。参考实现中，模型排第一的候选实测只排第二，模型排第四的反而实测最快。原因是模型漏掉了 §4 中提到的序言重复计算，也没有考虑 Row kernel 对同一行的多次读取。**代价模型的准确性，往往是超优化器的瓶颈**。真实系统要么用学习型模型（TVM 的 Ansor），要么对排名前几的候选实际测量（Mirage、Triton autotune）。

### 5.4 Mirage 多了什么

lab 的搜索只在**图**这一层改写。Mirage 的搜索空间同时跨越三个层次：

```
kernel graph        : which ops form one kernel                    (lab3 Part B)
thread-block graph  : how one kernel's work is split into tiles,
                      what goes through shared memory, loop order  (FlashAttention lives here)
thread graph        : what each thread computes in registers
```

它把这三层统称为 **μGraph**。Mirage 正是在 thread-block 层上搜索，才能自动发现 FlashAttention 那样的 kernel：把 softmax 的归约和 matmul 放进同一个 kernel，按 tile 处理，并在线重缩放。lab 可选挑战 C6 要求你思考：为了做到这一步，还需要什么样的改写规则和 kernel 模板？

---

## §6 演进脉络

| 年份 | 工作 | 解决了上一步的什么问题 |
|---|---|---|
| 2013 | [Halide](https://halide-lang.org/)（PLDI） | **算法与调度分离**：“算什么”与“怎么算”分开描述，后者可以单独优化 |
| 2017 | XLA（TensorFlow） | 整图编译，自动融合逐元素运算；HLO IR |
| 2018 | [TVM](https://arxiv.org/abs/1802.04799)（OSDI） | Halide 的思路用于深度学习；学习型代价模型 + 自动调优 |
| 2019 | [TASO](https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf)（SOSP） | **自动生成**图改写规则，并用 SMT 形式化验证 |
| 2019 | [Triton](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)（MAPL） | 以 tile 为编程单位，块内细节交给编译器 |
| 2021 | [PET](https://www.usenix.org/conference/osdi21/presentation/wang)（OSDI） | 部分等价改写 + 自动修正，扩大了搜索空间 |
| 2022 | [TensorIR](https://arxiv.org/abs/2207.04296)（ASPLOS'23） | 以 Tensor Core 指令为单位的调度抽象 |
| 2023 | [TorchInductor / PyTorch 2](https://pytorch.org/assets/pytorch2-2.pdf)（ASPLOS'24） | Dynamo 从 Python 捕获图，Inductor 生成 Triton 代码；GEMM 模板 + 尾声融合，**lab3 B 的工业版** |
| 2025 | [TileLang](https://arxiv.org/abs/2504.17577) | 以 tile 为单位的 DSL，在 TVM 之上实现 |
| 2025 | **[Mirage](https://arxiv.org/abs/2405.05751)**（OSDI） | 跨 kernel、线程块、线程三层的 μGraph 超优化；有限域随机验证；**lab3 C 的原型** |
| 2025 | [Hazy Research: No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) | 手写 megakernel：整个 Llama-1B 前向放进一个常驻 kernel，消除 kernel 之间的启动开销和“气泡” |
| 2025 | **[MPK](https://arxiv.org/abs/2512.22219)** | **自动**把模型编译成任务图形式的 megakernel；SM 之间用事件同步，而不再依赖 kernel 边界 |

### 研究问题种子

- **更准的代价模型**：lab 的模型与实测排序不一致。能否只加入少数几个因素（序言重复计算、行的多次读取、wave quantization），就让排名前三的候选稳定地对上？
- **重算与物化的最优选择**：给定一张 DAG 和硬件参数，哪些节点应该物化、哪些应该重算？这是一个 NP 难的图划分问题吗？有没有在实际模型上足够好的启发式算法？
- **megakernel 的调度**：把整个模型放进一个 kernel 之后，各 SM 上的任务怎样排、怎样同步，才能把“气泡”降到最少？MPK 用的是什么策略，还有什么改进空间？

## 可视化资料汇总

| 资料 | 形式 | 配合哪一节 |
|---|---|---|
| [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)（Horace He） | 博客，融合的直觉图示 | §2 |
| [TorchInductor 设计帖](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) | 设计文档，IR 与调度器 | §1、§2、§4 |
| [depyf](https://github.com/thuml/depyf) | 工具，查看 torch.compile 生成的代码 | §1 |
| [tinygrad](https://github.com/tinygrad/tinygrad)（`DEBUG=4`） | 打印调度结果和生成的 kernel | §1–§3 |
| [TASO 演讲 slides](https://www.sigops.org/s/conferences/sosp/2019/slides/jia.pdf) | 改写的自动生成与验证 | §5 |
| [Mirage 文档](https://mirage-project.readthedocs.io/) | μGraph 示例 | §5.4 |
| [No Bubbles 博客](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) | megakernel 的流水线时间线 | §6 |
| [MLC 课程](https://mlc.ai/)（陈天奇） | 机器学习编译的完整课程 | 延伸 |

## 自测清单

- [ ] 能画出从计算图到 GPU 二进制的流水线，并说明每一步的输入和输出
- [ ] 能按吸收规则手动划分 softmax、layernorm、matmul_bias_gelu 的 kernel
- [ ] 能说出四种不能融合的情况，并各举一例
- [ ] 能解释为什么 Row kernel 选择重算中间值，而不是物化它们
- [ ] 能说明序言融合与尾声融合的代价差异
- [ ] 能用 Schwartz–Zippel 引理解释随机测试为什么可信
- [ ] 能说出 lab 的代价模型漏掉了什么，以及真实系统怎样弥补
- [ ] 能用三句话讲清 Mirage 比 TASO 多做了什么，MPK 又比 Mirage 多做了什么
