# 手动准备与构建

本项目无需 Python、PyTorch 或模型。**没有替你安装依赖；以下命令由你阅读后自行执行。当前源码未编译、未 GPU 实测。** 推荐在 WSL2 Linux shell 内构建；本文件中的命令不是 PowerShell 命令。

需要 NVIDIA Windows 驱动、WSL2、CUDA Toolkit 12.x 或兼容后续版本（包含 `nvcc` 和 cuBLAS 开发文件）、与所选 Toolkit 兼容的 GCC/G++、CMake ≥3.24，以及 Make 或 Ninja。不要只安装含 CUDA runtime 的 PyTorch wheel：它不等于 CUDA 编译工具链。版本组合以 [CUDA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) 的 host compiler 支持表为准。

按照 [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 在 Windows 侧安装驱动，在 WSL 中安装工具链。不要在 WSL 内另装 Linux 显卡驱动；若使用 apt，应选择所需 `cuda-toolkit-12-x` 而非携带驱动的 `cuda` / `cuda-12-x` 元包。Nsight 工具能否使用取决于 Windows、驱动版本和 profiling 权限，实际检查后记录。

```bash
# WSL 中只读检查
nvidia-smi
nvcc --version
cmake --version
g++ --version

# 进入源码目录；也可将 checkout 放进 WSL 自己的 Linux 文件系统
cd /mnt/d/__Coding__/github/mlsys-research-labs/lab1/cuda

# RTX 4060，构建 Release。Toolkit 未自动发现时显式提供 CUDACXX 或 CUDAToolkit_ROOT。
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 2

# 先确认提供的代码能运行，再进入学生任务
./build/gemm_lab --test --reference-only
./build/gemm_lab --test
# 第二条在原始 TODO 状态下必须失败，看到 NOT_IMPLEMENTED 属正常现象。
```

若你用其他 GPU，先核对 [NVIDIA compute capability](https://developer.nvidia.com/cuda/gpus)，再改 CMake 参数。不要在不同 GPU 上直接复用仅为 `sm_89` 编译的程序。CMake 配置失败时保留完整日志，核实 `nvcc` 路径、compiler 版本和 toolkit/driver 兼容性；无需尝试安装全部 AI 软件。

完成学生代码后执行并留证：

```bash
set -o pipefail
mkdir -p results
cmake --build build --parallel 2 2>&1 | tee results/build.log
ctest --test-dir build --output-on-failure 2>&1 | tee results/tests.log
compute-sanitizer --tool memcheck ./build/gemm_lab --test 2>&1 | tee results/memcheck.log
compute-sanitizer --tool racecheck ./build/gemm_lab --test 2>&1 | tee results/racecheck.log
./build/gemm_lab --benchmark --iterations 20 2>&1 | tee results/benchmark.csv
```

`pipefail` 让前面的失败不会被 `tee` 的成功掩盖。检查命令退出码，不能仅看日志有 PASS 子串。CSV 中 `#` 开头是元数据/正确性记录；分析时忽略注释行。`results/` 默认不入 Git，完成报告时另行归档，不要删除原始日志。

仓库系统层评分器的 Python 桥接入口为 `lab1.cuda.tests.NativeGPU`，默认寻找**仓库根目录** `build/lab1-cuda/gemm_lab`（Windows 为 `.exe`）。如果按上面的局部目录构建，先设置显式路径，再从仓库根运行；修改 `.cu` 后必须重新构建，评分器不替你编译。

```bash
# 仍从 lab1/cuda 目录开始
export MLSYS_CUDA_EXE="$PWD/build/gemm_lab"
cd ../..
python -m unittest lab1.cuda.tests.NativeGPU -v
# 也可从仓库根统一构建，免去该变量：
# cmake -S lab1/cuda -B build/lab1-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
# cmake --build build/lab1-cuda --parallel 2
```

执行后续本目录 profile 示例前，请重新 `cd lab1/cuda`。Python 桥接层遇到未构建程序会给 `ENV:` skip；评分器须将其视为环境阻塞。原生程序存在但测试非零退出会判失败；不能以 unittest 的 `OK (skipped=...)` 作为完成证据。

可选 profile（不计入稳定计时，避免 profiler 扰动）：

```bash
# 用 reference-only 预检 profiler 环境；实现后移除该参数采集 tiled_gemm
ncu --set basic --kernel-name regex:naive_gemm --launch-count 1 \
  -o results/naive ./build/gemm_lab --test --reference-only

# 资源报告：单独目录，输出寄存器/共享内存/栈使用
cmake -S . -B build-resources -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS="--ptxas-options=-v"
cmake --build build-resources --parallel 2 2>&1 | tee results/resources.log
```

注意上面的 profile 只采集第一个匹配 kernel，属于工具烟测。性能分析应选择具体开发 shape 并记录 selector/launch 编号；不能拿单元素 fixture 的计数器解释 1024³ GEMM。建议进阶时添加明确的 `--shape` 选择入口，但保持默认正确性矩阵不变。

故障记录至少包含 Windows/WSL 版本、GPU/显存、driver、nvcc、CMake、compiler、cuBLAS 和构建参数。程序会自动打印部分 GPU/runtime 版本；还应保存源码 diff 或 commit、供电模式和其他 GPU 负载。环境不可用时记录 `blocked`，不把缺环境算作实验通过。
