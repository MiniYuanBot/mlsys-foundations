# 代码格式化

以下路径与命令均以仓库根目录为基准。

所有 Lab 共用根目录配置：UTF-8、LF、末尾换行，目标行宽 100 列（Shell 不强制折行）。
C++/CUDA、Python、Shell 使用 4 空格；CMake、Markdown、JSON、YAML 使用 2 空格。
C++ 保留同行左花括号、变量侧的指针和引用（例如 `int *ptr`、`const Tensor &tensor`），include 在现有分组内排序；兼容 C++17。
Markdown 保留段落换行、行尾双空格和代码块内容。这里只配置格式化，不运行 lint、自动修复或 import 整理。

| 文件                        | Formatter                            | 配置                                    |
| --------------------------- | ------------------------------------ | --------------------------------------- |
| C++ / CUDA（含头文件）      | clang-format 20.1.8                  | `.clang-format`、`.clang-format-ignore` |
| `CMakeLists.txt`、`*.cmake` | gersemi 0.29.2                       | `.gersemirc`                            |
| Python                      | Ruff Formatter 0.17.0                | `pyproject.toml`                        |
| Shell（含 `*.sbatch`）      | shfmt 3.12.0                         | `.editorconfig`                         |
| Markdown / JSON / YAML      | Prettier 3.9.10                      | `.prettierrc.json`、`.prettierignore`   |
| Rust                        | rustfmt（Rust 1.99.0，edition 2021） | 默认规则：4 空格、100 列                |

## 安装

依赖 Python 3.10+、Node.js LTS（含 npm）。建议使用上表版本，让编辑器与命令行输出一致。
格式化工具可独立于实验环境安装；无需 CUDA。Rust 源码另使用 Rust 工具链自带的 rustfmt，已有 rustup 时执行 `rustup component add rustfmt` 即可。

WSL / Linux / macOS，在仓库根目录执行（Debian/Ubuntu 如缺少 venv，先安装 `python3-venv`）：

```bash
python3 -m venv .venv-format
source .venv-format/bin/activate
python -m pip install clang-format==20.1.8 gersemi==0.29.2 ruff==0.17.0
npm install --global prettier@3.9.10
```

Windows PowerShell：

```powershell
py -3 -m venv .venv-format
.\.venv-format\Scripts\Activate.ps1
python -m pip install clang-format==20.1.8 gersemi==0.29.2 ruff==0.17.0
npm.cmd install --global prettier@3.9.10
```

shfmt：macOS 可用 `brew install shfmt`；Windows、WSL、Linux 可从
[shfmt v3.12.0 Releases](https://github.com/mvdan/sh/releases/tag/v3.12.0)
下载对应系统和 CPU 架构的可执行文件，重命名为 `shfmt`（Windows 为 `shfmt.exe`）后加入 PATH；
WSL/Linux/macOS 还需执行 `chmod +x /path/to/shfmt`。已有 shfmt 须至少为 3.8，以支持忽略规则。
安装后的对应命令应都能在当前环境中运行；WSL / SSH 远程开发时，工具和扩展均安装在远端。

## VS Code

打开仓库根目录或现有 `mlsys-research.code-workspace`，在扩展面板安装本仓库推荐的扩展。
先激活格式化工具环境，再从同一终端执行 `code .` 或 `code mlsys-research.code-workspace`；
已运行的 VS Code 若找不到命令，完全退出后重开，使扩展宿主继承正确的 PATH。
Prettier 扩展会查找全局安装的版本，无需添加 npm 项目或 Prettier 插件。

保存支持的语言文件会调用对应 Formatter，也可执行 **Format Document**。
Ruff 的 lint、修复和 import 整理均已关闭；多根工作区文件也包含 Ruff 的窗口级开关。
共享设置不指定 `cmake.sourceDirectory`、构建目录、生成器或 Kit，可继续在个人/工作区设置中选择 Lab，
例如 `cmake.sourceDirectory = "${workspaceFolder:mlsys-research-labs}/lab0"`。
除 `settings.json`、`extensions.json` 外的 `.vscode/` 文件仍保持本地忽略。

## 手动格式化

在仓库根目录、已激活工具环境的终端中，按需执行相应命令，仅传入本次编辑的文件：

```text
clang-format -i lab0/src/tensor_student.cpp lab1/src/basics_student.cu
gersemi --in-place lab0/CMakeLists.txt
ruff format lab2/attn/softmax_student.py
rustfmt --edition 2021 lab4/src/kv_student.rs
shfmt --apply-ignore -w tools/check_env.sh lab6/hpc/nccl_allreduce.sbatch
prettier --write README.md lab7/examples/bucket_overlap.prereg.json
```

同样可给 gersemi 传入手写的 `*.cmake`，或给 Prettier 传入 `*.yaml` / `*.yml`。
Windows 若阻止 npm 的 PowerShell 启动脚本，使用 `prettier.cmd`；
如不能激活 venv，可直接调用 `.venv-format\Scripts\` 下的工具。

构建目录（`build*`、`cmake-build-*`、`target`、`out` 等）、生成文件、实验产物、
第三方依赖和虚拟环境已加入忽略配置。gersemi 的目录扫描遵循 `.gitignore`；
显式文件路径或编辑器缓冲区可能绕过部分工具的忽略规则，因此不要对生成文件或第三方代码主动执行格式化。
仓库现有源码已统一格式化；后续日常修改建议只格式化本次编辑的文件，减少无关 diff。
`.gitattributes` 约定文本文件检出使用 LF，无需执行全仓库换行归一化。
