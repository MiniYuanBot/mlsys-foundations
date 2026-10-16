# 代码格式化

以下命令默认在 `labs/` 目录执行。格式配置位于仓库根目录，对 `course/`、`labs/` 和 `solutions/` 生效；格式化工具安装在 `labs/.venv-format/`。

所有 Lab 共用仓库根目录的配置：UTF-8、LF、末尾换行，目标行宽 100 列（Shell 不强制折行）。
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

版本号只在 `tools/formatter-versions.env` 中修改，上表与该文件保持一致。

## 安装

依赖 Python 3.10+、Node.js LTS（含 npm），以及 rustup（仅 Rust 源码需要）。格式化工具无需 CUDA，可独立于实验环境安装。

WSL / Linux / macOS，在 `labs/` 目录执行：

```bash
bash tools/setup-format.sh
```

Windows PowerShell，在 `labs/` 目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup-format.ps1
```

脚本会在 `.venv-format` 中安装 clang-format、gersemi、ruff 和 shfmt，用 npm 全局安装 prettier，并通过 rustup 添加 rustfmt。
可以重复执行，版本已经正确的工具会跳过。只检查、不修改：`bash tools/setup-format.sh --check`（Windows 下加 `-Check`）。

- Debian/Ubuntu 若报 venv 相关错误，先执行 `sudo apt install python3-venv`。
- macOS 可执行 `brew install python@3.12` 获得 Python 3.10+。
- 脚本不会用 sudo，也不会修改实验环境。

安装后执行检查：

```bash
python3 tools/format_check.py
```

它检查整个仓库中 Git 跟踪的文件，包括课程、实验和参考答案，不修改任何内容；全部通过时退出码为 0，失败时列出未格式化的文件。
Windows 下把 `python3` 换成 `python`。

## VS Code

打开仓库根目录，在扩展面板安装推荐的扩展。新终端默认进入 `labs/`。
`.vscode/settings.json` 直接指向 `${workspaceFolder}/labs/.venv-format/bin/` 下的 clang-format 和 shfmt，
不需要激活 venv，也不依赖终端的 PATH，因此从图形界面启动 VS Code 同样可用。
原生 Windows 上这些文件位于 `labs/.venv-format/Scripts/`，需要在个人用户设置中覆盖 `C_Cpp.clang_format_path` 和 `shfmt.executablePath`；WSL 中无需处理。
Prettier 扩展会查找全局安装的版本，无需添加 npm 项目或 Prettier 插件。

保存支持的语言文件会调用对应 Formatter，也可执行 **Format Document**。
Ruff 的 lint、修复和 import 整理均已关闭。
共享设置不指定 `cmake.sourceDirectory`、构建目录、生成器或 Kit，可继续在个人/工作区设置中选择 Lab，
例如 `cmake.sourceDirectory = "${workspaceFolder}/labs/lab0"`。
除 `settings.json`、`extensions.json` 外的 `.vscode/` 文件仍保持本地忽略。

## 手动格式化

在 `labs/` 目录、已激活 `.venv-format` 的终端中，按需执行相应命令，仅传入本次编辑的文件。
未激活时，把命令换成 `.venv-format/bin/<工具>`（Windows 为 `.venv-format\Scripts\<工具>.exe`）：

```text
clang-format -i lab0/src/tensor_student.cpp lab1/src/basics_student.cu
gersemi --in-place lab0/CMakeLists.txt
ruff format lab2/attn/softmax_student.py
rustfmt --edition 2021 lab4/src/kv_student.rs
shfmt --apply-ignore -w tools/check_env.sh lab6/hpc/nccl_allreduce.sbatch
prettier --write README.md lab7/examples/bucket_overlap.prereg.json
```

同样可给 gersemi 传入手写的 `*.cmake`，或给 Prettier 传入 `*.yaml` / `*.yml`。
Windows 若阻止 npm 的 PowerShell 启动脚本，使用 `prettier.cmd`。

构建目录（`build*`、`cmake-build-*`、`target`、`out` 等）、生成文件、实验产物、
第三方依赖和虚拟环境已加入忽略配置。gersemi 的目录扫描遵循 `.gitignore`；
显式文件路径或编辑器缓冲区可能绕过部分工具的忽略规则，因此不要对生成文件或第三方代码主动执行格式化。
仓库现有源码已统一格式化；后续日常修改建议只格式化本次编辑的文件，减少无关 diff。
`.gitattributes` 约定文本文件检出使用 LF，无需执行全仓库换行归一化。
