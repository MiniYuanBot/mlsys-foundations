# 命令工具

从仓库根目录运行，参数用 `python -m tools.<名字> --help` 查看。

| 工具                                   | 用途                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------ |
| `doctor.py`                            | 检查解释器与实验依赖，导出环境信息                                                         |
| `capture.py`                           | 运行外部命令，并保存配置、环境与日志                                                       |
| `setup_wsl.sh`                         | WSL 中一次性安装 lab2–lab7 的环境（Python 虚拟环境、PyTorch/Triton、Rust、模型）           |
| `check_env.sh`                         | 逐项检查全部工具链，必需项都 OK 时退出码为 0                                               |
| `plot.py`                              | 从 CSV 画散点图（拒绝把模拟数据和实测数据、或不同设备画在同一张图里）                      |
| `setup-format.sh` / `setup-format.ps1` | 按 `formatter-versions.env` 安装格式化工具（WSL/Linux/macOS 与 Windows），`--check` 只检查 |
| `format_check.py`                      | 只读检查所有 Git 跟踪文件是否符合格式化工具版本，全部通过时退出码为 0                      |

`setup_wsl.sh` 和 `setup-format.*` 会安装依赖，其余工具都不会。格式化工具的版本只在 `formatter-versions.env` 中修改。
