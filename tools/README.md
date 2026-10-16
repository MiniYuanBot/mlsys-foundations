# 命令工具

从仓库根目录运行，参数用 `python -m tools.<名字> --help` 查看。

| 工具                                   | 用途                                                                                                                                |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `doctor.py`                            | 检查解释器与实验依赖，导出环境信息                                                                                                  |
| `capture.py`                           | 运行外部命令，并保存配置、环境与日志                                                                                                |
| `setup_env.sh` / `setup_env.ps1`       | 按 lab 编号分阶段安装课程环境：`.sh` 用于 WSL/Linux（全部 lab）和 macOS（lab0/6/7），`.ps1` 用于 Windows 原生（lab0/7）；可重复运行 |
| `check_env.sh` / `check_env.ps1`       | 只检查该 lab 需要的工具，必需项都 OK 时退出码为 0，失败项附修复命令                                                                 |
| `env-versions.env`                     | 课程环境的锁定版本（CUDA、PyTorch、Python 包），只在这里改                                                                          |
| `plot.py`                              | 从 CSV 画散点图（拒绝把模拟数据和实测数据、或不同设备画在同一张图里）                                                               |
| `setup-format.sh` / `setup-format.ps1` | 按 `formatter-versions.env` 安装格式化工具（WSL/Linux/macOS 与 Windows），`--check` 只检查                                          |
| `format_check.py`                      | 只读检查所有 Git 跟踪文件是否符合格式化工具版本，全部通过时退出码为 0                                                               |

`setup_env.*` 和 `setup-format.*` 会安装依赖，其余工具都不会。实验环境的版本只在 `env-versions.env` 中修改，格式化工具的版本只在 `formatter-versions.env` 中修改。
