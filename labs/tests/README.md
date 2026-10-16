# 仓库自检

检查课程、实验和参考答案的目录结构、Python 语法与 Markdown 链接，确认参考答案与学生文件对应，且学生文件里没有残留参考答案标记。下面的命令在 `labs/` 目录执行：

```bash
python -m unittest discover -s tests -v
```

也可以在仓库根目录执行 `python -m unittest discover -s labs/tests -v`。

这些检查通过，不代表你完成了 lab。各 lab 的测试命令见自己的 README；C++ / CUDA 用 CTest，Rust 用 `cargo test`，Python 用 pytest 或 unittest。
