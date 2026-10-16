# 仓库自检

检查各 lab 的目录结构、Python 语法和 Markdown 链接，并确认学生文件里没有残留参考答案标记。

```bash
python -m unittest discover -s tests -v
```

这些检查通过，不代表你完成了 lab。每个 lab 的测试在该 lab 目录内，用 CTest 或 cargo test 运行。
