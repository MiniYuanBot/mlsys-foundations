# 框架自检

这里检查公共工具、评分器、目录结构和文档链接。

- `test_infrastructure.py`：数据约束、命令生成与证据校验。
- `test_grading.py`：评分状态及层次隔离。
- `test_structure.py`：任务文件、评分配置、语法与链接。
- `grader_fixtures.py`：评分器测试样例。

修改公共工具或调整文件后，从仓库根目录运行 `python -m unittest discover -s tests -v`。这些检查通过不代表完成了 lab；学生测试在各实验目录。
