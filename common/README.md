# 公共代码

各 lab 共用的数据记录、测量和测试辅助函数。

- `benchmark.py`：计时与显存测量。
- `records.py`、`io.py`：运行目录、元信息与文件读写。
- `service.py`：请求级数据的校验函数，目前各 lab 都没有用到。
- `testing.py`、`temporary.py`：环境检查与测试临时目录。

先按 lab 的入口使用这些函数，遇到计时或数据问题再查实现。修改公共代码后运行 [框架自检](../tests/README.md)，并检查受影响的实验。
