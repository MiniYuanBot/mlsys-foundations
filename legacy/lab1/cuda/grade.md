# 自评：100 分，基础 → 实现 → 研究证据

这是 `lab1/cuda` 独立的 100 分评价表；主 lab 的整合方式见上级课程说明。**没有固定 TFLOP/s 或“超过 cuBLAS”要求。** 未实现默认非零退出，无 CUDA 是环境阻塞；只运行 reference-only 不给学生实现分。尚未对本脚手架进行实际编译/GPU 测试。

| 分项 | 分数 | 通过证据 |
|---|---:|---|
| 理解数据布局与基线 | 10 | 手算 fixture、解释 cuBLAS 的 N/M/K 和 N/K/N leading dimensions；保留基线测试日志 |
| shared-memory 实现 | 25 | 能解释协作 load、零填充、两次 barrier、tail store；只用自己的 kernel 完成输出，不调用参考答案 |
| 正确性与内存安全 | 20 | 默认 `--test` 与 CTest 全过 10 分；memcheck/racecheck 无学生 kernel 错误 5 分；新增至少两种反例/负测试并解释 5 分 |
| 测量与强基线 | 20 | 全部五个 shape × 三轮 × 三实现原始 CSV 8 分；同精度/同输入/事件计时与统计定义 6 分；naive/学生/cuBLAS 各自优势与失败形状 6 分 |
| 优化消融与硬件解释 | 15 | 至少三个配置，锁定每次变因；至少一份有效 profiler 或 ptxas 资源证据；将观测与访存/占用/边界模型连接 |
| 论文机制与留出验证 | 10 | 一张机制卡 3 分；先锁定假设/代码/留出矩阵 3 分；给出留出结果、反例和结论边界 4 分 |

80 分以上且 **默认正确性通过、学生 kernel 真实实现、计时不包含参考计算** 才算完成。性能分评价实验质量：正确而没有加速可以拿高分；失败 shape、负结果和 profiler 权限限制必须保留。缺 sanitizer/profiler 时相关未验证项暂不计分，不能用“未发现问题”代替工具结果。

## 必须保留的证据包

建议在 `results/<日期-实验名>/` 归档以下内容，不提交大型 profiler 文件到 Git：

- `environment.txt`、构建命令和版本、源码 commit/diff；标注 WSL、sm_89、供电/后台负载。
- `tests.log`、`memcheck.log`、`racecheck.log`，包含最终退出状态。修改测试只能新增覆盖，不能删边界/放宽误差来通过。
- `benchmark.csv`：全部三轮，不挑最好的单次运行；报告各 shape median/min/max 和 cuBLAS 效率。
- `ablation.csv`：TILE 8/16/32 或至少三种明确定义的变体；同 precision、数据、shape、计时条件。
- `report.md`：预注册假设、至少一个反例、解释图与论文机制卡；有模型估计但没有实际计数器时明确写“估计”。

## 留出任务

开发完成后冻结实现与调参结果，再自选 **3 个不同于默认矩阵** 的 shape，其中一个 M/N 很小、一个 K 很小、一个 M/N/K 全不整齐。由固定 seed 生成输入，先验证数值，再测同样的三轮；报告“此前最优配置是否仍最优”。留出 shape 不作为新的调参输入；若你据此再修改，原留出集转为开发集，另建一个新留出集。

额外提问：cuBLAS 为什么在某些小 shape 没有优势？扩大 tile 为什么可能退化？global bytes 的理论减少是否等价于 DRAM bytes 的减少？FP16 Tensor Cores 是否改变了比较问题？论文在 H100 上的异步流水线能否迁移到 Ada，缺的硬件原语是什么？回答需引用代码、日志或来源。

## Tensor Core 选修的独立评价

不并入以上满分门槛；建议单独记录 completed/partial/blocked。要求 FP16/BF16 输入 + FP32 accumulation、同精度 cuBLAS、误差分布、转换时间/端到端范围说明、一个指令级证据及至少一个退化形状。H100/Blackwell 特有路径必须标明真实设备；4060 上阅读源码或分析机制不能称为原论文硬件复现。
