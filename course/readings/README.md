# 阅读资料

| 文件                                       | 用途                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| [resources.md](resources.md)               | 按讲整理的论文脉络、可视化资料、代码与课程，以及苗旭鹏老师组的代表性工作 |
| [open-source-labs.md](open-source-labs.md) | 复用外部课程作业或开源项目时，核对任务范围、硬件与许可证                 |

每讲讲义末尾已经列出必读材料；这里用于查阅和选题。

## 检查外部链接

[audit_external_links.py](audit_external_links.py) 检查仓库 Markdown 文档中的 HTTP/HTTPS 链接，只需要 Python 标准库。从仓库根目录执行：

```bash
python3 course/readings/audit_external_links.py
```

结果保存在本目录的 `results/external-links.json`，该目录由 Git 忽略。每个链接记录来源文档、HTTP 状态及重定向后的地址；检查不会修改讲义和资料表。也可以指定输出路径：

```bash
python3 course/readings/audit_external_links.py --output /tmp/mlsys-external-links.json
```

在 `labs/` 目录中运行时，脚本路径写为 `../course/readings/audit_external_links.py`；默认输出位置不随当前目录改变。

HTTP 状态只反映检查时的可达性。403/429 可能来自访问限制或限流，需人工核对；页面可访问也不表示论文机制已复现。更新阅读资料时，另行核实论文版本、代码 commit 和实验条件。
