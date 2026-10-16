# Lab 7 验收清单

## Part A：统计工具

- [ ] `python -m unittest lab7.tests.test_stats -v`：13 项全部 PASS

## Part B：示例研究

- [ ] dev、lock、held-out、report 全部跑通；确认在 lock 之前 held-out 会被拒绝
- [ ] 报告中贴出示例的结果表，并用 3 句话解释：为什么 dev 是 inconclusive？为什么 P=2 没有效果？消融说明了什么？

## Part C：研究项目

- [ ] 设计评审（`design-review.md` 的格式），包括 ≥ 3 项最接近的工作和检索记录
- [ ] 预注册文件在第一次 dev 运行**之前**提交到 git（以 git log 的时间为证）
- [ ] 预注册包含：强基线、处理、≥ 1 个消融、≥ 1 个负对照、≥ 2 个 held-out 条件、每组 ≥ 5 次重复
- [ ] `.lock` 文件已提交；held-out 只运行了一次（`results/lab7/<name>/` 下只有一个 heldout 目录，或者附有说明）
- [ ] `report --markdown` 的完整输出，包括所有失败的运行
- [ ] 实现代码有测试（正确性），并说明额外开销
- [ ] 两页提案（[课程模板](../../course/templates/research-proposal.md)）
- [ ] 10 分钟汇报提纲（6 页以内）
- [ ] “局限”一节至少写出 3 条，说明你的结论在哪些条件下可能不成立

不要求结论是 supported。一个经过严格检验的 refuted 或 inconclusive，只要你能解释原因、给出下一步，同样算完成。
