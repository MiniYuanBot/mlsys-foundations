# 参考答案

这里存放每个 lab 中 `*_student.*` 文件的完整参考实现，目录结构与对应的 lab 相同。例如，`solutions/lab4/src/kv_student.rs` 对应 `labs/lab4/src/kv_student.rs`。[脚手架验证范围](../labs/docs/validation-scope.md) 记录了参考实现的测试结果、性能数据和未验证的条件。

建议先自己做完，或者至少在一个 TODO 上卡住 30 分钟以上，再来看答案。

## 怎么用

以下命令都在仓库根目录执行。

**对照自己的实现**：

```bash
git diff --no-index labs/lab2/attn/flash_student.py solutions/lab2/attn/flash_student.py
```

**在独立副本中运行参考实现**：

```bash
git worktree add --detach ../mlsys-foundations-ref HEAD
cp -r solutions/lab4/. ../mlsys-foundations-ref/labs/lab4/
cd ../mlsys-foundations-ref/labs/lab4
cargo test --release
```

lab5 依赖 lab4。创建上述副本后，运行参考版 lab5 前需要同时覆盖副本中的 lab4 和 lab5；从主仓库根目录执行：

```bash
cp -r solutions/lab4/. ../mlsys-foundations-ref/labs/lab4/
cp -r solutions/lab5/. ../mlsys-foundations-ref/labs/lab5/
cd ../mlsys-foundations-ref/labs/lab5
cargo test --release
```

其他 lab 也按同样方式把 `solutions/labN/` 的内容覆盖到副本中的 `labs/labN/`，再从副本的 `labs/` 目录按该 lab 的 README 运行。不要覆盖主工作区中自己的实现后提交。
