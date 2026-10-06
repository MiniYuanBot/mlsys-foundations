# 参考答案

每个 lab 中 `*_student.*` 文件的完整参考实现，目录结构与对应的 lab 相同，例如 `solutions/lab4/src/kv_student.rs` 对应 `lab4/src/kv_student.rs`。这些就是 [docs/validation-scope.md](../docs/validation-scope.md) 里所有数字背后的实现：用它们替换学生文件后，各 lab 的测试会全部通过，性能达到 README 中列出的参考值。

> **建议**：先自己做完，或者至少在一个 TODO 上卡住 30 分钟以上，再来看。直接看答案省下的时间，往往会在 lab7 做自己的研究时加倍还回去。

## 怎么用

**对照自己的实现**（不改动你的代码）：

```bash
git diff --no-index lab2/attn/flash_student.py solutions/lab2/attn/flash_student.py
```

**在一份独立的副本里跑参考实现**（例如想看参考性能，又不想覆盖自己的代码）：

```bash
git worktree add ../labs-ref HEAD           # a second checkout next to this one
cp -r solutions/lab4/* ../labs-ref/lab4/    # overwrite the student files there only
cd ../labs-ref/lab4 && cargo test --release
```

lab5 依赖 lab4：要在副本中运行参考版 lab5，需要同时复制 `solutions/lab4` 和 `solutions/lab5`。

不要把 `solutions/` 中的文件直接复制进 `labN/` 然后提交，否则你自己的实现就丢了。
