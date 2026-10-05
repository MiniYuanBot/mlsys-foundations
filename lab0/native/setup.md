# C++ 环境（自行配置）

需要C++17编译器与CMake≥3.18。推荐在WSL使用同一Linux工具链，Mac使用本机C++编译器也可；Windows可用CMake识别的MSVC/MinGW，三者的性能结果不混合。本项目无pip包/pybind11依赖。

```bash
c++ --version
cmake --version
cmake -S lab0/native -B build/lab0-native -DCMAKE_BUILD_TYPE=Release
cmake --build build/lab0-native --config Release
ctest --test-dir build/lab0-native -C Release --output-on-failure
```

多配置生成器可能把可执行文件放在Release子目录，按实际位置运行；评分包装器可通过环境变量MLSYS_NATIVE_EXE指定路径。Linux默认路径见 [任务说明](assignment.md)。初始学生test报TODO是正常结果；supplied_reference通过只验证提供的oracle，不给学生分。先建results/lab0再重定向CSV。保存完整构建命令/版本、CPU型号、线程和电源信息。

不需要CUDA Toolkit。硬件事件可按平台工具说明选用perf等，但无权限时用壁钟+资源账本并写限制，不把理论cache miss当实测。

