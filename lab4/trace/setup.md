# S04环境

复用lab3/4的固定vLLM服务环境。给源码patch另建工作副本/独立环境并固定**与你的wheel版本对应**的commit，按上游安装文档选择editable/source build；不要用main源码解释不对应版本的wheel。上游要求CUDA编译时按环境指南安装兼容工具链，生成trace脚本本身只需标准库。

请求客户端使用上游已验证实现或你明确记录版本的HTTP库；先20请求smoke核对实际token/流式事件，再跑正式矩阵。内网localhost两个终端；不要让单线程同步请求替代开放到达过程。不能拿客户端chunk数当token数。

没有GPU可完成trace生成和审计测试，真实cache/scheduler证据保持未执行。工具不自动下载上游代码/模型，不启动服务。

