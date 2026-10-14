"""工具包：AgentLoop 可调用的能力集合。

新增工具零改内核：
    1. 在本目录新建 xxx.py；
    2. 写 `@tool("名字") class XxxTool(Tool): ...`；
    3. 在 config.yaml 的 tools 列表里加上这个名字。
"""