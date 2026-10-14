"""IO 包：消息进出口（渠道）。

新增渠道零改内核：
    1. 本目录新建 xxx.py，写 `@channel("名字") class XxxChannel(Channel)`；
    2. config.yaml 的 channels 列表加名字，并按需加同名配置小节。
"""