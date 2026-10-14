"""WebUI 管理后台（阶段二新增，见方案 §5.3）。

零依赖实现（决策 D1）：
    * HTTP 服务用标准库 http.server.ThreadingHTTPServer，跑在守护线程；
    * 业务逻辑经「线程 ↔ asyncio 桥」回到主事件循环执行（server.py）；
    * 前端是原生 HTML/CSS/JS 单页，无构建步骤。

本包依赖 ska.admin（服务/存储/策略），但不被内核依赖。
"""