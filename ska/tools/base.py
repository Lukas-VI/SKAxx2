"""工具基类与注册装饰器。

设计：
    * Tool 是抽象基类，提供 run()（执行）与 schema()（生成 OpenAI 函数声明）；
    * @tool("名字") 把「类」注册进内核的 TOOL_REGISTRY（不是实例——
      实例化推迟到 main.py 装配时，好把配置/记忆/网关等依赖注入进去）；
    * build_tools() 按配置名单实例化，tool_schemas() 生成给 LLM 的 tools 参数。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..kernel.registry import TOOL_REGISTRY

# 解码候选顺序：先声明的（如网页 meta charset），再 UTF-8，最后 GB18030。
# 选 GB18030 而不是 GBK：它是 GBK/GB2312 的超集，能多认一批生僻字。
_DECODE_FALLBACKS = ("utf-8", "gb18030")


def decode_bytes(raw: bytes, *, preferred: str = "") -> str:
    """把「来源编码未知」的字节流解码成文本（Windows 中文乱码的统一出口）。

    背景：Windows 下子进程/中文站点大量使用 GBK(CP936)，而本项目其余链路全是
    UTF-8。若直接 `decode("utf-8")`，中文就会变成 `ϵͳ�Ҳ���...` 这类乱码；
    反过来直接按 GBK 解码又会把正常的 UTF-8 文本解坏。所以按「声明编码 →
    UTF-8 → GB18030」逐个严格尝试，全都失败才带替换符兜底（保证不抛异常）。
    """
    data = raw or b""
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(_DECODE_FALLBACKS)
    for encoding in candidates:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue                      # 该编码解不通 → 换下一个
    return data.decode("utf-8", errors="replace")


def decode_output(raw: bytes) -> str:
    """解码子进程输出：**逐行**尝试编码，兼容同一条输出里 UTF-8 与 GBK 混排。

    为什么不能整段解一次：Windows 上 cmd.exe 自身的提示是 GBK，而被执行的
    Python/其它程序（尤其我们注入了 PYTHONIOENCODING=utf-8 之后）输出 UTF-8。
    两种字节混在同一段里，无论整段按哪个编码都会把另一半解成乱码；
    逐行处理则每行各自挑能解通的编码，两种来源都能正确显示。
    安全性：UTF-8 的多字节序列里不会出现 0x0A，按行切分不会切断字符。
    """
    if not raw:
        return ""
    return "".join(decode_bytes(line) for line in raw.splitlines(keepends=True))


@dataclass
class ToolDeps:
    """注入给工具的共享依赖（按需取用，无依赖的工具忽略即可）。"""

    cfg: Any = None       # kernel.config.Config
    gateway: Any = None   # llm.gateway.LLMGateway
    store: Any = None     # memory.store.MemoryStore
    bus: Any = None       # kernel.bus.EventBus
    extra: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """所有工具的基类。"""

    name: str = ""                     # 由 @tool("名字") 填充
    description: str = ""              # 给 LLM 看的功能说明
    # JSON Schema 描述入参；默认无参
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, deps: ToolDeps | None = None) -> None:
        self.deps = deps or ToolDeps()

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """执行工具并返回文本结果（会回填进对话上下文，故应简洁）。"""
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI tools 参数中的单项声明。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(name: str):
    """类装饰器：把工具类注册进全局注册表，并把 name 写回类属性。"""

    def decorator(cls: type[Tool]) -> type[Tool]:
        cls.name = name
        TOOL_REGISTRY.register(name, cls)
        return cls

    return decorator


def build_tools(names: list[str], deps: ToolDeps) -> dict[str, Tool]:
    """按配置名单实例化工具。

    Raises:
        KeyError: 名单里出现未注册的工具名（配置写错时尽早暴露）。
    """
    built: dict[str, Tool] = {}
    for name in names:
        cls = TOOL_REGISTRY.require(name)
        built[name] = cls(deps)
    return built


def tool_schemas(tools: dict[str, Tool],
                 allow: set[str] | None = None) -> list[dict[str, Any]]:
    """生成传给 LLM 的 tools 参数。

    Args:
        allow: 允许的工具名集合；为 None 表示「不裁剪」（阶段一行为），
               非 None 时只声明集合内的工具（阶段二权限裁剪，见 Policy.allowed_tools）。
    """
    return [t.schema() for name, t in tools.items() if allow is None or name in allow]