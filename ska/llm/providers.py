"""多 LLM 提供商管理（阶段二新增，见方案 §5.2）。

阶段一的做法是「一个 llm 小节写死一个后端」；本模块把它升级为：
    config/providers.yaml 里可以有任意多个提供商实例，运行时用 active 指定生效的那个。

设计要点：
    * 全项目仍然只有 **一个** LLMGateway 实例 —— ProviderManager 持有它，
      切换提供商时调用 gateway.reconfigure(...)，避免各处引用失效；
    * YAML 是唯一持久化介质（D4：本地明文、透明可手改）；
    * presets 只是「快速添加」的模板，不参与实际调用；
    * 首次启动若 providers.yaml 不存在，用 config.yaml 的 llm 小节迁移出首个实例，
      保证阶段一配置继续可用（向后兼容）。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from ..kernel.config import package_root
from .gateway import LLMError, LLMGateway

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 内置预设模板：WebUI「一键添加」时预填 base_url / model 等字段
# 注意 api_key 一律留空（需用户自己填），避免把占位串当真实密钥发出。
# ---------------------------------------------------------------------------
PRESETS: list[dict[str, Any]] = [
    {"id": "ollama", "label": "Ollama 本地", "kind": "openai_compatible",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "qwen2.5:7b"},
    {"id": "deepseek", "label": "DeepSeek", "kind": "openai_compatible",
     "base_url": "https://api.deepseek.com/v1", "api_key": "", "model": "deepseek-chat"},
    {"id": "openai", "label": "OpenAI", "kind": "openai_compatible",
     "base_url": "https://api.openai.com/v1", "api_key": "", "model": "gpt-4o-mini"},
    {"id": "dashscope", "label": "通义千问（DashScope）", "kind": "openai_compatible",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "",
     "model": "qwen-plus"},
    {"id": "zhipu", "label": "智谱 GLM", "kind": "openai_compatible",
     "base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": "", "model": "glm-4-flash"},
    {"id": "moonshot", "label": "Moonshot（Kimi）", "kind": "openai_compatible",
     "base_url": "https://api.moonshot.cn/v1", "api_key": "", "model": "moonshot-v1-8k"},
    {"id": "custom", "label": "自定义（OpenAI 兼容）", "kind": "openai_compatible",
     "base_url": "", "api_key": "", "model": ""},
]

# 每个提供商实例的默认字段（缺失时补齐，避免下游到处写 .get 兜底）
_DEFAULTS: dict[str, Any] = {
    "kind": "openai_compatible",
    "api_key": "",
    "temperature": 0.8,
    "max_tokens": 1024,
    "timeout": 60,
    "enabled": True,
}

# id 合法字符：字母数字、下划线、连字符、点（用于自动生成与校验）
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class ProviderError(RuntimeError):
    """提供商配置非法（缺字段 / id 冲突 / 不存在）。"""


class ProviderManager:
    """提供商实例的唯一管理者，同时持有全项目唯一的 LLMGateway。"""

    def __init__(self, path: str | Path = "config/providers.yaml",
                 *, fallback: dict[str, Any] | None = None) -> None:
        # 相对路径统一挂到项目根下（与 Config.load_yaml 的约定一致）
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = package_root() / self.path

        self._data = self._load(fallback or {})          # 内存中的配置（dict）
        # 首次运行（文件不存在）时把迁移结果立刻落盘，方便用户直接查看/手改 YAML
        if not self.path.exists():
            self.save()
        # 用当前生效的提供商配置构造唯一网关；无配置时退回 fallback
        active = self.active_cfg() or dict(fallback or {})
        self.gateway = LLMGateway(active)
        self.gateway.provider_id = self.active_id()
        logger.info("LLM 提供商就绪：active=%s（共 %d 个）",
                    self.active_id() or "(未设置)", len(self._data["providers"]))

    # ------------------------------------------------------------------ 载入 / 保存
    def _load(self, fallback: dict[str, Any]) -> dict[str, Any]:
        """读取 providers.yaml；不存在或结构不合规时生成/迁移出一份。"""
        if self.path.exists():
            try:
                raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError) as e:
                logger.warning("读取 %s 失败，将重建：%s", self.path, e)
                raw = {}
            if isinstance(raw, dict) and raw.get("providers"):
                # 结构合规：补齐 active / presets 等键，逐个规范化实例
                providers = [self._normalize(p) for p in raw["providers"]
                             if isinstance(p, dict) and p.get("id")]
                data = {
                    "active": str(raw.get("active") or (providers[0]["id"] if providers else "")),
                    "providers": providers,
                    "presets": raw.get("presets") or PRESETS,
                }
                return data

        # 迁移路径：用 config.yaml 的 llm 小节造出第一个提供商实例
        data: dict[str, Any] = {"active": "", "providers": [], "presets": PRESETS}
        if fallback:
            migrated = self._normalize({"id": "default", "label": "默认（迁移自 config.yaml）",
                                        **fallback})
            data["providers"].append(migrated)
            data["active"] = migrated["id"]
            logger.info("未找到 providers.yaml，已用 config.yaml 的 llm 小节生成默认提供商")
        return data

    def save(self) -> None:
        """把内存配置写回 YAML（保持中文可读，不做 ASCII 转义）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active": self._data.get("active", ""),
            "providers": self._data["providers"],
            "presets": self._data.get("presets") or PRESETS,
        }
        self.path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=200),
            encoding="utf-8")

    # ------------------------------------------------------------------ 规范化 / 校验
    @staticmethod
    def _normalize(entry: dict[str, Any]) -> dict[str, Any]:
        """补齐默认字段并做类型归一（base_url 去尾斜杠、数值转 float/int）。"""
        out = dict(_DEFAULTS)
        out.update({k: v for k, v in entry.items() if v is not None})
        out["id"] = str(out.get("id", "")).strip()
        out["label"] = str(out.get("label") or out["id"])
        out["base_url"] = str(out.get("base_url", "")).rstrip("/")
        out["api_key"] = str(out.get("api_key", ""))
        out["model"] = str(out.get("model", ""))
        out["kind"] = str(out.get("kind") or "openai_compatible")
        out["temperature"] = float(out.get("temperature", 0.8))
        out["max_tokens"] = int(out.get("max_tokens", 1024))
        out["timeout"] = int(out.get("timeout", 60))
        out["enabled"] = bool(out.get("enabled", True))
        return out

    @staticmethod
    def _validate(entry: dict[str, Any]) -> None:
        """校验必填项（WebUI 传来的数据需尽早暴露错误）。"""
        if not _ID_RE.match(str(entry.get("id", ""))):
            raise ProviderError("提供商 id 只能包含字母/数字/下划线/连字符/点，且不能为空")
        if not entry.get("base_url"):
            raise ProviderError("base_url 不能为空")
        if not entry.get("model"):
            raise ProviderError("model 不能为空")

    def _gen_id(self, entry: dict[str, Any]) -> str:
        """根据预设 id 或 label 生成一个不冲突的 id。"""
        base = str(entry.get("preset") or entry.get("id") or "provider").strip() or "provider"
        base = re.sub(r"[^A-Za-z0-9_.\-]", "-", base)
        if base not in {p["id"] for p in self._data["providers"]}:
            return base
        n = 2
        while f"{base}-{n}" in {p["id"] for p in self._data["providers"]}:
            n += 1
        return f"{base}-{n}"

    # ------------------------------------------------------------------ 查询
    def active_id(self) -> str:
        return str(self._data.get("active") or "")

    def active_cfg(self) -> dict[str, Any] | None:
        """返回当前生效提供商的配置（不存在则 None）。"""
        return self.get(self.active_id())

    def get(self, provider_id: str) -> dict[str, Any] | None:
        for item in self._data["providers"]:
            if item["id"] == provider_id:
                return dict(item)                      # 返回副本，调用方改不到内部状态
        return None

    def list(self) -> list[dict[str, Any]]:
        """全部提供商实例（副本），供 WebUI 列表展示。"""
        return [dict(p) for p in self._data["providers"]]

    def presets(self) -> list[dict[str, Any]]:
        return [dict(p) for p in (self._data.get("presets") or PRESETS)]

    def public_list(self) -> list[dict[str, Any]]:
        """给前端用的列表：附 is_active 标记，并**移除密钥明文**（只保留 has_key）。

        前端编辑表单留空即表示「不修改密钥」（见 update 的空值处理），
        因此这里不回传 api_key，避免密钥随页面/日志外泄。
        """
        out = []
        for item in self.list():
            item.pop("api_key", None)                    # 不回明文
            item["is_active"] = item["id"] == self.active_id()
            item["has_key"] = bool(self.get(item["id"]).get("api_key"))
            out.append(item)
        return out

    # ------------------------------------------------------------------ 增删改切
    def add(self, entry: dict[str, Any]) -> dict[str, Any]:
        """新增一个提供商实例。未给 id 时自动生成。"""
        if not str(entry.get("id") or "").strip():
            entry = {**entry, "id": self._gen_id(entry)}
        if self.get(entry["id"]) is not None:
            raise ProviderError(f"提供商 id 已存在：{entry['id']}")
        item = self._normalize(entry)
        self._validate(item)
        self._data["providers"].append(item)
        if not self.active_id():                        # 第一个实例自动成为 active
            self._data["active"] = item["id"]
            self.gateway.reconfigure(item)
            self.gateway.provider_id = item["id"]
        self.save()
        return dict(item)

    def update(self, provider_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """修改一个提供商实例；若它正是 active，则立刻热更新网关。

        约定：patch 里的 api_key 为空串/None 表示「不改密钥」（前端看不到明文，
        留空即保持原值），避免一次保存就把密钥擦掉。
        """
        for idx, item in enumerate(self._data["providers"]):
            if item["id"] != provider_id:
                continue
            clean = {k: v for k, v in patch.items() if not (k == "api_key" and not v)}
            merged = self._normalize({**item, **clean, "id": provider_id})
            self._validate(merged)
            self._data["providers"][idx] = merged
            if provider_id == self.active_id():         # active 被改 → 立即生效
                self.gateway.reconfigure(merged)
            self.save()
            return dict(merged)
        raise ProviderError(f"提供商不存在：{provider_id}")

    def remove(self, provider_id: str) -> None:
        """删除一个提供商实例；删到 active 时自动切到剩余的第一个。"""
        before = len(self._data["providers"])
        self._data["providers"] = [p for p in self._data["providers"] if p["id"] != provider_id]
        if len(self._data["providers"]) == before:
            raise ProviderError(f"提供商不存在：{provider_id}")
        if self.active_id() == provider_id:
            nxt = self._data["providers"][0]["id"] if self._data["providers"] else ""
            self._data["active"] = nxt
            if nxt:
                self.gateway.reconfigure(self.get(nxt) or {})
                self.gateway.provider_id = nxt
        self.save()

    def activate(self, provider_id: str) -> dict[str, Any]:
        """把某个提供商设为生效项，并热更新网关（无需重启进程）。"""
        item = self.get(provider_id)
        if item is None:
            raise ProviderError(f"提供商不存在：{provider_id}")
        self._data["active"] = provider_id
        self.gateway.reconfigure(item)
        self.gateway.provider_id = provider_id
        self.save()
        logger.info("已切换 LLM 提供商：%s（模型 %s）", provider_id, item.get("model"))
        return dict(item)

    def active_info(self) -> dict[str, Any]:
        """当前生效提供商的摘要（供 /api/status 展示）。"""
        item = self.active_cfg() or {}
        return {"id": self.active_id(), "model": item.get("model", ""),
                "base_url": item.get("base_url", "")}

    # ------------------------------------------------------------------ 连通性测试
    async def test(self, provider_id: str) -> dict[str, Any]:
        """用指定提供商做一次极小调用，返回连通性结果（供「API 管理」页试调用）。"""
        item = self.get(provider_id)
        if item is None:
            raise ProviderError(f"提供商不存在：{provider_id}")
        # 临时网关：短超时、少 token，避免测试把界面卡住
        probe = LLMGateway({**item, "timeout": 15, "max_tokens": 16, "retry": 0})
        try:
            resp = await probe.complete(
                "你是连通性测试助手。",
                "请只回复两个字：正常",
                max_tokens=16,
                temperature=0.0,
                task="provider_test",
            )
            return {"ok": True, "model": resp.model or item.get("model", ""),
                    "reply": (resp.content or "").strip()[:200],
                    "usage": resp.usage or {}}
        except LLMError as e:
            return {"ok": False, "error": str(e)}
        finally:
            await probe.aclose()                        # 无论成败都要关掉临时连接池