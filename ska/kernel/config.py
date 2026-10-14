"""配置加载：YAML → 类型化的简单容器。

设计取舍：
    方案要求「声明式配置 + 内核可读」，但对配置项本身没有强类型校验需求，
    所以这里提供两个轻量工具，而不是引 pydantic-settings：
      - load_yaml：把单个 YAML 文件读成 dict；
      - 通过 package_root() 统一换算项目根目录下的相对路径。

如果后续配置项增多、需要默认值/校验，可以平滑替换为 dataclass 方案。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    """项目根目录 = 本文件所在目录的上两级（ska/kernel/config.py → 项目根）。"""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取一个 YAML 文件为 dict。路径相对于项目根目录。"""
    full = Path(path)
    if not full.is_absolute():
        full = package_root() / full
    with full.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


class Config:
    """顶层配置容器：把主配置 + 人设 + 定时任务表聚合在一起，方便各处引用。

    Attributes:
        cfg:         config.yaml 内容（dict）
        persona:     persona.yaml 内容（dict）
        scheduler:   scheduler.yaml 内容（dict）
    """

    def __init__(self) -> None:
        self.cfg = load_yaml("config/config.yaml")
        self.persona = load_yaml("config/persona.yaml")
        self.scheduler = load_yaml("config/scheduler.yaml")

    # ------------------------------------------------------------------ 便捷访问
    def get(self, key: str, default: Any = None) -> Any:
        """读取主配置顶层键，缺失时返回 default。"""
        return self.cfg.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        """读取主配置的某个小节（llm / throttle / memory / qq / cli ...），缺失返回 {}。"""
        section = self.cfg.get(name, {})
        return section if isinstance(section, dict) else {}

    @property
    def llm(self) -> dict[str, Any]:
        return self.section("llm")

    @property
    def throttle(self) -> dict[str, Any]:
        return self.section("throttle")

    @property
    def memory(self) -> dict[str, Any]:
        return self.section("memory")

    def persona_profile(self) -> dict[str, Any]:
        """按 persona.profile 选择人设档，返回该档的全部字段。"""
        profile = self.cfg.get("persona", {}).get("profile", "local_full")
        return self.persona.get(profile, self.persona.get("local_full", {}))

    def build_system_prompt(self) -> str:
        """由人设字段拼出完整 system_prompt（分档选择）。

        拼装顺序：身份 → 背景 → 性格 → 世界观 → 喜好 → 口吻规则。
        每块字段允许缺失，缺失就跳过，保证配置裁剪不报错。
        """
        p = self.persona_profile()
        blocks = [
            p.get("identity"),
            p.get("background"),
            p.get("personality"),
            p.get("world_view"),
        ]
        likes = p.get("likes") or {}
        if likes:
            like_lines = [f"- 早饭喜欢吃{likes['breakfast']}。"
                          if likes.get("breakfast") else "",
                          f"- 最喜欢的歌曲是{likes['favorite_song']}。"
                          if likes.get("favorite_song") else "",
                          f"- 零食喜欢吃{likes['snack']}。"
                          if likes.get("snack") else ""]
            blocks.append("你的日常喜好：\n" + "\n".join(l for l in like_lines if l))
        blocks.append(p.get("speech_style"))
        return "\n\n".join(b for b in blocks if b).strip()