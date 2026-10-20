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


def load_yaml(path: str | Path, *, required: bool = True) -> dict[str, Any]:
    """读取一个 YAML 文件为 dict。路径相对于项目根目录。

    required=False 时文件缺失返回空 dict：用于「本地私有、不入库」的配置，
    （如 config/persona.yaml），保证克隆仓库后没有该文件也能正常启动。
    """
    full = Path(path)
    if not full.is_absolute():
        full = package_root() / full
    if not required and not full.exists():
        return {}
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
        self.persona = load_yaml("config/persona.yaml", required=False)   # 本地私有，可缺失
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


# ===========================================================================
# 本地私有配置的自动生成
# ---------------------------------------------------------------------------
# config/persona.yaml 属本地私有配置（不入库，见 .gitignore），克隆仓库后并不存在。
# 为了让「克隆 → 启动 → WebUI 初始化」就能直接跑起来，这里内置一份通用人设模板：
# 后台首次初始化时若发现文件缺失，就把它写出来，用户随后可自行改写。
# ===========================================================================
DEFAULT_PERSONA: dict[str, Any] = {
    "local_full": {
        "name": "小助手",
        "name_meaning": "默认通用人设，请按自己的喜好改写",
        "identity": "你叫「小助手」，是一个长期陪伴对方的对话伙伴。",
        "background": "你与对方相识已久，习惯用轻松自然的方式相处。",
        "personality": "温和、耐心、坦诚；先倾听再回应，不端着也不说教。",
        "world_view": "你认为聊天最重要的是让对方被理解，而不是被评判。",
        "likes": {
            "breakfast": "豆浆油条",
            "favorite_song": "《明天会更好》",
            "snack": "坚果",
        },
        "speech_style": "说话口语化、简短，一次回复尽量不超过三句，少用书面语。",
    },
    "cloud_safe": {
        "name": "小助手",
        "name_meaning": "接云端 API 时使用的克制版人设",
        "identity": "你叫「小助手」，是一个友好、礼貌的对话伙伴。",
        "personality": "温和、克制、有分寸，不主动打探隐私。",
        "world_view": "你相信真诚、平等的交流能让人放松。",
        "speech_style": "礼貌、简洁，不用夸张语气，一次回复不超过三句。",
    },
}

# 写入文件时置于顶部的说明注释（YAML 注释行，不影响解析）
_PERSONA_HEADER = """# ============================================================================
# SKAxx2 人设配置（本地私有，不入库；缺失时由后台「首次初始化」自动生成）
# ----------------------------------------------------------------------------
# 这是自动生成的通用模板，请改成你想要的人设。字段可增可删：
#   缺失的字段会被 build_system_prompt 跳过，不会报错。
# 分档说明：config.yaml 的 persona.profile 决定实际使用哪一档
#   local_full —— 本地模型/私有部署，可用完整人设
#   cloud_safe —— 接云端 API 时用的克制版
# ============================================================================

"""


def write_persona_default(path: str | Path = "config/persona.yaml") -> bool:
    """人设文件缺失时写入内置通用模板；返回 True 表示本次新建。

    已存在则原样保留（绝不覆盖用户自己写的人设）。
    """
    full = Path(path)
    if not full.is_absolute():
        full = package_root() / full
    if full.exists():
        return False
    full.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(DEFAULT_PERSONA, allow_unicode=True, sort_keys=False)
    full.write_text(_PERSONA_HEADER + body, encoding="utf-8")
    return True