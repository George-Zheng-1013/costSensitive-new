import os
import json
from dataclasses import dataclass


@dataclass
class AIConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Pro/zai-org/GLM-4.7"
    system_prompt: str = (
        "你是网络流量安全分析助手。"
        "请基于结构化告警信息进行场景判断、风险评估和处置建议，"
        "不得编造不存在的证据，输出必须是 JSON。"
    )
    analyze_every_n: int = 30
    window_size: int = 60
    timeout_s: float = 25.0

    @staticmethod
    def default_config_path() -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "config", "ai_config.json")

    @staticmethod
    def from_file(path: str) -> "AIConfig":
        cfg = AIConfig()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return cfg

            cfg.enabled = bool(raw.get("enabled", cfg.enabled))
            cfg.api_key = str(raw.get("api_key", cfg.api_key)).strip()
            cfg.base_url = str(raw.get("base_url", cfg.base_url)).strip()
            cfg.model = str(raw.get("model", cfg.model)).strip()
            cfg.system_prompt = str(raw.get("system_prompt", cfg.system_prompt)).strip()
            cfg.analyze_every_n = max(
                1, int(raw.get("analyze_every_n", cfg.analyze_every_n))
            )
            cfg.window_size = max(10, int(raw.get("window_size", cfg.window_size)))
            cfg.timeout_s = max(5.0, float(raw.get("timeout_s", cfg.timeout_s)))
            return cfg
        except Exception:
            return AIConfig()

    @staticmethod
    def from_env() -> "AIConfig":
        config_path = os.getenv("NETGUARD_AI_CONFIG", AIConfig.default_config_path())
        base = AIConfig.from_file(config_path)

        enabled_env = os.getenv("NETGUARD_AI_ENABLED")
        enabled_raw = enabled_env.strip().lower() if enabled_env is not None else ""
        enabled_override = enabled_raw in {"1", "true", "yes", "on"}
        enabled_has_override = enabled_raw in {
            "1",
            "true",
            "yes",
            "on",
            "0",
            "false",
            "no",
            "off",
        }
        return AIConfig(
            enabled=enabled_override if enabled_has_override else base.enabled,
            api_key=os.getenv("NETGUARD_AI_API_KEY", base.api_key).strip(),
            base_url=os.getenv("NETGUARD_AI_BASE_URL", base.base_url).strip(),
            model=os.getenv("NETGUARD_AI_MODEL", base.model).strip(),
            system_prompt=os.getenv(
                "NETGUARD_AI_SYSTEM_PROMPT",
                base.system_prompt,
            ).strip(),
            analyze_every_n=max(
                1, int(os.getenv("NETGUARD_AI_EVERY_N", str(base.analyze_every_n)))
            ),
            window_size=max(
                10, int(os.getenv("NETGUARD_AI_WINDOW", str(base.window_size)))
            ),
            timeout_s=max(
                5.0, float(os.getenv("NETGUARD_AI_TIMEOUT_S", str(base.timeout_s)))
            ),
        )
