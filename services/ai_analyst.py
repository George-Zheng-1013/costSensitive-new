import json
import importlib
from collections import Counter
from datetime import datetime
from typing import Dict, List

from .ai_config import AIConfig


def _mask_ip(ip: str) -> str:
    parts = str(ip).split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.*.*"
    return str(ip)


class TrafficAIAnalyst:
    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self.available = False
        self.client = None
        if not cfg.enabled or not cfg.api_key:
            return

        try:
            openai_mod = importlib.import_module("openai")
            OpenAI = getattr(openai_mod, "OpenAI")
            self.client = OpenAI(
                api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s
            )
            self.available = True
        except Exception:
            self.available = False

    @staticmethod
    def _build_payload(records: List[Dict]) -> Dict:
        total = len(records)
        if total == 0:
            return {
                "total": 0,
                "alert_level_dist": {},
                "top_categories": [],
                "top_src_ip_masked": [],
                "avg_risk": 0.0,
                "unknown_ratio": 0.0,
                "evidence_samples": [],
            }

        level_counter = Counter(str(r.get("alert_level", "unknown")) for r in records)
        cat_counter = Counter(str(r.get("threat_category", "unknown")) for r in records)
        src_counter = Counter(
            _mask_ip(str(r.get("src_ip", "0.0.0.0"))) for r in records
        )
        avg_risk = sum(float(r.get("risk_score", 0.0) or 0.0) for r in records) / total
        unknown_cnt = sum(
            1
            for r in records
            if str(r.get("threat_category", "")) == "unknown_proxy_ood"
        )

        evidence_samples = []
        for r in records[:6]:
            evidence_samples.append(
                {
                    "threat": str(r.get("threat_category", "unknown")),
                    "risk_score": float(r.get("risk_score", 0.0) or 0.0),
                    "alert_level": str(r.get("alert_level", "low")),
                    "reason": str(r.get("explain_reason", ""))[:180],
                }
            )

        return {
            "total": total,
            "alert_level_dist": dict(level_counter),
            "top_categories": cat_counter.most_common(6),
            "top_src_ip_masked": src_counter.most_common(6),
            "avg_risk": round(avg_risk, 6),
            "unknown_ratio": round(unknown_cnt / total, 6),
            "evidence_samples": evidence_samples,
        }

    @staticmethod
    def _build_behavior_payload(records: List[Dict]) -> Dict:
        total = len(records)
        if total == 0:
            return {
                "total": 0,
                "known_category_dist": {},
                "protocol_dist": {},
                "top_src_ip_masked": [],
                "hourly_activity": {},
                "avg_risk": 0.0,
            }

        cat_counter = Counter(str(r.get("threat_category", "unknown")) for r in records)
        proto_counter = Counter(str(r.get("protocol", "UNKNOWN")) for r in records)
        src_counter = Counter(
            _mask_ip(str(r.get("src_ip", "0.0.0.0"))) for r in records
        )
        avg_risk = sum(float(r.get("risk_score", 0.0) or 0.0) for r in records) / total

        hourly = Counter()
        for r in records:
            ts = str(r.get("timestamp", "")).strip()
            if not ts:
                continue
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                hourly[f"{dt.hour:02d}"] += 1
            except Exception:
                continue

        return {
            "total": total,
            "known_category_dist": dict(cat_counter),
            "protocol_dist": dict(proto_counter),
            "top_src_ip_masked": src_counter.most_common(6),
            "hourly_activity": dict(hourly),
            "avg_risk": round(avg_risk, 6),
        }

    @staticmethod
    def _extract_json(text: str) -> Dict:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            frag = text[start : end + 1]
            try:
                return json.loads(frag)
            except Exception:
                return {}
        return {}

    def analyze_alerts(self, records: List[Dict]) -> Dict:
        if not self.available or self.client is None:
            return {}

        payload = self._build_payload(records)
        user_prompt = (
            "请根据以下告警窗口进行研判，并仅输出 JSON。\\n"
            "JSON schema: "
            "{scene: string, risk_level: low|medium|high|critical, summary: string, "
            "actions: string[], confidence: number, next_checks: string[]}\\n"
            f"告警窗口: {json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = ""
            if resp.choices and resp.choices[0].message:
                content = resp.choices[0].message.content or ""
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return {}
            return parsed
        except Exception:
            return {}

    def analyze_known_behavior(self, records: List[Dict]) -> Dict:
        if not self.available or self.client is None:
            return {}

        payload = self._build_behavior_payload(records)
        user_prompt = (
            "请根据以下已知类别流量窗口分析用户行为模式，并仅输出 JSON。\n"
            "JSON schema: "
            "{scene: string, behavior_profile: string, behavior_tag: 办公行为|娱乐行为|可疑自动化行为, "
            "behavior_tag_confidence: number(0~1), risk_level: low|medium|high, "
            "summary: string, suspicious_patterns: string[], recommendations: string[], confidence: number}\n"
            f"已知流量窗口: {json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = ""
            if resp.choices and resp.choices[0].message:
                content = resp.choices[0].message.content or ""
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return {}
            if "behavior_tag" not in parsed:
                parsed["behavior_tag"] = "办公行为"
            try:
                parsed["behavior_tag_confidence"] = float(
                    parsed.get("behavior_tag_confidence", 0.6)
                )
            except Exception:
                parsed["behavior_tag_confidence"] = 0.6
            parsed["behavior_tag_confidence"] = max(
                0.0, min(1.0, parsed["behavior_tag_confidence"])
            )
            return parsed
        except Exception:
            return {}

    def analyze(self, records: List[Dict]) -> Dict:
        # Backward-compatible alias for existing call sites.
        return self.analyze_alerts(records)
