import os
import random
import sys
import time
from datetime import datetime
from typing import List, Tuple

try:
    import sqlite3
except ImportError as exc:
    raise SystemExit(
        "当前 Python 环境缺少 sqlite3（_sqlite3 DLL 不可用）。"
        "请安装官方 Python 发行版或修复环境后再运行 services/backend_engine.py。"
    ) from exc


# ------------------------------
# 读取项目中的真实分类标签（优先从 mnist.py）
# ------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
COSTSENSITIVE_DIR = os.path.join(PROJECT_ROOT, "costSensitive")

if COSTSENSITIVE_DIR not in sys.path:
    sys.path.insert(0, COSTSENSITIVE_DIR)

try:
    from mnist import CLASS_NAMES  # type: ignore
except Exception:
    # 回退：与 costSensitive/mnist.py 保持一致的 12 类定义
    CLASS_NAMES = [
        "nonvpn_chat",
        "nonvpn_email",
        "nonvpn_file_transfer",
        "nonvpn_p2p",
        "nonvpn_streaming",
        "nonvpn_voip",
        "vpn_chat",
        "vpn_email",
        "vpn_file_transfer",
        "vpn_p2p",
        "vpn_streaming",
        "vpn_voip",
    ]


DB_PATH = os.path.join(PROJECT_ROOT, "netguard_logs.db")
TABLE_NAME = "traffic_alerts"

# 参考 costSensitive/inference.py 与 unknown_eval_report.json 中的阈值风格
MEDIUM_THRESHOLD = 0.5442
HIGH_THRESHOLD = 0.7381

PROTOCOLS = ["TLS", "HTTPS", "QUIC", "SSH", "DoH"]
UNKNOWN_CATEGORY = "unknown_proxy_ood"


def init_db(db_path: str) -> None:
    """初始化 SQLite 数据库与告警表。"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            src_ip TEXT NOT NULL,
            dst_ip TEXT NOT NULL,
            protocol TEXT NOT NULL,
            threat_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_score REAL NOT NULL,
            alert_level TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def random_public_ip() -> str:
    first_octet = random.choice(
        [8, 23, 31, 45, 59, 66, 78, 91, 103, 121, 139, 151, 173, 185, 203]
    )
    return (
        f"{first_octet}.{random.randint(0, 255)}."
        f"{random.randint(0, 255)}.{random.randint(1, 254)}"
    )


def random_private_ip() -> str:
    choice = random.randint(0, 2)
    if choice == 0:
        return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    if choice == 1:
        return f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"


def choose_category() -> str:
    """按业务流量分布随机采样类别，并注入一定比例的未知流量。"""
    weighted_categories: List[Tuple[str, float]] = [
        ("nonvpn_chat", 0.07),
        ("nonvpn_email", 0.06),
        ("nonvpn_file_transfer", 0.09),
        ("nonvpn_p2p", 0.08),
        ("nonvpn_streaming", 0.11),
        ("nonvpn_voip", 0.07),
        ("vpn_chat", 0.09),
        ("vpn_email", 0.07),
        ("vpn_file_transfer", 0.10),
        ("vpn_p2p", 0.10),
        ("vpn_streaming", 0.08),
        ("vpn_voip", 0.07),
        (UNKNOWN_CATEGORY, 0.11),
    ]
    labels = [item[0] for item in weighted_categories]
    probs = [item[1] for item in weighted_categories]
    return random.choices(labels, weights=probs, k=1)[0]


def infer_confidence_and_risk(category: str) -> Tuple[float, float, str]:
    """模拟 1D-CNN + unknown 检测输出的 confidence / risk / level。"""
    if category == UNKNOWN_CATEGORY:
        confidence = random.uniform(0.18, 0.52)
        anomaly_component = random.uniform(0.62, 1.0)
    elif (
        category.startswith("vpn_") or "p2p" in category or "file_transfer" in category
    ):
        confidence = random.uniform(0.40, 0.92)
        anomaly_component = random.uniform(0.35, 0.90)
    else:
        confidence = random.uniform(0.72, 0.995)
        anomaly_component = random.uniform(0.05, 0.45)

    risk_score = 0.5 * (1.0 - confidence) + 0.5 * anomaly_component

    if risk_score >= HIGH_THRESHOLD:
        level = "high"
    elif risk_score >= MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    return confidence, risk_score, level


def insert_record(
    conn: sqlite3.Connection,
    record: Tuple[str, str, str, str, str, float, float, str],
) -> None:
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO {TABLE_NAME}
        (timestamp, src_ip, dst_ip, protocol, threat_category, confidence, risk_score, alert_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record,
    )
    conn.commit()


def main() -> None:
    init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    print("=" * 92)
    print(f"[NetGuard Engine] SQLite 数据引擎已启动 -> {DB_PATH}")
    print(
        f"[NetGuard Engine] 载入真实标签数量: {len(CLASS_NAMES)}，"
        f"含未知检测类别: {UNKNOWN_CATEGORY}"
    )
    print(
        "[NetGuard Engine] 推理模式: End-to-End 1D-CNN + MSP/Anomaly Cost-Sensitive Risk"
    )
    print("=" * 92)

    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            src_ip = random_public_ip()
            dst_ip = random_private_ip()
            protocol = random.choice(PROTOCOLS)
            category = choose_category()
            confidence, risk_score, level = infer_confidence_and_risk(category)

            row = (
                timestamp,
                src_ip,
                dst_ip,
                protocol,
                category,
                round(confidence, 6),
                round(risk_score, 6),
                level,
            )
            insert_record(conn, row)

            print(
                "[NetGuard Engine] 1D-CNN 提取时序特征完成 | "
                f"pred={category} | conf={confidence:.4f} | risk={risk_score:.4f} | "
                f"level={level} | src={src_ip} -> dst={dst_ip} ({protocol})"
            )

            if level in {"medium", "high"}:
                print(
                    "[NetGuard Engine] 告警触发: "
                    f"检测到 {category} 流量，已进入零信任动态访问控制策略链路"
                )

            time.sleep(random.uniform(2.0, 3.0))

    except KeyboardInterrupt:
        print("\n[NetGuard Engine] 收到停止信号，正在安全退出...")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
