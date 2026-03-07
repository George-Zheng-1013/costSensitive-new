import argparse
import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

try:
    import sqlite3
except ImportError as exc:
    raise SystemExit(
        "Current Python environment is missing sqlite3 (_sqlite3 DLL unavailable). "
        "Please switch to a Python environment with sqlite support and retry."
    ) from exc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "netguard_logs.db")
TABLE_NAME = "traffic_alerts"

REALTIME_JSONL = os.path.join(
    PROJECT_ROOT,
    "costSensitive",
    "pytorch_model",
    "realtime_sessions",
    "realtime_predictions.jsonl",
)

MEDIUM_THRESHOLD = 1.0
HIGH_THRESHOLD = 1.5


def init_db(db_path: str) -> None:
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
            alert_level TEXT NOT NULL,
            centroid_distance REAL,
            centroid_threshold REAL,
            explain_reason TEXT,
            evidence_json TEXT,
            packet_contrib_json TEXT,
            byte_heatmap_json TEXT
        )
        """
    )

    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    existing_cols = {row[1] for row in cursor.fetchall()}
    extra_cols = {
        "centroid_distance": "REAL",
        "centroid_threshold": "REAL",
        "explain_reason": "TEXT",
        "evidence_json": "TEXT",
        "packet_contrib_json": "TEXT",
        "byte_heatmap_json": "TEXT",
    }
    for col, col_type in extra_cols.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()


def clear_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {TABLE_NAME}")
    conn.commit()
    conn.close()


def protocol_name(proto_num: int) -> str:
    if proto_num == 6:
        return "TCP"
    if proto_num == 17:
        return "UDP"
    return f"IP_{proto_num}"


def parse_flow_key(flow_key_raw: str) -> Tuple[str, str, str]:
    try:
        tup = ast.literal_eval(flow_key_raw)
        if isinstance(tup, tuple) and len(tup) >= 5:
            src_ip = str(tup[0])
            dst_ip = str(tup[1])
            proto = protocol_name(int(tup[4]))
            return src_ip, dst_ip, proto
    except Exception:
        pass
    return "0.0.0.0", "0.0.0.0", "UNKNOWN"


def normalize_timestamp(record: Dict) -> str:
    infer_ts = record.get("infer_ts")
    if infer_ts is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        return datetime.fromtimestamp(float(infer_ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_alert_level(anomaly_score: float, raw_level: Optional[str]) -> str:
    if isinstance(raw_level, str) and raw_level in {"low", "medium", "high"}:
        return raw_level
    if anomaly_score >= HIGH_THRESHOLD:
        return "high"
    if anomaly_score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def build_explainability(
    confidence: float,
    risk_score: float,
    alert_level: str,
    threat_category: str,
    centroid_distance: float,
    centroid_threshold: float,
    is_unknown: int,
) -> Tuple[str, str]:
    if centroid_threshold > 1e-12:
        distance_ratio = centroid_distance / centroid_threshold
    else:
        distance_ratio = 0.0

    if is_unknown == 1:
        reason = (
            "判为 unknown：会话 embedding 到预测类中心距离超过阈值，"
            f"distance/threshold={distance_ratio:.3f}，且风险分={risk_score:.3f}。"
        )
    else:
        reason = (
            "判为已知类：会话 embedding 与预测类中心距离在阈值内，"
            f"distance/threshold={distance_ratio:.3f}，风险分={risk_score:.3f}。"
        )

    evidence = {
        "threat_category": threat_category,
        "confidence": round(confidence, 6),
        "risk_score": round(risk_score, 6),
        "alert_level": alert_level,
        "centroid_distance": round(centroid_distance, 6),
        "centroid_threshold": round(centroid_threshold, 6),
        "distance_ratio": round(distance_ratio, 6),
        "is_unknown": int(is_unknown),
        "rules": {
            "unknown_if_distance_gt_threshold": bool(
                centroid_distance > centroid_threshold
            ),
            "high_if_risk_ge": HIGH_THRESHOLD,
            "medium_if_risk_ge": MEDIUM_THRESHOLD,
        },
    }
    return reason, json.dumps(evidence, ensure_ascii=False)


def convert_record(
    record: Dict,
) -> Tuple[
    str,
    str,
    str,
    str,
    str,
    float,
    float,
    str,
    float,
    float,
    str,
    str,
    str,
    str,
]:
    src_ip, dst_ip, protocol = parse_flow_key(str(record.get("flow_key", "")))
    timestamp = normalize_timestamp(record)

    threat_category = str(
        record.get("final_pred_name")
        or record.get("pred_name")
        or record.get("pred")
        or "unknown_proxy_ood"
    )

    try:
        confidence = float(record.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    try:
        risk_score = float(record.get("anomaly_score", 0.0))
    except Exception:
        risk_score = 0.0

    try:
        centroid_distance = float(record.get("centroid_distance", 0.0))
    except Exception:
        centroid_distance = 0.0

    try:
        centroid_threshold = float(record.get("centroid_threshold", 0.0))
    except Exception:
        centroid_threshold = 0.0

    try:
        is_unknown = int(record.get("is_unknown", 0))
    except Exception:
        is_unknown = 0

    alert_level = normalize_alert_level(risk_score, record.get("alert_level"))
    explain_reason, evidence_json = build_explainability(
        confidence=confidence,
        risk_score=risk_score,
        alert_level=alert_level,
        threat_category=threat_category,
        centroid_distance=centroid_distance,
        centroid_threshold=centroid_threshold,
        is_unknown=is_unknown,
    )
    packet_contrib_json = str(record.get("packet_contrib_json", "") or "")
    byte_heatmap_json = str(record.get("byte_heatmap_json", "") or "")

    return (
        timestamp,
        src_ip,
        dst_ip,
        protocol,
        threat_category,
        round(confidence, 6),
        round(risk_score, 6),
        alert_level,
        round(centroid_distance, 6),
        round(centroid_threshold, 6),
        explain_reason,
        evidence_json,
        packet_contrib_json,
        byte_heatmap_json,
    )


def insert_record(
    conn: sqlite3.Connection,
    row: Tuple[
        str,
        str,
        str,
        str,
        str,
        float,
        float,
        str,
        float,
        float,
        str,
        str,
        str,
        str,
    ],
) -> None:
    cursor = conn.cursor()
    cursor.execute(
        f"""
        INSERT INTO {TABLE_NAME}
        (
            timestamp,
            src_ip,
            dst_ip,
            protocol,
            threat_category,
            confidence,
            risk_score,
            alert_level,
            centroid_distance,
            centroid_threshold,
            explain_reason,
            evidence_json,
            packet_contrib_json,
            byte_heatmap_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )
    conn.commit()


def consume_jsonl_forever(
    jsonl_path: str,
    db_path: str,
    poll_s: float = 1.0,
    from_start: bool = False,
) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)

    print("=" * 92)
    print(f"[NetGuard Engine] SQLite writer started -> {db_path}")
    print(f"[NetGuard Engine] source realtime jsonl -> {jsonl_path}")
    print("[NetGuard Engine] mode: live model output -> sqlite dashboard table")
    print("=" * 92)

    file_pos = (
        0
        if from_start
        else (os.path.getsize(jsonl_path) if os.path.exists(jsonl_path) else 0)
    )
    inserted = 0
    last_wait_log = 0.0

    if file_pos > 0 and not from_start:
        print(f"[NetGuard Engine] tail mode enabled, start from byte offset={file_pos}")
    if from_start:
        print("[NetGuard Engine] replay mode enabled, start from beginning of jsonl")

    try:
        while True:
            if not os.path.exists(jsonl_path):
                now = time.time()
                if now - last_wait_log >= 10.0:
                    print(f"[NetGuard Engine] waiting for source file: {jsonl_path}")
                    last_wait_log = now
                time.sleep(poll_s)
                continue

            file_size = os.path.getsize(jsonl_path)
            if file_size < file_pos:
                file_pos = 0

            with open(jsonl_path, "r", encoding="utf-8") as f:
                f.seek(file_pos)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        row = convert_record(record)
                        insert_record(conn, row)
                        inserted += 1
                        if inserted % 20 == 0:
                            print(f"[NetGuard Engine] inserted={inserted}")
                    except Exception as exc:
                        print(f"[WARN] skip malformed record: {exc}")
                file_pos = f.tell()

            if inserted == 0:
                now = time.time()
                if now - last_wait_log >= 10.0:
                    print(
                        "[NetGuard Engine] no new records yet, waiting for realtime writer..."
                    )
                    last_wait_log = now

            time.sleep(poll_s)

    except KeyboardInterrupt:
        print("\n[NetGuard Engine] stop signal received, exiting...")
    finally:
        conn.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consume realtime_predictions.jsonl and write SQLite dashboard records"
    )
    parser.add_argument("--jsonl", default=REALTIME_JSONL)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="replay jsonl from the beginning instead of tailing only new lines",
    )
    parser.add_argument(
        "--clear-table",
        action="store_true",
        help="clear existing traffic_alerts rows before consuming",
    )
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        default=True,
        help="auto-start live capture/inference process in background (default: on)",
    )
    parser.add_argument(
        "--no-auto-capture",
        action="store_false",
        dest="auto_capture",
        help="disable auto-start of live capture/inference process",
    )
    parser.add_argument(
        "--capture-source",
        default=None,
        help="network interface for costSensitive/main_realtime.py --source",
    )
    parser.add_argument(
        "--capture-device",
        default=None,
        help="device for costSensitive/main_realtime.py --device (cpu/cuda)",
    )
    return parser.parse_args()


def start_capture_subprocess(
    source: Optional[str],
    device: Optional[str],
) -> subprocess.Popen:
    cost_dir = os.path.join(PROJECT_ROOT, "costSensitive")
    main_realtime = os.path.join(cost_dir, "main_realtime.py")
    if not os.path.exists(main_realtime):
        raise FileNotFoundError(f"main_realtime.py not found: {main_realtime}")

    cmd = [
        sys.executable,
        main_realtime,
        "--mode",
        "live",
        "--live-duration-seconds",
        "0",
    ]
    if source:
        cmd.extend(["--source", source])
    if device:
        cmd.extend(["--device", device])

    print("[NetGuard Engine] auto-start capture command:")
    print("  " + " ".join(cmd))
    # Launch from costSensitive so relative paths (model/outputs) stay correct.
    return subprocess.Popen(cmd, cwd=cost_dir)


def main() -> None:
    args = parse_args()
    capture_proc = None
    if args.clear_table:
        init_db(args.db_path)
        clear_table(args.db_path)
        print("[NetGuard Engine] traffic_alerts table cleared")

    if args.auto_capture:
        capture_proc = start_capture_subprocess(
            source=args.capture_source,
            device=args.capture_device,
        )
        print(f"[NetGuard Engine] capture process started pid={capture_proc.pid}")

    try:
        consume_jsonl_forever(
            jsonl_path=args.jsonl,
            db_path=args.db_path,
            poll_s=args.poll_s,
            from_start=args.from_start,
        )
    finally:
        if capture_proc is not None and capture_proc.poll() is None:
            print("[NetGuard Engine] stopping capture process...")
            capture_proc.terminate()
            try:
                capture_proc.wait(timeout=5)
            except Exception:
                capture_proc.kill()


if __name__ == "__main__":
    main()
