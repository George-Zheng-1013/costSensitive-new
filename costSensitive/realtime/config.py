from dataclasses import dataclass


@dataclass
class RealtimeConfig:
    num_packets: int = 12
    packet_len: int = 256
    byte_pad_token: int = 256
    flow_timeout_s: float = 10.0
    sweep_interval_s: float = 1.0
    max_active_flows: int = 50000
    infer_batch_size: int = 32
    batch_wait_ms: int = 50
    emit_once_per_flow: bool = True
