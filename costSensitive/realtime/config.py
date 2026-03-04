from dataclasses import dataclass


@dataclass
class RealtimeConfig:
    packet_len: int = 784
    png_size: int = 28
    pad_len: int = 50
    flow_timeout_s: float = 10.0
    sweep_interval_s: float = 1.0
    max_active_flows: int = 50000
    batch_size: int = 64
    batch_wait_ms: int = 50
    emit_once_per_flow: bool = True
