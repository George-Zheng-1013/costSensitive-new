import socket
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import dpkt
import numpy as np

from .config import RealtimeConfig


FlowKey = Tuple[str, str, int, int, int]


@dataclass
class PacketEvent:
    ts: float
    flow_key: FlowKey
    payload: bytes
    fin: bool = False
    rst: bool = False


@dataclass
class FlowSample:
    sample_id: int
    flow_key: FlowKey
    ts: float
    trigger: str
    session_bytes: np.ndarray
    packet_mask: np.ndarray


@dataclass
class FlowState:
    packets: List[np.ndarray] = field(default_factory=list)
    last_seen: float = 0.0
    emitted: bool = False


def _safe_ip(ip_bytes: bytes) -> str:
    try:
        return socket.inet_ntoa(ip_bytes)
    except OSError:
        return "0.0.0.0"


def _canonical_flow_key(
    src_ip: str,
    src_port: int,
    dst_ip: str,
    dst_port: int,
    proto: int,
) -> FlowKey:
    left = (src_ip, src_port)
    right = (dst_ip, dst_port)
    if left <= right:
        return src_ip, dst_ip, src_port, dst_port, proto
    return dst_ip, src_ip, dst_port, src_port, proto


def _payload_to_packet_vec(
    payload: bytes, packet_len: int, pad_token: int
) -> np.ndarray:
    arr = np.full((packet_len,), pad_token, dtype=np.uint16)
    if len(payload) == 0:
        return arr
    packet = np.frombuffer(payload[:packet_len], dtype=np.uint8).astype(np.uint16)
    arr[: packet.shape[0]] = packet
    return arr


def parse_packet_to_event(
    ts: float, raw_pkt: bytes, datalink: int
) -> Optional[PacketEvent]:
    ip_pkt = None
    try:
        if datalink == 1:
            eth = dpkt.ethernet.Ethernet(raw_pkt)
            if isinstance(eth.data, dpkt.ip.IP):
                ip_pkt = eth.data
        elif datalink == 101:
            ip_pkt = dpkt.ip.IP(raw_pkt)
        elif datalink == 113:
            sll = dpkt.sll.SLL(raw_pkt)
            if isinstance(sll.data, dpkt.ip.IP):
                ip_pkt = sll.data
    except Exception:
        return None

    if ip_pkt is None:
        return None

    src_ip = _safe_ip(getattr(ip_pkt, "src", b"\x00\x00\x00\x00"))
    dst_ip = _safe_ip(getattr(ip_pkt, "dst", b"\x00\x00\x00\x00"))

    payload = None
    fin = False
    rst = False

    if isinstance(ip_pkt.data, dpkt.tcp.TCP):
        tcp = ip_pkt.data
        payload = bytes(tcp.data)
        flags = int(getattr(tcp, "flags", 0))
        fin = bool(flags & dpkt.tcp.TH_FIN)
        rst = bool(flags & dpkt.tcp.TH_RST)
        if flags == dpkt.tcp.TH_ACK and len(payload) == 0:
            return None
        flow_key = _canonical_flow_key(
            src_ip,
            int(getattr(tcp, "sport", 0)),
            dst_ip,
            int(getattr(tcp, "dport", 0)),
            6,
        )
    elif isinstance(ip_pkt.data, dpkt.udp.UDP):
        udp = ip_pkt.data
        payload = bytes(udp.data)
        flow_key = _canonical_flow_key(
            src_ip,
            int(getattr(udp, "sport", 0)),
            dst_ip,
            int(getattr(udp, "dport", 0)),
            17,
        )
    else:
        return None

    return PacketEvent(ts=ts, flow_key=flow_key, payload=payload, fin=fin, rst=rst)


class FlowAssembler:
    def __init__(self, cfg: RealtimeConfig):
        self.cfg = cfg
        self.active: "OrderedDict[FlowKey, FlowState]" = OrderedDict()
        self.sample_seq = 0

    def _touch(self, key: FlowKey) -> None:
        self.active.move_to_end(key)

    def _evict_if_needed(self) -> None:
        while len(self.active) > self.cfg.max_active_flows:
            self.active.popitem(last=False)

    def _finalize(
        self, key: FlowKey, state: FlowState, ts: float, trigger: str
    ) -> Optional[FlowSample]:
        if state.emitted and self.cfg.emit_once_per_flow:
            return None
        if len(state.packets) == 0:
            return None

        session_bytes = np.full(
            (self.cfg.num_packets, self.cfg.packet_len),
            self.cfg.byte_pad_token,
            dtype=np.uint16,
        )
        packet_mask = np.zeros((self.cfg.num_packets,), dtype=np.bool_)

        real_count = min(len(state.packets), self.cfg.num_packets)
        session_bytes[:real_count] = np.stack(state.packets[:real_count], axis=0)
        packet_mask[:real_count] = True

        state.emitted = True
        self.sample_seq += 1
        return FlowSample(
            sample_id=self.sample_seq,
            flow_key=key,
            ts=ts,
            trigger=trigger,
            session_bytes=session_bytes,
            packet_mask=packet_mask,
        )

    def feed(self, pkt: PacketEvent) -> List[FlowSample]:
        out: List[FlowSample] = []
        key = pkt.flow_key
        state = self.active.get(key)
        if state is None:
            state = FlowState(last_seen=pkt.ts)
            self.active[key] = state

        self._touch(key)
        state.last_seen = pkt.ts

        if not (state.emitted and self.cfg.emit_once_per_flow):
            state.packets.append(
                _payload_to_packet_vec(
                    pkt.payload,
                    packet_len=self.cfg.packet_len,
                    pad_token=self.cfg.byte_pad_token,
                )
            )

        if len(state.packets) >= self.cfg.num_packets:
            sample = self._finalize(key, state, pkt.ts, "packet_count")
            if sample is not None:
                out.append(sample)
            if self.cfg.emit_once_per_flow:
                self.active.pop(key, None)

        if key in self.active and (pkt.fin or pkt.rst):
            sample = self._finalize(key, state, pkt.ts, "fin_rst")
            if sample is not None:
                out.append(sample)
            self.active.pop(key, None)

        self._evict_if_needed()
        return out

    def sweep_timeouts(self, now_ts: Optional[float] = None) -> List[FlowSample]:
        now_ts = now_ts if now_ts is not None else time.time()
        timeout_keys: List[FlowKey] = []
        out: List[FlowSample] = []

        for key, state in self.active.items():
            if now_ts - state.last_seen >= self.cfg.flow_timeout_s:
                timeout_keys.append(key)

        for key in timeout_keys:
            state = self.active.get(key)
            if state is None:
                continue
            sample = self._finalize(key, state, now_ts, "timeout")
            if sample is not None:
                out.append(sample)
            self.active.pop(key, None)

        return out
