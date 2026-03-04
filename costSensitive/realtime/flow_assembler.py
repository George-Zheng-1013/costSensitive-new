import socket
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import dpkt

from .config import RealtimeConfig


FlowKey = Tuple[str, str, int, int, int]


@dataclass
class PacketEvent:
    ts: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    payload: bytes
    fin: bool = False
    rst: bool = False


@dataclass
class FlowSample:
    sample_id: int
    flow_key: FlowKey
    ts: float
    trigger: str
    raw784: bytes


@dataclass
class FlowState:
    payload_buf: bytearray = field(default_factory=bytearray)
    last_seen: float = 0.0
    emitted: bool = False


def _ip_to_str(ip_bytes: bytes) -> str:
    try:
        return socket.inet_ntoa(ip_bytes)
    except OSError:
        return "0.0.0.0"


def sanitize_ip_packet(raw_ip_packet: bytes) -> Optional[Tuple[bytes, bool, bool]]:
    try:
        ip_pkt = dpkt.ip.IP(raw_ip_packet)
    except Exception:
        return None

    ip_pkt.src = b"\x00\x00\x00\x00"
    ip_pkt.dst = b"\x00\x00\x00\x00"

    fin = False
    rst = False

    if isinstance(ip_pkt.data, dpkt.tcp.TCP):
        trans = ip_pkt.data
        fin = bool(trans.flags & dpkt.tcp.TH_FIN)
        rst = bool(trans.flags & dpkt.tcp.TH_RST)
        trans.sport = 0
        trans.dport = 0
        if trans.flags == dpkt.tcp.TH_ACK and len(trans.data) == 0:
            return None
    elif isinstance(ip_pkt.data, dpkt.udp.UDP):
        trans = ip_pkt.data
        trans.sport = 0
        trans.dport = 0
    else:
        return None

    payload = bytes(ip_pkt)
    if len(payload) == 0:
        return None
    return payload, fin, rst


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

    src_ip = _ip_to_str(ip_pkt.src)
    dst_ip = _ip_to_str(ip_pkt.dst)

    if isinstance(ip_pkt.data, dpkt.tcp.TCP):
        proto = 6
        src_port = int(ip_pkt.data.sport)
        dst_port = int(ip_pkt.data.dport)
    elif isinstance(ip_pkt.data, dpkt.udp.UDP):
        proto = 17
        src_port = int(ip_pkt.data.sport)
        dst_port = int(ip_pkt.data.dport)
    else:
        return None

    sanitized = sanitize_ip_packet(bytes(ip_pkt))
    if sanitized is None:
        return None

    payload, fin, rst = sanitized
    return PacketEvent(
        ts=ts,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=proto,
        payload=payload,
        fin=fin,
        rst=rst,
    )


class FlowAssembler:
    def __init__(self, cfg: RealtimeConfig):
        self.cfg = cfg
        self.active: "OrderedDict[FlowKey, FlowState]" = OrderedDict()
        self.sample_seq = 0

    def _touch(self, key: FlowKey):
        self.active.move_to_end(key)

    def _evict_if_needed(self):
        while len(self.active) > self.cfg.max_active_flows:
            self.active.popitem(last=False)

    def _emit_from_state(
        self, key: FlowKey, state: FlowState, ts: float, trigger: str
    ) -> Optional[FlowSample]:
        if state.emitted and self.cfg.emit_once_per_flow:
            return None

        if len(state.payload_buf) >= self.cfg.packet_len:
            raw = bytes(state.payload_buf[: self.cfg.packet_len])
        elif len(state.payload_buf) > 0:
            raw = bytes(state.payload_buf) + (
                b"\x00" * (self.cfg.packet_len - len(state.payload_buf))
            )
        else:
            return None

        state.emitted = True
        self.sample_seq += 1
        return FlowSample(
            sample_id=self.sample_seq,
            flow_key=key,
            ts=ts,
            trigger=trigger,
            raw784=raw,
        )

    def feed(self, pkt: PacketEvent) -> List[FlowSample]:
        out: List[FlowSample] = []
        key: FlowKey = (
            pkt.src_ip,
            pkt.dst_ip,
            pkt.src_port,
            pkt.dst_port,
            pkt.protocol,
        )
        state = self.active.get(key)
        if state is None:
            state = FlowState(last_seen=pkt.ts)
            self.active[key] = state
        self._touch(key)

        state.last_seen = pkt.ts
        if not (state.emitted and self.cfg.emit_once_per_flow):
            state.payload_buf.extend(pkt.payload)
            state.payload_buf.extend(b"\x00" * self.cfg.pad_len)

        if len(state.payload_buf) >= self.cfg.packet_len:
            sample = self._emit_from_state(key, state, pkt.ts, "length")
            if sample is not None:
                out.append(sample)
            if self.cfg.emit_once_per_flow:
                self.active.pop(key, None)

        if key in self.active and (pkt.fin or pkt.rst):
            sample = self._emit_from_state(key, state, pkt.ts, "fin_rst")
            if sample is not None:
                out.append(sample)
            self.active.pop(key, None)

        self._evict_if_needed()
        return out

    def sweep_timeouts(self, now_ts: Optional[float] = None) -> List[FlowSample]:
        now_ts = now_ts if now_ts is not None else time.time()
        out: List[FlowSample] = []
        timeout_keys: List[FlowKey] = []

        for key, state in self.active.items():
            if now_ts - state.last_seen >= self.cfg.flow_timeout_s:
                timeout_keys.append(key)

        for key in timeout_keys:
            state = self.active.get(key)
            if state is None:
                continue
            sample = self._emit_from_state(key, state, now_ts, "timeout")
            if sample is not None:
                out.append(sample)
            self.active.pop(key, None)

        return out
