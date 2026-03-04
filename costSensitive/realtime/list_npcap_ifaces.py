import ctypes
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_uint,
    c_void_p,
    create_string_buffer,
)


class PcapIf(Structure):
    pass


PcapIf._fields_ = [
    ("next", POINTER(PcapIf)),
    ("name", c_char_p),
    ("description", c_char_p),
    ("addresses", c_void_p),
    ("flags", c_uint),
]


def main():
    lib = ctypes.CDLL(r"C:\Windows\System32\Npcap\wpcap.dll")
    alldevs = POINTER(PcapIf)()
    errbuf = create_string_buffer(256)

    rc = lib.pcap_findalldevs(byref(alldevs), errbuf)
    if rc != 0:
        print("pcap_findalldevs failed:", errbuf.value.decode(errors="ignore"))
        return

    idx = 1
    ptr = alldevs
    while bool(ptr):
        name = ptr.contents.name.decode(errors="ignore") if ptr.contents.name else ""
        desc = (
            ptr.contents.description.decode(errors="ignore")
            if ptr.contents.description
            else ""
        )
        print(f"{idx}. {name} | {desc}")
        ptr = ptr.contents.next
        idx += 1

    lib.pcap_freealldevs(alldevs)


if __name__ == "__main__":
    main()
