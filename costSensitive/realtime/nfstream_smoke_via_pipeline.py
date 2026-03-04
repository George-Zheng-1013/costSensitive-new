import time
from realtime.single_pipeline import NFStreamer, _NFSTREAM_IMPORT_ERROR

print("nfstream_import_error=", _NFSTREAM_IMPORT_ERROR)
if NFStreamer is None:
    raise RuntimeError("NFStreamer unavailable")

source = r"\Device\NPF_{94B4B764-8E09-485A-9EF1-10C26641DF79}"
print("start source=", source)
streamer = NFStreamer(
    source=source,
    statistical_analysis=False,
    decode_tunnels=True,
    promiscuous_mode=True,
    idle_timeout=1,
    active_timeout=1,
)

count = 0
start = time.time()
for flow in streamer:
    count += 1
    if count <= 5:
        print(
            "flow",
            count,
            getattr(flow, "src_ip", None),
            getattr(flow, "dst_ip", None),
            getattr(flow, "protocol", None),
        )
    if time.time() - start >= 20:
        break

print("done flow_count=", count)
