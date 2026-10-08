"""Same-machine wheel comparison; signal slicing and instance construction are untimed."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from signals import attenuation_db, echo_signal

from aec3_py import Aec3

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
results = []
modes = ["process"] + (["into", "batch"] if hasattr(Aec3, "process_into") else [])
for rate in (16000, 32000, 48000):
    for rch, cch in ((1, 1), (1, 2), (2, 1), (2, 2)):
        r, c = echo_signal(rate, rch, cch)
        n = rate // 100
        rf, cf = r.reshape(800, -1), c.reshape(800, -1)
        frames = list(zip(cf, rf, strict=True))
        for mode in modes:
            timings = []
            output = np.empty_like(cf)
            output_rows = list(output)
            batches = [
                (cf[i : i + 100], output[i : i + 100], rf[i : i + 100])
                for i in range(0, 800, 100)
            ]
            for _ in range(5):
                a = Aec3(rate, rch, cch, 30)
                start = time.perf_counter_ns()
                if mode == "process":
                    for i, (cap, ren) in enumerate(frames):
                        output[i] = a.process(cap, ren)[0]
                elif mode == "into":
                    for (cap, ren), out in zip(frames, output_rows, strict=True):
                        a.process_into(cap, out, ren)
                else:
                    for cap, out, ren in batches:
                        a.process_frames_into(cap, out, ren)
                timings.append((time.perf_counter_ns() - start) / 800 / 1000)
            results.append(
                dict(
                    rate=rate,
                    render_channels=rch,
                    capture_channels=cch,
                    mode=mode,
                    us_per_frame=statistics.median(timings),
                    trials=timings,
                    attenuation_db=attenuation_db(c, output.reshape(c.shape), rate),
                )
            )
args.output.write_text(
    json.dumps(
        dict(
            python=sys.version,
            numpy=np.__version__,
            platform=platform.platform(),
            frames_per_batch=100,
            results=results,
        ),
        indent=2,
    )
)
print(f"{len(results)} configurations measured: {args.output}")
