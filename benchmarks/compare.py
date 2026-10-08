"""Run with each installed version in an isolated Python process; compare JSON."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from signals import echo_signal, run_stream, attenuation_db
from aec3_py import Aec3

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--full-pipeline", action="store_true")
args = parser.parse_args()
results = []
for rate in (16000, 32000, 48000):
    for rch, cch in ((1, 1), (1, 2), (2, 1), (2, 2)):
        render, capture = echo_signal(rate, rch, cch)
        timings = []
        for _ in range(3):
            options = dict(enable_noise_suppression=True, enable_gain_controller2=True,
                           enable_post_filter=True) if args.full_pipeline else {}
            aec = Aec3(rate, rch, cch, 30, **options)
            start = time.perf_counter()
            output, metrics = run_stream(aec, render, capture)
            timings.append((time.perf_counter() - start) * 1e6 / 800)
        results.append(dict(rate=rate, render_channels=rch, capture_channels=cch,
                            attenuation_db=attenuation_db(capture, output, rate),
                            us_per_frame=float(np.median(timings)),
                            erle_db=metrics.echo_return_loss_enhancement))
args.output.write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
                                      full_pipeline=args.full_pipeline, results=results), indent=2))
print(args.output.read_text())
