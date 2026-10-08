"""Run with each installed version in an isolated Python process; compare JSON."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from signals import attenuation_db, echo_signal, run_stream

from aec3_py import Aec3, Metrics


def run_trial(
    rate: int,
    render_channels: int,
    capture_channels: int,
    render: NDArray[np.float32],
    capture: NDArray[np.float32],
    full_pipeline: bool,
) -> tuple[NDArray[np.float32], Metrics, float]:
    aec = Aec3(
        rate,
        render_channels,
        capture_channels,
        30,
        enable_noise_suppression=full_pipeline,
        enable_gain_controller2=full_pipeline,
        enable_post_filter=full_pipeline,
    )
    start = time.perf_counter()
    output, metrics = run_stream(aec, render, capture)
    us_per_frame = (time.perf_counter() - start) * 1e6 / 800
    return output, metrics, us_per_frame


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--full-pipeline", action="store_true")
args = parser.parse_args()
results = []
for rate in (16000, 32000, 48000):
    for rch, cch in ((1, 1), (1, 2), (2, 1), (2, 2)):
        render, capture = echo_signal(rate, rch, cch)
        trials = [
            run_trial(rate, rch, cch, render, capture, args.full_pipeline)
            for _ in range(3)
        ]
        output, metrics, _ = trials[-1]
        timings = [trial[2] for trial in trials]
        results.append(
            dict(
                rate=rate,
                render_channels=rch,
                capture_channels=cch,
                attenuation_db=attenuation_db(capture, output, rate),
                us_per_frame=float(np.median(timings)),
                erle_db=metrics.echo_return_loss_enhancement,
            )
        )
args.output.write_text(
    json.dumps(
        dict(
            python=sys.version,
            platform=platform.platform(),
            full_pipeline=args.full_pipeline,
            results=results,
        ),
        indent=2,
    )
)
print(args.output.read_text())
