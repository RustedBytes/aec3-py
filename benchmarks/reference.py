"""Run with the pre-optimization and optimized wheel in separate processes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from signals import echo_signal, run_stream

from aec3_py import Aec3

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
results = []
for rate in (16000, 32000, 48000):
    for rch, cch in ((1, 1), (1, 2), (2, 1), (2, 2)):
        for hp in (False, True):
            r, c = echo_signal(rate, rch, cch)
            a = Aec3(rate, rch, cch, 30, enable_high_pass=hp)
            output, metrics = run_stream(a, r, c, change_at=400)
            results.append(
                dict(
                    rate=rate,
                    render_channels=rch,
                    capture_channels=cch,
                    hp=hp,
                    sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                    metrics={
                        field: getattr(metrics, field)
                        for field in (
                            "echo_return_loss",
                            "echo_return_loss_enhancement",
                            "delay_ms",
                            "render_jitter_min",
                            "render_jitter_max",
                            "capture_jitter_min",
                            "capture_jitter_max",
                        )
                    },
                )
            )
args.output.write_text(json.dumps(results, indent=2))
