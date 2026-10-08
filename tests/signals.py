"""Deterministic normalized float32, delayed multi-tap echo fixtures."""
import numpy as np


def echo_signal(rate, render_channels, capture_channels, seconds=8, seed=2026):
    rng = np.random.default_rng(seed)
    count = int(rate * seconds)
    render = rng.normal(0, 0.12, (count, render_channels))
    # Band-limit without scipy; independent channels expose interleaving mistakes.
    for ch in range(render_channels):
        render[:, ch] = np.convolve(render[:, ch], np.ones(5) / 5, mode="same")
    render = render.astype(np.float32)
    source = render.mean(axis=1)
    echo = np.zeros(count, dtype=np.float32)
    for ms, gain in ((30, 0.6), (45, 0.25), (70, 0.1)):
        delay = rate * ms // 1000
        echo[delay:] += gain * source[:-delay]
    capture = np.stack([echo * (1 - 0.15 * ch) for ch in range(capture_channels)], axis=1)
    return render, capture.astype(np.float32)


def run_stream(aec, render, capture, split=False, change_at=None):
    frame = aec.frame_samples
    output = np.empty_like(capture)
    for i, start in enumerate(range(0, len(capture), frame)):
        r = render[start:start + frame].reshape(-1)
        c = capture[start:start + frame].reshape(-1)
        if split:
            aec.handle_render_frame(r)
            result, metrics = aec.process_capture_frame(c, level_change=i == change_at)
        else:
            result, metrics = aec.process(c, r, level_change=i == change_at)
        output[start:start + frame] = result.reshape(-1, capture.shape[1])
    return output, metrics


def attenuation_db(capture, output, rate):
    tail = slice(5 * rate, None)
    return float(10 * np.log10(np.mean(capture[tail].astype(np.float64) ** 2) /
                               max(np.mean(output[tail].astype(np.float64) ** 2), 1e-30)))
