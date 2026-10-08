# aec3-py

PyO3 + NumPy wrapper for [aec3-rs](https://github.com/RubyBit/aec3-rs), a Rust implementation of Google’s AEC3 (Acoustic Echo Cancellation) pipeline.

## Requirements
- Python 3.10–3.14 with NumPy
- Rust toolchain 1.88+ (edition 2024) to build the extension
- `maturin` for building wheels (`pip install maturin`)
- `soundfile` only for the WAV example

## Install from source
```bash
pip install maturin
pip install .
# or for editable dev installs:
maturin develop --release
```

## Quick start
Frames are **interleaved float32 1D arrays** with length `frame_samples * channels`, where `frame_samples` is the per-channel size of a 10 ms frame (e.g., 160 samples at 16 kHz).

```python
import numpy as np
import aec3_py

aec = aec3_py.Aec3(
    sample_rate_hz=48_000,       # one of {16000, 32000, 48000}
    render_channels=2,
    capture_channels=1,
    initial_delay_ms=120,        # setup-dependent; tune for your system
    enable_high_pass=True,
)

frame = aec.frame_samples
render = np.zeros(frame * 2, dtype=np.float32)   # stereo far-end
capture = np.zeros(frame * 1, dtype=np.float32)  # mono mic

out, metrics = aec.process(capture, render)
print("clean shape:", out.shape)
print("ERL dB:", metrics.echo_return_loss)
print("ERLE dB:", metrics.echo_return_loss_enhancement)
print("delay ms:", metrics.delay_ms)
```

## Process WAV files end-to-end
`examples/demo2.py` loads stereo render + mic signals, runs the AEC, and writes a cleaned WAV. Generate sample inputs and run the demo with:
```bash
pip install soundfile numpy
python examples/generate_wavs.py       # writes examples/render.wav and examples/mic_with_echo.wav
python examples/demo2.py examples/render.wav examples/mic_with_echo.wav output.wav
```

## API highlights
- `Aec3.frame_samples` — samples **per channel** in a 10 ms frame.
- `Aec3.handle_render_frame(render_frame)` — feed far-end audio (interleaved).
- `Aec3.process_capture_frame(capture_frame)` — process mic frame, returns `(out_frame, Metrics)`.
- `Aec3.process(capture_frame, render_frame=None)` — combined call; `render_frame` optional.
- `Aec3.set_audio_buffer_delay(delay_ms)` — update the render-to-capture delay estimate at runtime (raises `ValueError` if the graph rejects the update).
- `Aec3.metrics()` — read current metrics without processing.
- `Metrics` fields: `echo_return_loss`, `echo_return_loss_enhancement`, `delay_ms`, plus jitter stats `render_jitter_min` / `render_jitter_max` / `capture_jitter_min` / `capture_jitter_max`.

> `level_change=True` signals a capture gain change. It is forwarded as capture packet `discontinuity`, which the upstream AEC3 node passes to its gain-change handling.

## Notes
- Built on `aec3::pipelines::linear` (aec3-rs 0.4). The default is high-pass filter → AEC3, retaining the previous stage selection. New keyword-only `enable_noise_suppression`, `enable_gain_controller2`, and `enable_post_filter` options default to `False`. Enable them explicitly for HPF → AEC3 → NS → AGC2 → fullband post filter (post filter acts at 48 kHz).
- Inputs and outputs use normalized float32 audio (typically -1 to +1), not float-encoded PCM16. NaN/Inf are rejected before processing; inputs are borrowed read-only and output arrays own their memory.
- The same `Aec3` instance must stay on its creating thread. Process render before the corresponding capture frame. `initial_delay_ms` / `set_audio_buffer_delay()` supply an external delay hint, not an output delay or audio resampler.
- `set_audio_buffer_delay()` returns `None` on success and raises `ValueError` for graph errors. Metrics are cached snapshots of the latest AEC3 export; reading them repeatedly does not advance audio or reset adaptation. They describe the AEC stage before optional NS/AGC2.
- A graph call that unexpectedly produces no capture output raises `ValueError` rather than returning fabricated silence. Upstream algorithm changes mean default output is not bit-identical to 0.1.x.
- Supported sample rates: 16 kHz, 32 kHz, 48 kHz. Resample beforehand if needed.
- Input arrays must be contiguous `float32`. Shape validation mirrors the Rust API.
- `frame_samples` x `channels` always describes a 10 ms chunk; stream audio in those frame sizes for best performance.

## Validation

CI builds and installs release wheels before running regression tests on Linux, Windows and macOS with Python 3.10–3.14. Locally:

```sh
python -m pip install maturin numpy pytest
maturin develop --release --locked
python -m pytest -q
python benchmarks/compare.py --output benchmark.json
```

Run the benchmark with old and new wheels in separate environments on the same machine, sequentially. `--full-pipeline` explicitly enables the additional stages. See [migration review](docs/migration-review.md) for methodology, results and limitations.
