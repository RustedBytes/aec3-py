"""
pip install soundfile numpy
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import aec3_py  # this is the Rust + PyO3 extension module


def load_audio(path: Path):
    """
    Load WAV file as float32, always 2D array: shape (num_samples, channels).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def main(render_path: str, capture_path: str, out_path: str):
    render_path = Path(render_path)
    capture_path = Path(capture_path)
    out_path = Path(out_path)

    # 1. Load files
    render, sr_r = load_audio(render_path)
    capture, sr_c = load_audio(capture_path)

    if sr_r != sr_c:
        raise RuntimeError(
            f"Sample rates differ: render={sr_r} Hz, capture={sr_c} Hz. "
            "Resample to a common rate (e.g. 48000) before running AEC."
        )

    sample_rate = sr_r

    # AEC3 only supports specific sample rates (typically 16000, 32000, 48000).
    if sample_rate not in (16000, 32000, 48000):
        raise RuntimeError(
            f"Unsupported sample rate {sample_rate} Hz. "
            "Please resample both files to 16000, 32000, or 48000 Hz."
        )

    # Make sure both arrays are C-contiguous float32
    render = np.ascontiguousarray(render, dtype=np.float32)
    capture = np.ascontiguousarray(capture, dtype=np.float32)

    num_samples = min(render.shape[0], capture.shape[0])
    render = render[:num_samples]
    capture = capture[:num_samples]

    render_channels = render.shape[1]
    capture_channels = capture.shape[1]

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Render:  {num_samples} samples, {render_channels} ch")
    print(f"Capture: {num_samples} samples, {capture_channels} ch")

    # 2. Create AEC instance
    # initial_delay_ms is very setup-dependent; 100–150 ms is a reasonable guess to start.
    aec = aec3_py.Aec3(
        sample_rate_hz=sample_rate,
        render_channels=render_channels,
        capture_channels=capture_channels,
        initial_delay_ms=120,
        enable_high_pass=True,
    )

    frame_samples = aec.frame_samples  # per channel, 10 ms
    print(f"frame_samples: {frame_samples} samples per channel per 10 ms")

    total_full_frames = num_samples // frame_samples
    trimmed_samples = total_full_frames * frame_samples

    if trimmed_samples < num_samples:
        print(
            f"Truncating to {trimmed_samples} samples "
            f"({total_full_frames} full frames of {frame_samples})."
        )

    render = render[:trimmed_samples]
    capture = capture[:trimmed_samples]

    # Output buffer: same shape as capture
    out = np.zeros_like(capture, dtype=np.float32)

    # 3. Process frame-by-frame
    for frame_idx in range(total_full_frames):
        start = frame_idx * frame_samples
        end = start + frame_samples

        # Shape (frame_samples, channels)
        render_frame_2d = render[start:end, :]
        capture_frame_2d = capture[start:end, :]

        # Interleave channels into 1D for the Rust AEC:
        # [s0c0, s0c1, ..., s1c0, s1c1, ...]
        render_frame = render_frame_2d.reshape(-1)
        capture_frame = capture_frame_2d.reshape(-1)

        # Either:
        #   aec.handle_render_frame(render_frame)
        #   out_frame, metrics = aec.process_capture_frame(capture_frame)
        # or use combined:
        out_frame, metrics = aec.process(
            capture_frame,
            render_frame,
            level_change=False,
        )

        # Back to (frame_samples, capture_channels)
        out_frame_2d = out_frame.reshape(frame_samples, capture_channels)
        out[start:end, :] = out_frame_2d

        # Optional: log some metrics every N frames
        if frame_idx % 100 == 0:
            print(
                f"frame {frame_idx}/{total_full_frames} "
                f"ERL={metrics.echo_return_loss:.1f} dB "
                f"ERLE={metrics.echo_return_loss_enhancement:.1f} dB "
                f"delay={metrics.delay_ms} ms"
            )

    # 4. Save result
    sf.write(str(out_path), out, sample_rate)
    print(f"Saved echo-cancelled audio to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "  python run_aec_from_files.py render.wav mic_with_echo.wav output.wav"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
