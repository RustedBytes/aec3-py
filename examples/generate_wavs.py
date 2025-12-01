"""
Generate synthetic render + mic WAVs for examples/demo2.py.

pip install soundfile numpy
python examples/generate_wavs.py [render.wav] [mic_with_echo.wav]
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 48_000


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    return audio.astype(np.float32, copy=False)


def synthesize(sample_rate: int, duration_sec: float, seed: int):
    rng = np.random.default_rng(seed)
    num_samples = int(duration_sec * sample_rate)
    t = np.arange(num_samples, dtype=np.float32) / float(sample_rate)

    # Stereo render tone with slight differences per channel.
    left = 0.55 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 660.0 * t)
    right = 0.5 * np.sin(2 * np.pi * 392.0 * t + 0.2) + 0.18 * np.sin(
        2 * np.pi * 588.0 * t + 0.1
    )
    render = np.stack((left, right), axis=1)

    # Room-like impulse response to create an echo tail on the mic signal.
    ir = np.zeros(int(sample_rate * 0.18), dtype=np.float32)
    ir[0] = 0.65
    ir[int(sample_rate * 0.045)] = 0.32
    ir[int(sample_rate * 0.09)] = 0.18
    ir[int(sample_rate * 0.14)] = 0.1

    render_mono = render.mean(axis=1)
    echo = np.convolve(render_mono, ir)[:num_samples]

    near = 0.12 * np.sin(2 * np.pi * 205.0 * t) + 0.06 * np.sin(
        2 * np.pi * 255.0 * t + 0.5
    )
    noise = rng.normal(scale=0.01, size=num_samples)

    mic = echo + near + noise
    return render, mic[:, None]


def parse_args():
    script_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Generate render.wav and mic_with_echo.wav inputs for demo2.py."
    )
    parser.add_argument(
        "render_path",
        nargs="?",
        default=script_dir / "render.wav",
        type=Path,
        help="Path to write the far-end render WAV.",
    )
    parser.add_argument(
        "mic_path",
        nargs="?",
        default=script_dir / "mic_with_echo.wav",
        type=Path,
        help="Path to write the mic recording containing echo.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Seconds of audio to synthesize.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        choices=(16000, 32000, 48000),
        help="Sample rate to use (must match AEC3-supported rates).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic noise.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    render, mic = synthesize(args.sample_rate, args.duration, args.seed)
    render = normalize(render)
    mic = normalize(mic)

    args.render_path.parent.mkdir(parents=True, exist_ok=True)
    args.mic_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(args.render_path), render, args.sample_rate)
    sf.write(str(args.mic_path), mic, args.sample_rate)

    print(f"Saved render to {args.render_path.resolve()}")
    print(f"Saved mic_with_echo to {args.mic_path.resolve()}")
    print(
        f"Sample rate: {args.sample_rate} Hz | Duration: {args.duration:.2f} s "
        f"| Seed: {args.seed}"
    )


if __name__ == "__main__":
    main()
