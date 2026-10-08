import numpy as np
import pytest
from signals import attenuation_db, echo_signal, run_stream

from aec3_py import Aec3, Metrics

RATES = (16000, 32000, 48000)
CHANNELS = ((1, 1), (1, 2), (2, 1), (2, 2))
FIELDS = (
    "echo_return_loss",
    "echo_return_loss_enhancement",
    "delay_ms",
    "render_jitter_min",
    "render_jitter_max",
    "capture_jitter_min",
    "capture_jitter_max",
)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("render_ch,capture_ch", CHANNELS)
def test_stream_and_echo(rate, render_ch, capture_ch):
    aec = Aec3(rate, render_ch, capture_ch, 30)
    assert aec.sample_rate_hz == rate and aec.frame_samples == rate // 100
    render, capture = echo_signal(rate, render_ch, capture_ch)
    saved_render, saved_capture = render.copy(), capture.copy()
    output, metrics = run_stream(aec, render, capture)
    assert output.dtype == np.float32 and output.shape == capture.shape
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(render, saved_render)
    np.testing.assert_array_equal(capture, saved_capture)
    # Echo-only test; separate near-end test prevents silence from passing.
    assert attenuation_db(capture, output, rate) > 10
    assert isinstance(metrics, Metrics)
    assert metrics.echo_return_loss_enhancement > 3
    assert 0 <= metrics.delay_ms <= 100
    assert metrics.render_jitter_min == metrics.render_jitter_max == 1
    assert metrics.capture_jitter_min == metrics.capture_jitter_max == 1
    for field in FIELDS:
        assert np.isfinite(getattr(metrics, field))
        assert getattr(aec.metrics(), field) == getattr(metrics, field)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("render_ch,capture_ch", CHANNELS)
def test_split_matches_combined(rate, render_ch, capture_ch):
    render, capture = echo_signal(rate, render_ch, capture_ch, seconds=1)
    a = Aec3(rate, render_ch, capture_ch, 30)
    b = Aec3(rate, render_ch, capture_ch, 30)
    combined, _ = run_stream(a, render, capture, change_at=50)
    split, _ = run_stream(b, render, capture, split=True, change_at=50)
    np.testing.assert_array_equal(combined, split)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("channels", (1, 2))
def test_near_end_and_owned_outputs(rate, channels):
    aec = Aec3(rate, channels, channels, enable_high_pass=False)
    rng = np.random.default_rng(34)
    c = rng.normal(0, 0.1, aec.frame_samples * channels).astype(np.float32)
    first, _ = aec.process(c)
    saved = first.copy()
    latest = first
    for _ in range(100):
        latest, _ = aec.process(c)
    np.testing.assert_array_equal(first, saved)
    assert np.mean(latest**2) > 1e-5
    assert not np.shares_memory(first, latest) and not np.shares_memory(first, c)
    assert aec.set_audio_buffer_delay(30) is None
    assert aec.set_audio_buffer_delay(0) is None


@pytest.mark.parametrize("rate", (0, -1, 8000, 44100, 96000))
def test_invalid_rate(rate):
    with pytest.raises(ValueError):
        Aec3(rate, 1, 1)


@pytest.mark.parametrize("r,c", ((0, 1), (1, 0), (65536, 1), (1, 65536)))
def test_invalid_channels(r, c):
    with pytest.raises(ValueError):
        Aec3(16000, r, c)


@pytest.mark.parametrize(
    "method", ("process", "process_capture_frame", "handle_render_frame")
)
@pytest.mark.parametrize(
    "kind", ("short", "long", "strided", "float64", "2d", "nan", "inf")
)
def test_invalid_frames(method, kind):
    aec = Aec3(16000, 1, 1)
    frames = {
        "short": np.zeros(159, np.float32),
        "long": np.zeros(161, np.float32),
        "strided": np.zeros(320, np.float32)[::2],
        "float64": np.zeros(160),
        "2d": np.zeros((160, 1), np.float32),
        "nan": np.full(160, np.nan, np.float32),
        "inf": np.full(160, np.inf, np.float32),
    }
    with pytest.raises((ValueError, TypeError)):
        getattr(aec, method)(frames[kind])
    # Invalid inputs must not poison state.
    output, _ = aec.process(np.zeros(160, np.float32), np.zeros(160, np.float32))
    assert np.isfinite(output).all()


def test_combined_validation_is_atomic():
    a, b = Aec3(16000, 1, 1), Aec3(16000, 1, 1)
    r, c = echo_signal(16000, 1, 1, seconds=1)
    with pytest.raises(ValueError):
        a.process(np.zeros(159, np.float32), r[:160].reshape(-1))
    with pytest.raises(ValueError):
        a.process(c[:160].reshape(-1), np.full(160, np.nan, np.float32))
    x, _ = run_stream(a, r, c)
    y, _ = run_stream(b, r, c)
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("channels", (1, 2))
def test_optional_full_pipeline(rate, channels):
    aec = Aec3(
        rate,
        channels,
        channels,
        enable_noise_suppression=True,
        enable_gain_controller2=True,
        enable_post_filter=True,
    )
    r, c = echo_signal(rate, channels, channels, seconds=1)
    out, _ = run_stream(aec, r, c)
    assert np.isfinite(out).all()


def test_initial_metrics_and_readonly_fields():
    m = Aec3(16000, 1, 1).metrics()
    for field in FIELDS:
        assert np.isfinite(getattr(m, field))
        with pytest.raises(AttributeError):
            setattr(m, field, 0)


def test_level_change_reaches_aec3():
    r, c = echo_signal(16000, 1, 1)
    c[400 * 160 :] *= 0.5
    a, b = Aec3(16000, 1, 1, 30), Aec3(16000, 1, 1, 30)
    changed, _ = run_stream(a, r, c, change_at=400)
    unchanged, _ = run_stream(b, r, c)
    np.testing.assert_array_equal(changed[: 400 * 160], unchanged[: 400 * 160])
    assert np.max(np.abs(changed[400 * 160 :] - unchanged[400 * 160 :])) > 1e-7
