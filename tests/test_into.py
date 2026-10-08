import numpy as np
import pytest
from signals import echo_signal, run_stream
from test_regression import CHANNELS, FIELDS, RATES

from aec3_py import Aec3


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("rch,cch", CHANNELS)
@pytest.mark.parametrize("hp", (True, False))
def test_direct_graph_and_into_equivalence(rate, rch, cch, hp):
    r, c = echo_signal(rate, rch, cch)
    a, b, d = [
        Aec3(rate, rch, cch, 30, enable_high_pass=hp, backend=backend)
        for backend in ("direct", "graph", "direct")
    ]
    a_out, am = run_stream(a, r, c, change_at=400)
    b_out, bm = run_stream(b, r, c, change_at=400)
    np.testing.assert_array_equal(a_out, b_out)
    for field in FIELDS:
        assert getattr(am, field) == getattr(bm, field)
    output = np.empty(d.frame_samples * cch, np.float32)
    for i in range(800):
        start = i * d.frame_samples
        assert (
            d.process_into(
                c[start : start + d.frame_samples].reshape(-1),
                output,
                r[start : start + d.frame_samples].reshape(-1),
                i == 400,
            )
            is None
        )
        np.testing.assert_array_equal(
            output, a_out[start : start + d.frame_samples].reshape(-1)
        )


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("rch,cch", CHANNELS)
def test_batch_matches_frames(rate, rch, cch):
    r, c = echo_signal(rate, rch, cch, seconds=1)
    a, b = Aec3(rate, rch, cch, 30), Aec3(rate, rch, cch, 30)
    expected, _ = run_stream(a, r, c, change_at=0)
    cf, rf = c.reshape(100, -1), r.reshape(100, -1)
    output = np.empty_like(cf)
    assert b.process_frames_into(cf, output, rf, level_change=True) is None
    np.testing.assert_array_equal(output.reshape(expected.shape), expected)


@pytest.mark.parametrize("batch", (False, True))
@pytest.mark.parametrize(
    "bad", ("readonly", "alias", "render_alias", "shape", "stride", "dtype", "nan")
)
def test_into_rejects_invalid_buffers(batch, bad):
    a, b = Aec3(16000, 1, 1), Aec3(16000, 1, 1)
    shape = (2, 160) if batch else (160,)
    c, r, o = [np.zeros(shape, np.float32) for _ in range(3)]
    if bad == "readonly":
        o.flags.writeable = False
    if bad == "alias":
        o = c.view()
    if bad == "render_alias":
        o = r.view()
    if bad == "shape":
        o = np.zeros((2, 159) if batch else 159, np.float32)
    if bad == "stride":
        o = np.zeros((2, 320) if batch else 320, np.float32)[..., ::2]
    if bad == "dtype":
        o = o.astype(np.float64)
    if bad == "nan":
        c.flat[-1] = np.nan
    fn = a.process_frames_into if batch else a.process_into
    with pytest.raises((ValueError, TypeError)):
        # Deliberately pass float64 to verify runtime rejection of invalid dtype.
        fn(c, o, r)  # pyright: ignore[reportArgumentType]
    normal = np.zeros(160, np.float32)
    x, _ = a.process(normal, normal)
    y, _ = b.process(normal, normal)
    np.testing.assert_array_equal(x, y)


def test_backend_selection_and_graph_stages():
    assert Aec3(16000, 1, 2).backend == "direct"
    assert Aec3(16000, 1, 1, enable_gain_controller2=True).backend == "graph"
    assert Aec3(16000, 3, 1).backend == "graph"
    with pytest.raises(ValueError):
        Aec3(16000, 1, 1, backend="invalid")
    with pytest.raises(ValueError):
        Aec3(16000, 1, 1, backend="direct", enable_noise_suppression=True)
    a, b = [
        Aec3(16000, 1, 1, enable_noise_suppression=True, enable_gain_controller2=True)
        for _ in range(2)
    ]
    r, c = echo_signal(16000, 1, 1, seconds=1)
    expected, _ = run_stream(a, r, c)
    output = np.empty((100, 160), np.float32)
    b.process_frames_into(c.reshape(100, 160), output, r.reshape(100, 160))
    np.testing.assert_array_equal(output.reshape(-1, 1), expected)


def test_batch_empty_and_capture_only():
    a = Aec3(16000, 1, 1)
    assert (
        a.process_frames_into(
            np.empty((0, 160), np.float32), np.empty((0, 160), np.float32)
        )
        is None
    )
    c = np.full((20, 160), 0.1, np.float32)
    o = np.empty_like(c)
    a.process_frames_into(c, o)
    assert np.isfinite(o).all() and np.mean(o**2) > 0


@pytest.mark.parametrize("array", (0, 1, 2))
def test_batch_rejects_fortran_layout_before_mutation(array):
    a, b = Aec3(16000, 1, 1), Aec3(16000, 1, 1)
    args = [np.zeros((2, 160), np.float32) for _ in range(3)]
    args[array] = np.asfortranarray(args[array])
    with pytest.raises(ValueError, match="C-contiguous"):
        a.process_frames_into(args[0], args[1], args[2])
    normal = np.zeros(160, np.float32)
    np.testing.assert_array_equal(
        a.process(normal, normal)[0], b.process(normal, normal)[0]
    )
