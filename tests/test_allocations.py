"""Count Rust allocations including NumPy borrow checks, excluding CPython's allocator."""

import numpy as np
import pytest
from signals import echo_signal
from test_regression import CHANNELS, RATES

import aec3_py

pytestmark = pytest.mark.skipif(
    not hasattr(aec3_py, "_start_allocation_tracking"),
    reason="requires allocation-tracking build",
)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("rch,cch", CHANNELS)
def test_no_per_frame_dsp_allocations(rate, rch, cch):
    a = aec3_py.Aec3(rate, rch, cch, 30, backend="direct")
    r, c = echo_signal(rate, rch, cch)
    n = a.frame_samples
    a.process_frames_into(
        c[: 700 * n].reshape(700, -1),
        np.empty((700, n * cch), np.float32),
        r[: 700 * n].reshape(700, -1),
    )
    for frames in (1, 10, 100):
        cf = c[700 * n : (700 + frames) * n].reshape(frames, -1)
        rf = r[700 * n : (700 + frames) * n].reshape(frames, -1)
        output = np.empty_like(cf)
        aec3_py._start_allocation_tracking()
        a.process_frames_into(cf, output, rf)
        allocations, requested_bytes = aec3_py._stop_allocation_tracking()
        # Each safe NumPy borrow can create one 180-byte borrow-map entry.
        # The bound is independent of batch length: DSP adds no allocations.
        assert allocations <= 3, (frames, allocations, requested_bytes)
        assert requested_bytes <= 540, (frames, allocations, requested_bytes)
