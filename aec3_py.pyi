"""Typed API of the native extension; arrays contain normalized float32 audio."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class Metrics:
    @property
    def echo_return_loss(self) -> float: ...
    @property
    def echo_return_loss_enhancement(self) -> float: ...
    @property
    def delay_ms(self) -> int: ...
    @property
    def render_jitter_min(self) -> int: ...
    @property
    def render_jitter_max(self) -> int: ...
    @property
    def capture_jitter_min(self) -> int: ...
    @property
    def capture_jitter_max(self) -> int: ...

class Aec3:
    def __init__(
        self,
        sample_rate_hz: int,
        render_channels: int,
        capture_channels: int,
        initial_delay_ms: int | None = None,
        enable_high_pass: bool | None = None,
        *,
        enable_noise_suppression: bool = False,
        enable_gain_controller2: bool = False,
        enable_post_filter: bool = False,
        backend: str = "auto",
    ) -> None: ...
    @property
    def backend(self) -> Literal["direct", "graph"]: ...
    @property
    def frame_samples(self) -> int: ...
    @property
    def sample_rate_hz(self) -> int: ...
    def set_audio_buffer_delay(self, delay_ms: int) -> None: ...
    def metrics(self) -> Metrics: ...
    def handle_render_frame(self, render_frame: NDArray[np.float32]) -> None: ...
    def process_capture_frame(
        self,
        capture_frame: NDArray[np.float32],
        level_change: bool = False,
    ) -> tuple[NDArray[np.float32], Metrics]: ...
    def process(
        self,
        capture_frame: NDArray[np.float32],
        render_frame: NDArray[np.float32] | None = None,
        level_change: bool = False,
    ) -> tuple[NDArray[np.float32], Metrics]: ...
    def process_into(
        self,
        capture_frame: NDArray[np.float32],
        output_frame: NDArray[np.float32],
        render_frame: NDArray[np.float32] | None = None,
        level_change: bool = False,
    ) -> None: ...
    def process_frames_into(
        self,
        capture_frames: NDArray[np.float32],
        output_frames: NDArray[np.float32],
        render_frames: NDArray[np.float32] | None = None,
        level_change: bool = False,
    ) -> None: ...

# These private hooks exist only in allocation-tracking builds.
def _start_allocation_tracking() -> None: ...
def _stop_allocation_tracking() -> tuple[int, int]: ...
