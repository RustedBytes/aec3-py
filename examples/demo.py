import numpy as np
import aec3_py

aec = aec3_py.Aec3(
    sample_rate_hz=48_000,
    render_channels=2,
    capture_channels=1,
    initial_delay_ms=116,
    enable_high_pass=True,
)

frame_samples = aec.frame_samples  # per channel, 10 ms

# 10 ms frames, interleaved
capture = np.zeros(frame_samples * 1, dtype=np.float32)  # mono
render = np.zeros(frame_samples * 2, dtype=np.float32)  # stereo

# Option 1: split calls
aec.handle_render_frame(render)
out, metrics = aec.process_capture_frame(capture, level_change=False)

# Option 2: combined convenience call
out2, metrics2 = aec.process(capture, render, level_change=False)

print("out.shape:", out.shape)
print("ERL:", metrics.echo_return_loss)
print("ERLE:", metrics.echo_return_loss_enhancement)
print("delay_ms:", metrics.delay_ms)
