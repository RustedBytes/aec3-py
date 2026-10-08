# AEC3 0.4 migration review

Reviewed PR #1 at f8651cedef9d4880bef91873c1435890304c7497 against main b6d9a6e5489d818ea090cb87b2dae86bf431aa8e.

## Confirmed issues and fixes

- `level_change` was silently ignored. Capture `PacketMeta.discontinuity` is passed by the upstream AEC3 node to `EchoCanceller3::process_capture` as its gain-change flag. Forward it with a monotonically increasing capture sequence; a regression checks that an explicit gain change changes post-event output and that split and combined methods agree.
- NS, AGC2 and post filtering were enabled implicitly. Keep previous HPF + AEC stage selection by default and expose additional stages as keyword-only opt-ins.
- The upstream capture method returns a readiness boolean. Treat missing output as an error rather than presenting silence as a valid processed frame.
- Non-finite samples could poison persistent DSP state. Reject them, along with malformed arrays, before feeding either stream. Combined-call argument validation is atomic.
- Add NumPy as a runtime dependency and regenerate the stale uv lock (previously Python >=3.8 and version 0.1.0).
- Correct the claimed Rust minimum: aec3 0.4 uses let chains, which require Rust 1.88.

## Buffer and API review

NumPy borrows stay alive for the call, are read-only and contiguous; no borrowed slice escapes. A fresh owned output Vec transfers to NumPy without an additional wrapper-side copy. The graph itself copies input into pooled chunks and allocates packet/state bookkeeping; this is not a zero-allocation pipeline. No thread-safety claim is made: the original unsendable class is retained. Frame lengths are validated before graph submission. Metrics sink uses a latest-value queue of capacity one; each call drains it and caches the most recent snapshot.

The public positional constructor and process signatures remain compatible, as do output tuple/dtype/shape and the original metric fields. Added jitter fields are read-only. Python 3.8/3.9 support is intentionally dropped. Graph errors, including delay update errors, map to ValueError. A failed graph operation is not guaranteed to be recoverable; recreate the instance for internal graph failures. Invalid input validation occurs before DSP mutation.

## Reproducible synthetic comparison

Linux x86_64, CPython 3.12.14, NumPy 2.5.3, Rust 1.99, identical release profile (opt-level=z, LTO), same host, isolated processes run sequentially. Three independent instances per configuration, median time per 10 ms frame across 800 frames. Timing includes Python dispatch, output allocation and metrics. It excludes signal generation and instance construction. No latency percentiles or allocation counts are claimed.

Seed 2026, 8 seconds of independent per-channel filtered Gaussian reference (stddev 0.12, five-tap moving average). Capture is mean render with taps at 30/45/70 ms and gains 0.6/0.25/0.1; capture channel two has gain 0.85. External delay hint 30 ms. Energy attenuation is measured over seconds 5–8, separately from the library-reported ERLE. Both versions use the same normalized float32 inputs and HPF setting.

| Hz | Render→capture | Old µs/frame | New µs/frame | Time change | Old attenuation dB | New attenuation dB | Full stages µs/frame | Full stages attenuation dB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 16000 | 1→1 | 255.6 | 103.3 | -59.6% | 18.6 | 62.4 | 151.0 | 53.1 |
| 16000 | 1→2 | 303.3 | 133.8 | -55.9% | 17.9 | 62.3 | 194.8 | 53.3 |
| 16000 | 2→1 | 291.4 | 106.7 | -63.4% | 15.9 | 34.9 | 160.2 | 25.5 |
| 16000 | 2→2 | 355.5 | 148.1 | -58.3% | 15.3 | 34.9 | 218.9 | 25.8 |
| 32000 | 1→1 | 274.0 | 126.1 | -54.0% | 18.0 | 51.8 | 195.9 | 42.0 |
| 32000 | 1→2 | 341.0 | 179.4 | -47.4% | 17.4 | 51.8 | 275.3 | 42.2 |
| 32000 | 2→1 | 318.5 | 138.1 | -56.6% | 15.3 | 44.8 | 207.8 | 35.1 |
| 32000 | 2→2 | 423.7 | 195.8 | -53.8% | 14.8 | 44.8 | 293.7 | 35.6 |
| 48000 | 1→1 | 354.5 | 227.9 | -35.7% | 17.6 | 47.0 | 355.1 | 41.1 |
| 48000 | 1→2 | 467.1 | 363.2 | -22.2% | 16.9 | 47.0 | 571.7 | 41.4 |
| 48000 | 2→1 | 438.0 | 255.8 | -41.6% | 14.7 | 44.1 | 385.4 | 36.7 |
| 48000 | 2→2 | 556.3 | 393.1 | -29.3% | 14.0 | 44.1 | 588.7 | 37.1 |

These are synthetic results, not speech quality scores or real-room guarantees. AGC can amplify residual echo, so output attenuation with full stages is not a direct measure of AEC improvement. The suite separately checks near-end output survives with no reference, split/combined equivalence, ownership, invalid inputs, streaming, metrics and full-stage processing. Real speech double-talk, microphone nonlinearities, drift, clipping, long-tail reverberation, and non-x86 performance remain outside this synthetic validation. CI tests built wheels on Linux x86_64, Windows x64 and macOS ARM64, Python 3.10–3.14; its uploaded benchmark JSON is platform-specific and not interchangeable with local timing.
