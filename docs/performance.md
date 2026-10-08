# Reusable-buffer performance

Baseline: merged main `de8529fcdfe49486cfc3e38dc1f284272c939a20` (aec3 0.4 graph, opt-level=z). Optimized: direct mono/stereo HPF + AEC, four vendored upstream fixes, opt-level=3; unchanged Python and NumPy versions, compiler and hardware. CPython 3.12.14, NumPy 2.5.3, Linux x86_64. Production wheels, allocation tracking disabled during timing.

`benchmarks/buffers.py`: five independent 800-frame runs per configuration, median µs per 10 ms audio frame. Signal generation, NumPy frame views, output allocation and instance construction occur outside timing. Legacy `process` timing includes a newly returned output array and metrics per frame; `into` reuses output, `batch` processes 100 frames per call. Results are observed timings on one shared host, not universal latency guarantees. Batch numbers are throughput, not realtime batch latency. Signal/echo fixture is described in migration-review.md. Generate local raw trials with `python benchmarks/buffers.py --output /tmp/performance.json`; measurement JSON files are not committed.

| Hz | Render→capture | Baseline process µs | New process µs | New into µs | New batch µs/frame | Batch time change |
|---:|:---:|---:|---:|---:|---:|---:|
| 16000 | 1→1 | 105.0 | 58.3 | 57.0 | 56.2 | -46.5% |
| 16000 | 1→2 | 138.2 | 70.8 | 69.4 | 68.4 | -50.5% |
| 16000 | 2→1 | 117.9 | 60.5 | 58.3 | 58.7 | -50.2% |
| 16000 | 2→2 | 158.8 | 75.0 | 74.3 | 73.6 | -53.7% |
| 32000 | 1→1 | 138.9 | 76.6 | 75.5 | 75.4 | -45.7% |
| 32000 | 1→2 | 192.6 | 103.5 | 102.4 | 102.2 | -46.9% |
| 32000 | 2→1 | 150.2 | 82.7 | 81.7 | 80.8 | -46.2% |
| 32000 | 2→2 | 201.0 | 114.5 | 110.8 | 109.6 | -45.5% |
| 48000 | 1→1 | 240.3 | 120.1 | 118.5 | 117.4 | -51.1% |
| 48000 | 1→2 | 379.3 | 180.9 | 180.1 | 179.5 | -52.7% |
| 48000 | 2→1 | 272.4 | 134.3 | 133.2 | 131.2 | -51.8% |
| 48000 | 2→2 | 418.0 | 199.4 | 197.0 | 195.4 | -53.2% |

## Allocation evidence and behavioral checks

An opt-in `allocation-tracking` global allocator counts Rust allocation/reallocation calls and requested bytes on the current thread, including upstream DSP and this extension's NumPy borrow bookkeeping. It excludes CPython/NumPy C allocations and does not report resident memory. Tests run in isolated Python environments so this extension owns the NumPy borrow-check capsule. It is absent from production builds.

Before upstream patches, an instrumented unmodified graph backend with reusable output took 72.5–134.5 Rust allocations/frame and requested approximately 33–93 KB/frame (16/32/48 kHz mono/stereo). The optimized direct `process_into` takes 3 allocations/540 requested bytes per call: all three are safe NumPy borrow-map bookkeeping. After 700 warmup frames, batch sizes 1, 10 and 100 all have the same <=3 / <=540 bound, proving no per-frame DSP allocations in these fixtures. Allocation tests cover all four mono/stereo render/capture combinations at all three rates.

`cargo run --release --locked --example allocation_probe` independently counts allocations with no Python involved: direct DSP has zero allocations for 24 rate/channel/HPF combinations after warmup, including gain-change flags, delay updates, full render-queue saturation and drain. It also checks a 700-second diagnostic-history simulation remains bounded at 64 samples without allocating. Stereo-content transitions that reconstruct the upstream DSP remain outside this guarantee.

`benchmarks/reference.py` compares full output SHA-256 hashes and every exposed metric against the original merged wheel: 24 eight-second streams (all rates/channel combinations, HPF on/off, gain-change flag at frame 400) matched bit-for-bit on this Linux host. Portable regression tests also compare direct and patched graph outputs/metrics, into/batch equivalence, ownership, malformed/read-only/overlapping buffers, non-finite input validation before mutation and full-stage graph fallback. Existing allocating API return contracts are retained.

No end-to-end zero-copy or zero-allocation claim is made. AEC requires normalization, deinterleaving and persistent state; Python snapshots and NumPy borrow checks allocate. Reusing arrays removes audio-buffer allocation at the boundary; the direct DSP core and reusable FFT/render storage remove the measured hot-loop heap allocations. Optional NS/AGC2/post-filter graph stages retain their own allocation behavior.
