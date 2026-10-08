# Vendored AEC3

`aec3/` contains the library sources of the published aec3 0.4.0 crate, archive checksum `95b94681e57494d8d3933597fbf49c3537be1f2f33647e41317aab3dfc3d6fc8`, https://crates.io/crates/aec3/0.4.0 and https://github.com/RubyBit/aec3-rs. LICENSE and PATENT are preserved. No dependencies were bumped. Vendoring keeps these DSP fixes reproducible in wheels and source builds without relying on an unpublished upstream commit.

Only library sources, a minimal Cargo manifest, LICENSE and PATENT are retained. Upstream CI, editor instructions, demos, CLI tools, integration tests, release notes, README, registry metadata, original manifest and nested lockfile are excluded. The manifest removes their targets and dev-dependencies; library dependencies and the diagnostics feature are unchanged. Test-support modules used by inline upstream unit tests are retained so those source modules remain intact.

DSP modifications (all other source files remain byte-identical):

- `aec3_fft.rs`: bounded stack scratch and `process_with_scratch`, constructor assertions on required scratch lengths.
- `echo_canceller3.rs`: preallocate all 100 render queue slots, swap/recycle tensor storage, preserve FIFO and drop-newest overflow behavior. Additional startup memory is approximately 65–391 KiB of sample storage for supported mono/stereo rates, plus Vec headers; it removes per-frame allocation/copy traffic.
- `matched_filter.rs`: compile dynamic diagnostic names only with `diagnostics`; preserve enabled diagnostic behavior.
- `multi_channel_content_detector.rs`: preallocate and bound diagnostic history to the most recent 64 samples (640 seconds), instead of growing a Vec every 10 seconds. Python AEC metrics are unchanged.

On an upstream update, compare these four files, rerun bit-equivalence and queue overflow tests, the native allocation probe, NumPy buffer tests, and the complete platform matrix. Do not enable diagnostics in a realtime zero-allocation build. Configuration changes that reconstruct the upstream canceller can still allocate.

Actual modified paths:
- `src/audio_processing/aec3/matched_filter.rs`
- `src/audio_processing/aec3/multi_channel_content_detector.rs`
- `src/audio_processing/aec3/aec3_fft.rs`
- `src/audio_processing/aec3/echo_canceller3.rs`
