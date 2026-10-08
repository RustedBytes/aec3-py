#[cfg(feature = "allocation-tracking")]
mod allocation;
mod direct;
use aec3::api::control::Metrics as RustMetrics;
use aec3::graph::{GraphError, PacketMeta};
use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-facing metrics object: thin wrapper around aec3::api::control::Metrics
#[pyclass(name = "Metrics")]
#[derive(Debug)]
pub struct PyMetrics {
    /// Echo Return Loss (dB)
    #[pyo3(get)]
    pub echo_return_loss: f64,
    /// Echo Return Loss Enhancement (dB)
    #[pyo3(get)]
    pub echo_return_loss_enhancement: f64,
    /// Estimated delay (ms)
    #[pyo3(get)]
    pub delay_ms: i32,
    /// Minimum number of consecutive render calls between capture calls.
    #[pyo3(get)]
    pub render_jitter_min: i32,
    /// Maximum number of consecutive render calls between capture calls.
    #[pyo3(get)]
    pub render_jitter_max: i32,
    /// Minimum number of consecutive capture calls between render calls.
    #[pyo3(get)]
    pub capture_jitter_min: i32,
    /// Maximum number of consecutive capture calls between render calls.
    #[pyo3(get)]
    pub capture_jitter_max: i32,
}

impl From<RustMetrics> for PyMetrics {
    fn from(m: RustMetrics) -> Self {
        PyMetrics {
            echo_return_loss: m.echo_return_loss,
            echo_return_loss_enhancement: m.echo_return_loss_enhancement,
            delay_ms: m.delay_ms,
            render_jitter_min: m.render_jitter_min,
            render_jitter_max: m.render_jitter_max,
            capture_jitter_min: m.capture_jitter_min,
            capture_jitter_max: m.capture_jitter_max,
        }
    }
}

/// High-level wrapper around aec3::pipelines::linear::LinearPipeline, using NumPy arrays.
///
/// The pipeline chain is: render reference + microphone capture -> high-pass filter
/// -> AEC3 -> noise suppression -> AGC2 (with post filter at 48 kHz).
///
/// All frames are **1D interleaved float32** arrays with length:
///   `frame_samples * channels`
/// where `frame_samples` is per-channel samples for a 10 ms frame.
#[pyclass(name = "Aec3", unsendable)]
pub struct PyAec3 {
    inner: Engine,
    sample_rate: i32,
    frame_samples: usize,
    render_channels: usize,
    capture_channels: usize,
    last_metrics: RustMetrics,
    capture_sequence: u64,
}

enum Engine {
    Direct(Box<direct::Direct>),
    Graph(Box<linear::LinearPipeline>),
}

fn map_graph_err(err: GraphError) -> PyErr {
    PyErr::new::<PyValueError, _>(err.to_string())
}

fn bad_len(kind: &str, got: usize, expected: usize) -> PyErr {
    PyErr::new::<PyValueError, _>(format!(
        "{kind} length {got} != expected {expected} \
         (frame_samples * {kind}_channels)"
    ))
}

fn validate_samples(kind: &str, samples: &[f32]) -> PyResult<()> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{kind} must contain only finite samples"
        )));
    }
    Ok(())
}

fn not_contiguous(kind: &str, e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{kind} array must be contiguous in memory: {e}"))
}

#[pymethods]
impl PyAec3 {
    /// __init__(
    ///   sample_rate_hz: int,
    ///   render_channels: int,
    ///   capture_channels: int,
    ///   initial_delay_ms: Optional[int] = None,
    ///   enable_high_pass: Optional[bool] = None,
    /// )
    ///
    /// sample_rate_hz must be one of {16000, 32000, 48000}.
    #[new]
    #[pyo3(
        signature = (
            sample_rate_hz,
            render_channels,
            capture_channels,
            initial_delay_ms = None,
            enable_high_pass = None,
            *,
            enable_noise_suppression = false,
            enable_gain_controller2 = false,
            enable_post_filter = false,
            backend = "auto",
        )
    )]
    #[allow(clippy::too_many_arguments)] // Preserve positional API and add keyword-only stages.
    fn new(
        sample_rate_hz: i32,
        render_channels: usize,
        capture_channels: usize,
        initial_delay_ms: Option<i32>,
        enable_high_pass: Option<bool>,
        enable_noise_suppression: bool,
        enable_gain_controller2: bool,
        enable_post_filter: bool,
        backend: &str,
    ) -> PyResult<Self> {
        if !matches!(sample_rate_hz, 16_000 | 32_000 | 48_000) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "sample_rate_hz {sample_rate_hz} not supported, expected one of 16000, 32000, 48000"
            )));
        }
        if render_channels == 0
            || capture_channels == 0
            || render_channels > u16::MAX as usize
            || capture_channels > u16::MAX as usize
        {
            return Err(PyErr::new::<PyValueError, _>(
                "render_channels and capture_channels must be in 1..=65535",
            ));
        }

        let render_format = AudioFormat::ten_ms(sample_rate_hz as u32, render_channels as u16);
        let capture_format = AudioFormat::ten_ms(sample_rate_hz as u32, capture_channels as u16);

        let mut builder = linear::builder(render_format, capture_format)
            .export_metrics(true)
            .enable_noise_suppression(enable_noise_suppression)
            .enable_gain_controller2(enable_gain_controller2)
            .enable_post_filter(enable_post_filter);

        if let Some(delay) = initial_delay_ms {
            builder = builder.initial_delay_ms(delay);
        }
        if let Some(hp) = enable_high_pass {
            builder = builder.enable_high_pass_filter(hp);
        }

        let direct_supported = render_channels <= 2
            && capture_channels <= 2
            && !enable_noise_suppression
            && !enable_gain_controller2
            && !enable_post_filter;
        let use_direct = match backend {
            "auto" => direct_supported,
            "graph" => false,
            "direct" if direct_supported => true,
            "direct" => {
                return Err(PyValueError::new_err(
                    "direct backend requires mono/stereo HPF + AEC only",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "backend must be auto, direct or graph",
                ));
            }
        };
        let pipeline = if use_direct {
            Engine::Direct(Box::new(direct::Direct::new(
                sample_rate_hz as usize,
                render_channels,
                capture_channels,
                enable_high_pass.unwrap_or(true),
                initial_delay_ms,
            )))
        } else {
            Engine::Graph(Box::new(builder.build().map_err(map_graph_err)?))
        };
        let frame_samples = capture_format.frames_per_channel as usize; // per 10 ms, per channel

        Ok(Self {
            inner: pipeline,
            sample_rate: sample_rate_hz,
            frame_samples,
            render_channels,
            capture_channels,
            last_metrics: RustMetrics::default(),
            capture_sequence: 0,
        })
    }

    /// Selected execution backend (direct for the default mono/stereo stages).
    #[getter]
    fn backend(&self) -> &'static str {
        match self.inner {
            Engine::Direct(_) => "direct",
            Engine::Graph(_) => "graph",
        }
    }

    /// Write into caller-owned contiguous float32 output and return None.
    /// Input/output arrays must not overlap. Read metrics separately when needed.
    #[pyo3(signature = (capture_frame, output_frame, render_frame=None, level_change=false))]
    fn process_into(
        &mut self,
        capture_frame: &Bound<'_, PyArray1<f32>>,
        output_frame: &Bound<'_, PyArray1<f32>>,
        render_frame: Option<&Bound<'_, PyArray1<f32>>>,
        level_change: bool,
    ) -> PyResult<()> {
        let capture_frame = capture_frame
            .try_readonly()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut output_frame = output_frame
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let render_frame = render_frame
            .map(|arr| arr.try_readonly())
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let capture = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;
        let output = output_frame
            .as_slice_mut()
            .map_err(|e| not_contiguous("output_frame", e))?;
        let expected = self.frame_samples * self.capture_channels;
        if capture.len() != expected {
            return Err(bad_len("capture_frame", capture.len(), expected));
        }
        if output.len() != expected {
            return Err(bad_len("output_frame", output.len(), expected));
        }
        validate_samples("capture_frame", capture)?;
        let render = render_frame
            .as_ref()
            .map(|arr| {
                arr.as_slice()
                    .map_err(|e| not_contiguous("render_frame", e))
            })
            .transpose()?;
        if let Some(render) = render {
            let expected = self.frame_samples * self.render_channels;
            if render.len() != expected {
                return Err(bad_len("render_frame", render.len(), expected));
            }
            validate_samples("render_frame", render)?;
            self.process_render(render)?;
        }
        self.process_capture(capture, level_change, output)?;
        self.pull_metrics()
    }

    /// Process a contiguous (frames, interleaved_samples) batch into reusable output.
    /// NumPy borrow checks happen once per batch. All inputs are validated first.
    #[pyo3(signature = (capture_frames, output_frames, render_frames=None, level_change=false))]
    fn process_frames_into(
        &mut self,
        capture_frames: &Bound<'_, PyArray2<f32>>,
        output_frames: &Bound<'_, PyArray2<f32>>,
        render_frames: Option<&Bound<'_, PyArray2<f32>>>,
        level_change: bool,
    ) -> PyResult<()> {
        if !capture_frames.is_c_contiguous()
            || !output_frames.is_c_contiguous()
            || render_frames.is_some_and(|arr| !arr.is_c_contiguous())
        {
            return Err(PyValueError::new_err(
                "batch arrays must be C-contiguous (frames, interleaved_samples)",
            ));
        }
        let capture_frames = capture_frames
            .try_readonly()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut output_frames = output_frames
            .try_readwrite()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let render_frames = render_frames
            .map(|arr| arr.try_readonly())
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape = capture_frames.shape();
        let samples = self.frame_samples * self.capture_channels;
        if shape[1] != samples || output_frames.shape() != shape {
            return Err(PyValueError::new_err(
                "capture/output shape must be (frames, frame_samples * capture_channels)",
            ));
        }
        let capture = capture_frames
            .as_slice()
            .map_err(|e| not_contiguous("capture_frames", e))?;
        let output = output_frames
            .as_slice_mut()
            .map_err(|e| not_contiguous("output_frames", e))?;
        validate_samples("capture_frames", capture)?;
        let render_samples = self.frame_samples * self.render_channels;
        let render = if let Some(ref arr) = render_frames {
            if arr.shape() != [shape[0], render_samples] {
                return Err(PyValueError::new_err(
                    "render shape must be (frames, frame_samples * render_channels)",
                ));
            }
            let slice = arr
                .as_slice()
                .map_err(|e| not_contiguous("render_frames", e))?;
            validate_samples("render_frames", slice)?;
            Some(slice)
        } else {
            None
        };
        for (i, (capture, output)) in capture
            .chunks_exact(samples)
            .zip(output.chunks_exact_mut(samples))
            .enumerate()
        {
            if let Some(render) = render {
                self.process_render(&render[i * render_samples..(i + 1) * render_samples])?;
            }
            self.process_capture(capture, level_change && i == 0, output)?;
        }
        self.pull_metrics()
    }

    /// Number of samples **per channel** in a 10 ms frame.
    #[getter]
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Configured sample rate (Hz).
    #[getter]
    fn sample_rate_hz(&self) -> i32 {
        self.sample_rate
    }

    /// Update the render-to-capture delay estimate (ms).
    ///
    /// This is equivalent to `LinearPipeline::set_delay_ms`.
    fn set_audio_buffer_delay(&mut self, delay_ms: i32) -> PyResult<()> {
        match &mut self.inner {
            Engine::Direct(inner) => {
                inner.delay(delay_ms);
                Ok(())
            }
            Engine::Graph(inner) => inner.set_delay_ms(delay_ms).map_err(map_graph_err),
        }
    }

    /// Get current AEC metrics (drains any metrics emitted since the last call).
    fn metrics(&mut self) -> PyResult<PyMetrics> {
        self.pull_metrics()?;
        Ok(PyMetrics::from(self.last_metrics))
    }

    /// Feed a far-end (render) frame into the pipeline.
    ///
    /// Parameters
    /// ----------
    /// render_frame : numpy.ndarray
    ///     1D float32 array, length = frame_samples * render_channels
    fn handle_render_frame(&mut self, render_frame: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let slice = render_frame
            .as_slice()
            .map_err(|e| not_contiguous("render_frame", e))?;

        let expected = self.frame_samples * self.render_channels;
        if slice.len() != expected {
            return Err(bad_len("render_frame", slice.len(), expected));
        }

        validate_samples("render_frame", slice)?;
        self.process_render(slice)?;
        self.pull_metrics()
    }

    /// Process a capture (microphone) frame.
    ///
    /// Parameters
    /// ----------
    /// capture_frame : numpy.ndarray
    ///     1D float32 array, length = frame_samples * capture_channels
    /// level_change : bool, optional
    ///     Signals a capture gain change to AEC3 via packet metadata.
    ///
    /// Returns
    /// -------
    /// (out_frame, metrics)
    ///   out_frame : numpy.ndarray (float32, same length as capture_frame)
    ///   metrics   : Metrics
    #[pyo3(signature = (capture_frame, level_change=false))]
    fn process_capture_frame<'py>(
        &mut self,
        py: Python<'py>,
        capture_frame: PyReadonlyArray1<'py, f32>,
        level_change: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, PyMetrics)> {
        let capture_slice = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;

        let expected = self.frame_samples * self.capture_channels;
        if capture_slice.len() != expected {
            return Err(bad_len("capture_frame", capture_slice.len(), expected));
        }

        validate_samples("capture_frame", capture_slice)?;
        let mut out = vec![0.0f32; capture_slice.len()];
        self.process_capture(capture_slice, level_change, &mut out)?;
        self.pull_metrics()?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(self.last_metrics)))
    }

    /// Combined convenience method: optionally feed a render frame, then process
    /// a capture frame.
    ///
    /// Parameters
    /// ----------
    /// capture_frame : numpy.ndarray
    ///     1D float32 array, length = frame_samples * capture_channels
    /// render_frame : Optional[numpy.ndarray]
    ///     1D float32 array, length = frame_samples * render_channels
    /// level_change : bool, optional
    ///     Signals a capture gain change to AEC3 via packet metadata.
    ///
    /// Returns
    /// -------
    /// (out_frame, metrics)
    ///   out_frame : numpy.ndarray (float32, same length as capture_frame)
    ///   metrics   : Metrics
    #[pyo3(signature = (capture_frame, render_frame=None, level_change=false))]
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        capture_frame: PyReadonlyArray1<'py, f32>,
        render_frame: Option<PyReadonlyArray1<'py, f32>>,
        level_change: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, PyMetrics)> {
        let capture_slice = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;

        let expected_capture = self.frame_samples * self.capture_channels;
        if capture_slice.len() != expected_capture {
            return Err(bad_len(
                "capture_frame",
                capture_slice.len(),
                expected_capture,
            ));
        }

        let render_slice_opt: Option<&[f32]> = if let Some(ref arr) = render_frame {
            let slice = arr
                .as_slice()
                .map_err(|e| not_contiguous("render_frame", e))?;

            let expected_render = self.frame_samples * self.render_channels;
            if slice.len() != expected_render {
                return Err(bad_len("render_frame", slice.len(), expected_render));
            }

            Some(slice)
        } else {
            None
        };

        validate_samples("capture_frame", capture_slice)?;
        if let Some(render_slice) = render_slice_opt {
            validate_samples("render_frame", render_slice)?;
            self.process_render(render_slice)?;
        }

        let mut out = vec![0.0f32; capture_slice.len()];
        self.process_capture(capture_slice, level_change, &mut out)?;
        self.pull_metrics()?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(self.last_metrics)))
    }
}

impl PyAec3 {
    fn process_render(&mut self, render: &[f32]) -> PyResult<()> {
        match &mut self.inner {
            Engine::Direct(inner) => {
                inner.render(render);
                Ok(())
            }
            Engine::Graph(inner) => inner.handle_render_frame(render).map_err(map_graph_err),
        }
    }
    fn process_capture(
        &mut self,
        capture: &[f32],
        level_change: bool,
        out: &mut [f32],
    ) -> PyResult<()> {
        let meta = PacketMeta {
            sequence: Some(self.capture_sequence),
            discontinuity: level_change,
            ..PacketMeta::default()
        };
        self.capture_sequence = self.capture_sequence.wrapping_add(1);
        let ready = match &mut self.inner {
            Engine::Direct(inner) => {
                self.last_metrics = inner.capture(capture, level_change, out);
                true
            }
            Engine::Graph(inner) => inner
                .process_capture_frame_with_meta(capture, meta, out)
                .map_err(map_graph_err)?,
        };
        if !ready {
            return Err(PyValueError::new_err(
                "audio pipeline produced no capture frame",
            ));
        }
        Ok(())
    }

    /// Drain the metrics export sink, caching the most recent sample.
    fn pull_metrics(&mut self) -> PyResult<()> {
        let Engine::Graph(inner) = &mut self.inner else {
            return Ok(());
        };
        loop {
            match inner.try_pull_metrics().map_err(map_graph_err)? {
                Some(packet) => self.last_metrics = *packet.payload(),
                None => return Ok(()),
            }
        }
    }
}

/// Python module init.
/// The name here (`aec3_py`) must match the `[lib].name` in Cargo.toml.
#[pymodule]
fn aec3_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "allocation-tracking")]
    allocation::register(m)?;
    m.add_class::<PyAec3>()?;
    m.add_class::<PyMetrics>()?;
    Ok(())
}
