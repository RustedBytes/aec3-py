use aec3::api::control::Metrics as RustMetrics;
use aec3::graph::GraphError;
use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
    inner: linear::LinearPipeline,
    frame_samples: usize,
    render_channels: usize,
    capture_channels: usize,
    last_metrics: RustMetrics,
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

fn not_contiguous(kind: &str, e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<PyValueError, _>(format!(
        "{kind} array must be contiguous in memory: {e}"
    ))
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
        )
    )]
    fn new(
        sample_rate_hz: i32,
        render_channels: usize,
        capture_channels: usize,
        initial_delay_ms: Option<i32>,
        enable_high_pass: Option<bool>,
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

        let mut builder = linear::builder(render_format, capture_format).export_metrics(true);

        if let Some(delay) = initial_delay_ms {
            builder = builder.initial_delay_ms(delay);
        }
        if let Some(hp) = enable_high_pass {
            builder = builder.enable_high_pass_filter(hp);
        }

        let pipeline = builder.build().map_err(map_graph_err)?;
        let frame_samples = capture_format.frames_per_channel as usize; // per 10 ms, per channel

        Ok(Self {
            inner: pipeline,
            frame_samples,
            render_channels,
            capture_channels,
            last_metrics: RustMetrics::default(),
        })
    }

    /// Number of samples **per channel** in a 10 ms frame.
    #[getter]
    fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Configured sample rate (Hz).
    #[getter]
    fn sample_rate_hz(&self) -> i32 {
        self.inner.capture_format().sample_rate_hz as i32
    }

    /// Update the render-to-capture delay estimate (ms).
    ///
    /// This is equivalent to `LinearPipeline::set_delay_ms`.
    fn set_audio_buffer_delay(&mut self, delay_ms: i32) -> PyResult<()> {
        self.inner.set_delay_ms(delay_ms).map_err(map_graph_err)
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

        self.inner.handle_render_frame(slice).map_err(map_graph_err)?;
        self.pull_metrics()
    }

    /// Process a capture (microphone) frame.
    ///
    /// Parameters
    /// ----------
    /// capture_frame : numpy.ndarray
    ///     1D float32 array, length = frame_samples * capture_channels
    /// level_change : bool, optional
    ///     Accepted for backwards compatibility; ignored by the new pipeline.
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
        #[allow(unused_variables)] level_change: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, PyMetrics)> {
        let capture_slice = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;

        let expected = self.frame_samples * self.capture_channels;
        if capture_slice.len() != expected {
            return Err(bad_len("capture_frame", capture_slice.len(), expected));
        }

        let mut out = vec![0.0f32; capture_slice.len()];
        self.inner
            .process_capture_frame(capture_slice, &mut out)
            .map_err(map_graph_err)?;
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
    ///     Accepted for backwards compatibility; ignored by the new pipeline.
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
        #[allow(unused_variables)] level_change: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, PyMetrics)> {
        let capture_slice = capture_frame
            .as_slice()
            .map_err(|e| not_contiguous("capture_frame", e))?;

        let expected_capture = self.frame_samples * self.capture_channels;
        if capture_slice.len() != expected_capture {
            return Err(bad_len("capture_frame", capture_slice.len(), expected_capture));
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

        if let Some(render_slice) = render_slice_opt {
            self.inner
                .handle_render_frame(render_slice)
                .map_err(map_graph_err)?;
        }

        let mut out = vec![0.0f32; capture_slice.len()];
        self.inner
            .process_capture_frame(capture_slice, &mut out)
            .map_err(map_graph_err)?;
        self.pull_metrics()?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(self.last_metrics)))
    }
}

impl PyAec3 {
    /// Drain the metrics export sink, caching the most recent sample.
    fn pull_metrics(&mut self) -> PyResult<()> {
        loop {
            match self.inner.try_pull_metrics().map_err(map_graph_err)? {
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
    m.add_class::<PyAec3>()?;
    m.add_class::<PyMetrics>()?;
    Ok(())
}
