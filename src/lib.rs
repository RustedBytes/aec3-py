use aec3::api::control::Metrics as RustMetrics;
use aec3::voip::{VoipAec3, VoipAec3Builder, VoipAec3Error};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python-facing metrics object: thin wrapper around aec3::api::control::Metrics
#[pyclass(name = "Metrics")]
#[derive(Clone, Debug)]
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
}

impl From<RustMetrics> for PyMetrics {
    fn from(m: RustMetrics) -> Self {
        PyMetrics {
            echo_return_loss: m.echo_return_loss,
            echo_return_loss_enhancement: m.echo_return_loss_enhancement,
            delay_ms: m.delay_ms,
        }
    }
}

/// High-level wrapper around aec3::voip::VoipAec3, using NumPy arrays.
///
/// All frames are **1D interleaved float32** arrays with length:
///   `frame_samples * channels`
/// where `frame_samples` is per-channel samples for a 10 ms frame. :contentReference[oaicite:2]{index=2}
#[pyclass(name = "Aec3", unsendable)]
pub struct PyAec3 {
    inner: VoipAec3,
    frame_samples: usize,
    render_channels: usize,
    capture_channels: usize,
}

fn map_voip_err(err: VoipAec3Error) -> PyErr {
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
    /// sample_rate_hz must be one of {16000, 32000, 48000}. :contentReference[oaicite:3]{index=3}
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
        let mut builder: VoipAec3Builder =
            VoipAec3::builder(sample_rate_hz, render_channels, capture_channels);

        if let Some(delay) = initial_delay_ms {
            builder = builder.initial_delay_ms(delay);
        }
        if let Some(hp) = enable_high_pass {
            builder = builder.enable_high_pass(hp);
        }

        let pipeline = builder.build().map_err(map_voip_err)?;
        let frame_samples = pipeline.frame_samples(); // per 10 ms, per channel

        Ok(Self {
            inner: pipeline,
            frame_samples,
            render_channels,
            capture_channels,
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
        self.inner.sample_rate_hz()
    }

    /// Update the audio buffer delay hint (ms).
    ///
    /// This is equivalent to `VoipAec3::set_audio_buffer_delay`. :contentReference[oaicite:4]{index=4}
    fn set_audio_buffer_delay(&mut self, delay_ms: i32) {
        self.inner.set_audio_buffer_delay(delay_ms);
    }

    /// Get current AEC metrics without processing a frame.
    fn metrics(&self) -> PyMetrics {
        PyMetrics::from(self.inner.metrics())
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

        self.inner.handle_render_frame(slice).map_err(map_voip_err)
    }

    /// Process a capture (microphone) frame.
    ///
    /// Parameters
    /// ----------
    /// py : Python
    /// capture_frame : numpy.ndarray
    ///     1D float32 array, length = frame_samples * capture_channels
    /// level_change : bool, optional
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

        let mut out = vec![0.0f32; capture_slice.len()];
        let metrics = self
            .inner
            .process_capture_frame(capture_slice, level_change, &mut out)
            .map_err(map_voip_err)?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(metrics)))
    }

    /// Combined convenience method mirroring `VoipAec3::process`.
    ///
    /// Parameters
    /// ----------
    /// py : Python
    /// capture_frame : numpy.ndarray
    ///     1D float32 array, length = frame_samples * capture_channels
    /// render_frame : Optional[numpy.ndarray]
    ///     1D float32 array, length = frame_samples * render_channels
    /// level_change : bool, optional
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

        let mut out = vec![0.0f32; capture_slice.len()];
        let metrics = self
            .inner
            .process(capture_slice, render_slice_opt, level_change, &mut out)
            .map_err(map_voip_err)?;

        let out_array = out.into_pyarray(py);
        Ok((out_array, PyMetrics::from(metrics)))
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
