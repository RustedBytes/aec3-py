//! Synchronous HPF + AEC path with reusable audio storage for mono/stereo.
//! Stage order and normalized conversion match aec3 0.4's linear graph.
use aec3::api::config::EchoCanceller3Config;
use aec3::api::control::{EchoControl, Metrics};
use aec3::audio_processing::aec3::echo_canceller3::EchoCanceller3;
use aec3::audio_processing::audio_buffer::AudioBuffer;
use aec3::audio_processing::high_pass_filter::HighPassFilter;
use aec3::audio_processing::stream_config::StreamConfig;

struct Io {
    config: StreamConfig,
    buffer: AudioBuffer,
    planar: [[f32; 480]; 2],
}
impl Io {
    fn new(rate: usize, channels: usize) -> Self {
        Self {
            config: StreamConfig::new(rate, channels, false),
            buffer: AudioBuffer::from_sample_rates(rate, channels, rate, channels, rate),
            planar: [[0.0; 480]; 2],
        }
    }
    fn load(&mut self, input: &[f32]) {
        if self.config.num_channels() == 1 {
            self.buffer.copy_from(&[input], &self.config);
        } else {
            let n = self.config.num_frames();
            for (i, sample) in input.as_chunks::<2>().0.iter().enumerate() {
                self.planar[0][i] = sample[0];
                self.planar[1][i] = sample[1];
            }
            self.buffer
                .copy_from(&[&self.planar[0][..n], &self.planar[1][..n]], &self.config);
        }
    }
    fn store(&mut self, out: &mut [f32]) {
        if self.config.num_channels() == 1 {
            self.buffer.copy_to_stream(&self.config, &mut [out]);
        } else {
            let n = self.config.num_frames();
            let [left, right] = &mut self.planar;
            self.buffer
                .copy_to_stream(&self.config, &mut [&mut left[..n], &mut right[..n]]);
            for (i, sample) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                sample[0] = left[i];
                sample[1] = right[i];
            }
        }
    }
}

pub(crate) struct Direct {
    echo: EchoCanceller3,
    render: Io,
    capture: Io,
    high_pass: Option<(Io, HighPassFilter, Vec<Vec<f32>>)>,
    filtered: [f32; 960],
}
impl Direct {
    pub(crate) fn new(
        rate: usize,
        render: usize,
        capture: usize,
        hp: bool,
        delay: Option<i32>,
    ) -> Self {
        let mut echo = EchoCanceller3::with_multichannel_config(
            EchoCanceller3Config::default(),
            Some(EchoCanceller3Config::create_default_multichannel_config()),
            rate as i32,
            render,
            capture,
        );
        if let Some(delay) = delay {
            echo.set_audio_buffer_delay(delay);
        }
        Self {
            echo,
            render: Io::new(rate, render),
            capture: Io::new(rate, capture),
            high_pass: hp.then(|| {
                (
                    Io::new(rate, capture),
                    HighPassFilter::new(rate as i32, capture),
                    vec![vec![0.0; 160]; capture],
                )
            }),
            filtered: [0.0; 960],
        }
    }
    pub(crate) fn delay(&mut self, delay: i32) {
        self.echo.set_audio_buffer_delay(delay);
    }
    pub(crate) fn render(&mut self, input: &[f32]) {
        self.render.load(input);
        self.render.buffer.split_into_frequency_bands();
        self.echo.analyze_render(&mut self.render.buffer);
    }
    pub(crate) fn capture(&mut self, input: &[f32], change: bool, output: &mut [f32]) -> Metrics {
        if let Some((io, hp, scratch)) = &mut self.high_pass {
            io.load(input);
            io.buffer.split_into_frequency_bands();
            for (ch, slot) in scratch.iter_mut().enumerate() {
                slot.copy_from_slice(io.buffer.split_band(ch, 0));
            }
            hp.process(scratch);
            for (ch, slot) in scratch.iter().enumerate() {
                io.buffer.split_band_mut(ch, 0).copy_from_slice(slot);
            }
            io.buffer.merge_frequency_bands();
            io.store(&mut self.filtered[..input.len()]);
            self.capture.load(&self.filtered[..input.len()]);
        } else {
            self.capture.load(input);
        }
        self.echo.analyze_capture(&mut self.capture.buffer);
        self.capture.buffer.split_into_frequency_bands();
        self.echo.process_capture(&mut self.capture.buffer, change);
        self.capture.buffer.merge_frequency_bands();
        self.capture.store(output);
        self.echo.metrics()
    }
}
