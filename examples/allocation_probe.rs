//! Native allocation regression: no Python/NumPy metadata is involved.
#[path = "../src/direct.rs"]
mod direct;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static ENABLED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
fn main() {
    use aec3::audio_processing::aec3::multi_channel_content_detector::MultiChannelContentDetector;
    let mut detector = MultiChannelContentDetector::new(true, 2, 0.0, 1, 0.0);
    let frame = vec![vec![vec![0.0; 160], vec![1.0; 160]]];
    COUNT.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
    for _ in 0..70_000 {
        detector.update_detection(&frame);
    }
    ENABLED.store(false, Ordering::Relaxed);
    assert_eq!(
        COUNT.load(Ordering::Relaxed),
        0,
        "diagnostic history allocated"
    );
    assert_eq!(
        detector
            .metrics_logger()
            .unwrap()
            .processing_persistent_multichannel_content()
            .len(),
        64
    );
    println!("700-second diagnostic history: bounded to 64 samples, 0 allocations");
    for rate in [16000, 32000, 48000] {
        for rch in [1, 2] {
            for cch in [1, 2] {
                for hp in [false, true] {
                    let mut a = direct::Direct::new(rate, rch, cch, hp, Some(30));
                    let n = rate / 100;
                    let r: Vec<f32> = (0..n * rch)
                        .map(|i| (i as f32 * 0.123).sin() * 0.1)
                        .collect();
                    let c: Vec<f32> = (0..n * cch)
                        .map(|i| (i as f32 * 0.079).sin() * 0.05)
                        .collect();
                    let mut out = vec![0.0; n * cch];
                    for _ in 0..800 {
                        a.render(&r);
                        a.capture(&c, false, &mut out);
                    }
                    COUNT.store(0, Ordering::Relaxed);
                    ENABLED.store(true, Ordering::Relaxed);
                    for i in 0..100 {
                        a.render(&r);
                        a.capture(&c, i == 50, &mut out);
                    }
                    a.delay(40);
                    // Queue saturation and drain must also reuse preallocated slots.
                    for _ in 0..101 {
                        a.render(&r);
                    }
                    a.capture(&c, false, &mut out);
                    ENABLED.store(false, Ordering::Relaxed);
                    let count = COUNT.load(Ordering::Relaxed);
                    assert_eq!(
                        count, 0,
                        "{rate} {rch}->{cch} hp={hp}: heap allocation in DSP"
                    );
                    println!("{rate} {rch}->{cch} hp={hp}: 0 DSP allocations");
                }
            }
        }
    }
}
