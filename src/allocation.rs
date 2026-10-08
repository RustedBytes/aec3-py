//! Opt-in per-thread Rust allocation counter. Python/NumPy allocations are outside it.
use pyo3::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
thread_local! { static COUNTS: Cell<Option<(usize, usize)>> = const { Cell::new(None) }; }
struct Tracked;
fn record(bytes: usize) {
    let _ = COUNTS.try_with(|counts| {
        if let Some((n, size)) = counts.get() {
            counts.set(Some((n + 1, size + bytes)));
        }
    });
}
unsafe impl GlobalAlloc for Tracked {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Tracked = Tracked;
#[pyfunction]
fn _start_allocation_tracking() {
    COUNTS.with(|c| c.set(Some((0, 0))));
}
#[pyfunction]
fn _stop_allocation_tracking() -> (usize, usize) {
    COUNTS.with(|c| c.replace(None).unwrap_or_default())
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_start_allocation_tracking, m)?)?;
    m.add_function(wrap_pyfunction!(_stop_allocation_tracking, m)?)?;
    Ok(())
}
