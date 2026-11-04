//! Linear-memory buffers for the browser benchmark page.
//!
//! The benchmark exports take raw pointers into WebAssembly linear memory so a
//! whole workload crosses the JS/WASM boundary in one call. JavaScript owns the
//! buffer lifetime: it allocates with [`bench_alloc`], fills the bytes through a
//! typed-array view and releases them with [`bench_free`] using the same size.

use std::alloc::{Layout, alloc, dealloc};

/// Alignment of every benchmark buffer; enough for `f64` and `u32` views.
const BUFFER_ALIGN: usize = 8;

fn layout(bytes: u32) -> Option<Layout> {
    Layout::from_size_align(bytes.max(1) as usize, BUFFER_ALIGN).ok()
}

/// Allocates `bytes` (8-byte aligned) of linear memory, or returns null.
#[unsafe(no_mangle)]
pub extern "C" fn bench_alloc(bytes: u32) -> *mut u8 {
    match layout(bytes) {
        // SAFETY: `layout` never has zero size.
        Some(layout) => unsafe { alloc(layout) },
        None => std::ptr::null_mut(),
    }
}

/// Releases a buffer returned by [`bench_alloc`].
///
/// # Safety
/// `ptr` must come from `bench_alloc(bytes)` with the same `bytes` and must not
/// be used afterwards. A null pointer is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bench_free(ptr: *mut u8, bytes: u32) {
    if ptr.is_null() {
        return;
    }
    if let Some(layout) = layout(bytes) {
        // SAFETY: guaranteed by the caller contract above.
        unsafe { dealloc(ptr, layout) };
    }
}

/// Borrows `len` values starting at `ptr`.
///
/// # Safety
/// `ptr` must be aligned for `T` and valid for `len` reads.
pub(crate) unsafe fn input<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    // SAFETY: guaranteed by the caller.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Mutably borrows `len` values starting at `ptr`.
///
/// # Safety
/// `ptr` must be aligned for `T`, valid for `len` writes and not aliased.
pub(crate) unsafe fn output<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 {
        return &mut [];
    }
    // SAFETY: guaranteed by the caller.
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

#[cfg(test)]
mod tests {
    use super::{bench_alloc, bench_free};

    #[test]
    fn buffers_are_aligned_and_releasable() {
        let ptr = bench_alloc(24);
        assert!(!ptr.is_null());
        assert_eq!(ptr as usize % 8, 0);
        // SAFETY: allocated above with the same size.
        unsafe { bench_free(ptr, 24) };
        // SAFETY: null is explicitly ignored.
        unsafe { bench_free(std::ptr::null_mut(), 0) };
    }
}
