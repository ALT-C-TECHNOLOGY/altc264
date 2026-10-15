//! Threaded WebAssembly: V8 updates a thread's view of the memory size late
//! after another thread grows it, and accesses past the stale size trap.
//! Linked with `--wrap` for the allocator functions and `sbrk` (see
//! .cargo/config.toml), the allocator counts each growth, and every allocator
//! call, under one lock, first refreshes the calling thread's view if the
//! count has changed since it last looked.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

static LOCK: Mutex<()> = Mutex::new(());
static GROWN: AtomicU32 = AtomicU32::new(0);

thread_local! {
    static SEEN: Cell<u32> = const { Cell::new(0) };
}

unsafe extern "C" {
    fn __real_malloc(size: usize) -> *mut c_void;
    fn __real_calloc(n: usize, size: usize) -> *mut c_void;
    fn __real_realloc(p: *mut c_void, size: usize) -> *mut c_void;
    fn __real_free(p: *mut c_void);
    fn __real_aligned_alloc(align: usize, size: usize) -> *mut c_void;
    fn __real_posix_memalign(p: *mut *mut c_void, align: usize, size: usize) -> i32;
    fn __real_sbrk(increment: isize) -> *mut c_void;
}

/// Refreshes this thread's view of the memory if it has grown since.
fn sync() {
    let grown = GROWN.load(Ordering::Acquire);
    SEEN.with(|seen| {
        if seen.get() != grown {
            std::arch::wasm32::memory_grow(0, 0);
            seen.set(grown);
        }
    });
}

fn locked<T>(f: impl FnOnce() -> T) -> T {
    let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    sync();
    f()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_malloc(size: usize) -> *mut c_void {
    locked(|| unsafe { __real_malloc(size) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_calloc(n: usize, size: usize) -> *mut c_void {
    locked(|| unsafe { __real_calloc(n, size) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_realloc(p: *mut c_void, size: usize) -> *mut c_void {
    locked(|| unsafe { __real_realloc(p, size) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_free(p: *mut c_void) {
    locked(|| unsafe { __real_free(p) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_aligned_alloc(align: usize, size: usize) -> *mut c_void {
    locked(|| unsafe { __real_aligned_alloc(align, size) })
}

#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_posix_memalign(p: *mut *mut c_void, align: usize, size: usize) -> i32 {
    locked(|| unsafe { __real_posix_memalign(p, align, size) })
}

// Called by the allocator, under the lock.
#[unsafe(no_mangle)]
unsafe extern "C" fn __wrap_sbrk(increment: isize) -> *mut c_void {
    let p = unsafe { __real_sbrk(increment) };
    if increment != 0 {
        GROWN.fetch_add(1, Ordering::Release);
    }
    p
}
