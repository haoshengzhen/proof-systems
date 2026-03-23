//! A `no_std`-compatible replacement for [`std::sync::LazyLock`].
//!
//! When the `std` feature is enabled, this delegates directly to
//! [`std::sync::LazyLock`]. In `no_std` mode, initialization is coordinated
//! with two [`AtomicBool`](core::sync::atomic::AtomicBool)s:
//!
//! - `initializing` — claimed via `compare_exchange` by the first thread to
//!   arrive. The winner runs the init function and writes the value.
//! - `initialized` — set with `Release` ordering after the value is written.
//!   All readers `Acquire` this flag before accessing the value.
//!
//! Concurrent callers that lose the race spin on `initialized` via
//! [`core::hint::spin_loop`].

use core::ops::Deref;

/// A thread-safe, lazily-initialized value.
///
/// Constructed with a function (typically `fn() -> T`) so that it can be used
/// in `static` items via [`LazyLock::new`], which is `const`.
///
/// # Examples
///
/// ```
/// use o1_utils::lazy_lock::LazyLock;
///
/// static VALUE: LazyLock<u64> = LazyLock::new(|| 1 + 2);
///
/// assert_eq!(*VALUE, 3);
/// ```
#[cfg(feature = "std")]
pub struct LazyLock<T, F = fn() -> T> {
    inner: std::sync::LazyLock<T, F>,
}

#[cfg(feature = "std")]
impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    pub const fn new(init: F) -> Self {
        Self {
            inner: std::sync::LazyLock::new(init),
        }
    }
}

#[cfg(feature = "std")]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

/// A thread-safe, lazily-initialized value (no-std fallback).
///
/// See the [module-level documentation](self) for the synchronization protocol.
#[cfg(not(feature = "std"))]
pub struct LazyLock<T, F = fn() -> T> {
    initialized: core::sync::atomic::AtomicBool,
    initializing: core::sync::atomic::AtomicBool,
    data: core::cell::UnsafeCell<core::mem::MaybeUninit<T>>,
    init: core::cell::UnsafeCell<Option<F>>,
}

// SAFETY: Access to `data` and `init` is synchronized through the two atomic
// bools. Only the thread that wins the `initializing` compare-exchange writes
// to `data` and `init`. All other threads spin until `initialized` is set,
// after which `data` is read-only.
#[cfg(not(feature = "std"))]
#[allow(unsafe_code)]
unsafe impl<T: Send + Sync, F: Send> Sync for LazyLock<T, F> {}
#[cfg(not(feature = "std"))]
#[allow(unsafe_code)]
unsafe impl<T: Send, F: Send> Send for LazyLock<T, F> {}

#[cfg(not(feature = "std"))]
impl<T, F: FnOnce() -> T> LazyLock<T, F> {
    pub const fn new(init: F) -> Self {
        Self {
            initialized: core::sync::atomic::AtomicBool::new(false),
            initializing: core::sync::atomic::AtomicBool::new(false),
            data: core::cell::UnsafeCell::new(core::mem::MaybeUninit::uninit()),
            init: core::cell::UnsafeCell::new(Some(init)),
        }
    }

    #[allow(unsafe_code)]
    fn force(&self) -> &T {
        use core::sync::atomic::Ordering;

        if self.initialized.load(Ordering::Acquire) {
            // SAFETY: `initialized` is only set after `data` has been fully
            // written, and `Acquire` ordering synchronizes with the `Release`
            // store in the initializing thread.
            return unsafe { (*self.data.get()).assume_init_ref() };
        }

        if self
            .initializing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // We won the race — initialize the value.
            // SAFETY: We are the only thread past the compare-exchange, so we
            // have exclusive access to `data` and `init`.
            unsafe {
                let init = (*self.init.get()).take().unwrap();
                (*self.data.get()).write(init());
            }
            self.initialized.store(true, Ordering::Release);
        } else {
            // Another thread is initializing — spin until done.
            while !self.initialized.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }
        }

        // SAFETY: `initialized` is true, so `data` is fully written.
        #[allow(unsafe_code)]
        unsafe {
            (*self.data.get()).assume_init_ref()
        }
    }
}

#[cfg(not(feature = "std"))]
impl<T, F: FnOnce() -> T> Deref for LazyLock<T, F> {
    type Target = T;

    fn deref(&self) -> &T {
        self.force()
    }
}
