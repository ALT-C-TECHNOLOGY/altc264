//! Decoding progress of a picture that other pictures may use before it
//! is finished (frame threading): how many of its rows are final, and its
//! macroblocks once it is done.
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use crate::slice_decoder::MbInfo;

pub struct PicState {
    /// Luma rows of the frame (all its planes' corresponding rows) whose
    /// samples are final; usize::MAX once the picture is done.
    rows: AtomicUsize,
    /// The macroblocks: written by the one thread decoding the picture
    /// (through [`Self::info_ptr`]) until their row is published, then only
    /// read, by any thread (see the Sync impl).
    info: UnsafeCell<Vec<MbInfo>>,
    /// Rows of macroblocks published as they are decoded, for direct
    /// prediction of pictures decoded meanwhile.
    row_ready: Box<[AtomicBool]>,
    /// Macroblocks per published row, and in the picture.
    per_row: usize,
    mbs: usize,
    lock: Mutex<()>,
    changed: Condvar,
}

// SAFETY: the macroblocks are written only by the thread decoding the
// picture, and only those of rows not yet published (publish_info_row,
// after which that thread neither writes nor reborrows them mutably);
// other threads read only published rows (mb waits for them) or the whole
// once the picture is done (Release/Acquire on row_ready and rows).
unsafe impl Sync for PicState {}

impl PicState {
    /// A finished picture.
    pub fn done(info: Vec<MbInfo>) -> std::sync::Arc<Self> {
        let s = Self::pending(0, 0, info);
        s.finish();
        s
    }

    /// A picture about to be decoded, with its macroblocks `info`
    /// published in `rows` rows of `per_row`.
    pub fn pending(rows: usize, per_row: usize, info: Vec<MbInfo>) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            rows: AtomicUsize::new(0),
            mbs: info.len(),
            info: UnsafeCell::new(info),
            row_ready: (0..rows).map(|_| AtomicBool::new(false)).collect(),
            per_row,
            lock: Mutex::new(()),
            changed: Condvar::new(),
        })
    }

    /// The macroblocks, for the thread decoding the picture to write.
    ///
    /// # Safety
    ///
    /// Only that thread may write through it, and only macroblocks of rows
    /// it has not published, before the picture is done.
    pub(crate) unsafe fn info_ptr(&self) -> (*mut MbInfo, usize) {
        // SAFETY: no reference to the Vec is made; the buffer is not resized.
        unsafe { ((*self.info.get()).as_mut_ptr(), self.mbs) }
    }

    /// The macroblocks of a finished picture nothing else refers to.
    pub fn into_info(self) -> Option<Vec<MbInfo>> {
        Some(self.info.into_inner())
    }

    /// Number of macroblocks.
    pub fn len(&self) -> usize {
        self.mbs
    }

    pub fn is_empty(&self) -> bool {
        self.mbs == 0
    }

    /// Macroblock `addr`, waiting for its row to be decoded.
    pub fn mb(&self, addr: usize) -> Option<&MbInfo> {
        if addr >= self.mbs {
            return None;
        }
        let readable = || {
            self.is_done()
                || self
                    .row_ready
                    .get(addr / self.per_row.max(1))
                    .is_some_and(|r| r.load(Ordering::Acquire))
        };
        if !readable() {
            crate::pool::blocking(|| {
                let mut g = self.lock.lock().unwrap();
                while !readable() {
                    g = self.changed.wait(g).unwrap();
                }
            });
        }
        // SAFETY: its row is published or the picture done: not written
        // any more (see the Sync impl).
        Some(unsafe { &*(*self.info.get()).as_ptr().add(addr) })
    }

    /// Row `r` of macroblocks is decoded: it is not written any more.
    pub(crate) fn publish_info_row(&self, r: usize) {
        let Some(row) = self.row_ready.get(r) else {
            return;
        };
        if !row.swap(true, Ordering::Release) {
            let _g = self.lock.lock().unwrap();
            self.changed.notify_all();
        }
    }

    pub fn is_done(&self) -> bool {
        self.rows.load(Ordering::Acquire) == usize::MAX
    }

    /// Waits until luma rows `0..rows` are final.
    #[inline]
    pub fn wait_rows(&self, rows: usize) {
        if self.rows.load(Ordering::Acquire) >= rows {
            return;
        }
        self.wait_slow(rows);
    }

    #[cold]
    fn wait_slow(&self, rows: usize) {
        crate::pool::blocking(|| {
            let mut g = self.lock.lock().unwrap();
            while self.rows.load(Ordering::Acquire) < rows {
                g = self.changed.wait(g).unwrap();
            }
        })
    }

    pub fn wait_done(&self) {
        self.wait_rows(usize::MAX);
    }

    /// The macroblocks, once the picture is done.
    pub fn info(&self) -> &Vec<MbInfo> {
        self.wait_done();
        // SAFETY: done, so not written any more (see the Sync impl).
        unsafe { &*self.info.get() }
    }

    /// Rows `0..rows` are final: their samples are not written any more.
    pub(crate) fn publish_rows(&self, rows: usize) {
        if rows <= self.rows.load(Ordering::Relaxed) {
            return;
        }
        let _g = self.lock.lock().unwrap();
        self.rows.store(rows, Ordering::Release);
        self.changed.notify_all();
    }

    /// The picture is done: its macroblocks are not written any more.
    pub(crate) fn finish(&self) {
        let _g = self.lock.lock().unwrap();
        self.rows.store(usize::MAX, Ordering::Release);
        self.changed.notify_all();
    }
}
