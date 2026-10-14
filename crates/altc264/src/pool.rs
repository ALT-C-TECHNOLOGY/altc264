//! A fixed set of worker threads for the parallel parts of decoding a
//! picture (slices, deblocking rows), so they are not spawned per picture.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// The job of one broadcast, with its lifetime erased (see `broadcast`).
type Job = &'static (dyn Fn(usize) + Sync);

#[derive(Default)]
struct State {
    job: Option<Job>,
    /// Incremented for each broadcast, so a worker runs each job once.
    generation: u64,
    /// Workers still running the current job.
    running: usize,
    panic: Option<Box<dyn std::any::Any + Send>>,
    quit: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    work: Condvar,
    done: Condvar,
}

pub(crate) struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pool of `workers` threads; broadcasts run on `workers + 1`
    /// participants, the caller being the last.
    pub(crate) fn new(workers: usize) -> Self {
        let shared = Arc::new(Shared::default());
        let handles = (0..workers)
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("altc264-pool{i}"))
                    .spawn(move || worker(&shared, i))
                    .expect("spawn decoder thread")
            })
            .collect();
        Self { shared, handles }
    }

    /// Number of participants in a broadcast.
    pub(crate) fn size(&self) -> usize {
        self.handles.len() + 1
    }

    /// Runs `f(i)` for every participant `i` in `0..size()` at the same
    /// time, the caller running the last, and returns when all are done.
    pub(crate) fn broadcast<F: Fn(usize) + Sync>(&self, f: F) {
        let n = self.handles.len();
        if n == 0 {
            return f(0);
        }
        let job: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: the workers only use the job between this point and the
        // wait below for `running` to reach zero, which happens before this
        // function returns or unwinds (the guard waits on drop), so `f`
        // outlives every use.
        let job: Job = unsafe { std::mem::transmute(job) };
        {
            let mut st = self.shared.state.lock().unwrap();
            st.job = Some(job);
            st.generation += 1;
            st.running = n;
            st.panic = None;
        }
        self.shared.work.notify_all();

        struct Wait<'a>(&'a Shared);
        impl Drop for Wait<'_> {
            fn drop(&mut self) {
                let mut st = self.0.state.lock().unwrap();
                while st.running > 0 {
                    st = self.0.done.wait(st).unwrap();
                }
                st.job = None;
            }
        }
        let wait = Wait(&self.shared);
        f(n);
        drop(wait);
        // Take the payload first: unwinding with the lock held would poison it.
        let panic = self.shared.state.lock().unwrap().panic.take();
        if let Some(p) = panic {
            resume_unwind(p);
        }
    }
}

fn worker(shared: &Shared, i: usize) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if st.quit {
                    return;
                }
                if st.generation != seen
                    && let Some(job) = st.job
                {
                    seen = st.generation;
                    break job;
                }
                st = shared.work.wait(st).unwrap();
            }
        };
        let r = catch_unwind(AssertUnwindSafe(|| job(i)));
        let mut st = shared.state.lock().unwrap();
        if let Err(p) = r {
            st.panic.get_or_insert(p);
        }
        st.running -= 1;
        if st.running == 0 {
            shared.done.notify_all();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.work.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    #[cfg_attr(target_os = "wasi", ignore = "wasm32-wasip1 has no threads")]
    fn every_participant_runs_each_broadcast_once() {
        let pool = Pool::new(3);
        for round in 0..100 {
            let hits: Vec<AtomicUsize> = (0..pool.size()).map(|_| AtomicUsize::new(0)).collect();
            let local = [round; 4]; // borrowed by the job
            pool.broadcast(|i| {
                hits[i].fetch_add(local[0] + 1 - round, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    #[cfg_attr(target_os = "wasi", ignore = "wasm32-wasip1 has no threads")]
    fn worker_panics_reach_the_caller() {
        let pool = Pool::new(2);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.broadcast(|i| assert!(i != 0, "boom"));
        }));
        assert!(r.is_err());
        // The pool still works afterwards.
        let n = AtomicUsize::new(0);
        pool.broadcast(|_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 3);
    }
}

/// Worker threads running whole jobs (pictures, with frame threading) in
/// submission order, at most `cap` submitted and unfinished at a time.
///
/// Jobs may wait for jobs submitted before them (a picture for rows of its
/// references) but never for later ones; as they start in submission
/// order, the oldest unfinished job is always running, so waits end.
///
/// There are more workers than may run at once (`active`): one waiting
/// (see [`blocking`]) gives its turn to another, which keeps the CPUs busy
/// without more threads competing for them.
pub(crate) struct JobQueue {
    shared: Arc<JobShared>,
    handles: Vec<JoinHandle<()>>,
}

type Task = Box<dyn FnOnce() + Send>;

struct JobShared {
    state: Mutex<JobState>,
    work: Condvar,
    slot: Condvar,
    cap: usize,
    turns: Turns,
}

/// Turns to run: free ones, and workers waiting to resume a job.
struct Turns {
    st: Mutex<(usize, usize)>,
    changed: Condvar,
}

impl Turns {
    /// Takes a turn; a worker resuming a job goes before one starting one,
    /// as later jobs depend on earlier ones.
    fn take(&self, resume: bool) {
        let mut g = self.st.lock().unwrap();
        g.1 += resume as usize;
        while g.0 == 0 || !resume && g.1 > 0 {
            g = self.changed.wait(g).unwrap();
        }
        g.0 -= 1;
        g.1 -= resume as usize;
    }

    fn give(&self) {
        self.st.lock().unwrap().0 += 1;
        self.changed.notify_all();
    }
}

thread_local! {
    /// The turns of the job queue whose worker this thread is, if any.
    static TURNS: std::cell::Cell<*const Turns> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// Runs `wait`, which blocks until another job makes progress, giving the
/// worker's turn to another meanwhile.
pub(crate) fn blocking<R>(wait: impl FnOnce() -> R) -> R {
    let t = TURNS.get();
    if t.is_null() {
        return wait();
    }
    // SAFETY: set by job_worker, whose Arc keeps the queue alive.
    let t = unsafe { &*t };
    t.give();
    let r = wait();
    t.take(true);
    r
}

#[derive(Default)]
struct JobState {
    queue: std::collections::VecDeque<Task>,
    /// Submitted and not finished.
    unfinished: usize,
    quit: bool,
}

impl JobQueue {
    /// `workers` threads of which `active` run at a time.
    pub(crate) fn new(workers: usize, active: usize, cap: usize) -> Self {
        let shared = Arc::new(JobShared {
            state: Mutex::new(JobState::default()),
            work: Condvar::new(),
            slot: Condvar::new(),
            cap: cap.max(1),
            turns: Turns {
                st: Mutex::new((active.max(1), 0)),
                changed: Condvar::new(),
            },
        });
        let handles = (0..workers.max(1))
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("altc264-frame{i}"))
                    .spawn(move || job_worker(&shared))
                    .expect("spawn decoder thread")
            })
            .collect();
        Self { shared, handles }
    }

    /// Queues `task`, first waiting while `cap` jobs are unfinished.
    pub(crate) fn submit(&self, task: impl FnOnce() + Send + 'static) {
        let mut st = self.shared.state.lock().unwrap();
        while st.unfinished >= self.shared.cap {
            st = self.shared.slot.wait(st).unwrap();
        }
        st.unfinished += 1;
        st.queue.push_back(Box::new(task));
        drop(st);
        self.shared.work.notify_one();
    }
}

fn job_worker(shared: &JobShared) {
    TURNS.set(&shared.turns);
    loop {
        let task = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if let Some(t) = st.queue.pop_front() {
                    break t;
                }
                if st.quit {
                    return;
                }
                st = shared.work.wait(st).unwrap();
            }
        };
        shared.turns.take(false);
        // A panicking job is lost (its picture is finished by its own guard).
        let _ = catch_unwind(AssertUnwindSafe(task));
        shared.turns.give();
        shared.state.lock().unwrap().unfinished -= 1;
        shared.slot.notify_all();
    }
}

impl Drop for JobQueue {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.work.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}
