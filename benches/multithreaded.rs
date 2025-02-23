use deferred_slotmap::{
    ConcurrentSlotMap, Handle, new::DeferredSlotMap, new::Handle as DeferredHandle,
};
use divan::{Bencher, black_box};
use rand::prelude::*;
use std::sync::{Arc, RwLock};

type LocklessSlotMap<T> = rust_lockless_slotmap::LocklessSlotmap<T, parking_lot::RawRwLock>;

// to rename it in bench
pub struct EpochSlotMap<T>(concurrent_slotmap::SlotMap<T>);

fn main() {
    divan::main();
}

const LENS: &[usize] = &[1_000, 10_000, 20_000];
const NUM_THREADS: usize = 4;

fn max_threads() -> usize {
    std::thread::available_parallelism().unwrap().get()
}

fn map<M: Map<u32>>(len: usize) -> M {
    let size = len * max_threads();
    M::new(size, size)
}

// Using our own threading instead of using divans built in threading,
// because divan only offers bencher.with_inputs which gives each thread it's own map
// instead of all threads sharing one map. And creating **one** map for all samples
// of a benchmark is also not the right solution for benches for `insert` for example.
#[divan::bench(types = [LocklessSlotMap<u32>, EpochSlotMap<u32>, DeferredSlotMap<u32>, ConcurrentSlotMap<u32>, LockedSlotMap<u32>], args = LENS)]
fn get_sequential<M: Map<u32>>(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| {
            let mut map: M = map(len);
            let pin = map.pin();
            let handles: Vec<_> = (0..len).map(|i| map.insert(i as u32, &pin)).collect();
            map.flush_deferred();
            (map, handles)
        })
        .bench_values(|(map, handles)| {
            std::thread::scope(|s| {
                for _ in 0..NUM_THREADS {
                    s.spawn(|| {
                        let pin = map.pin();
                        for handle in &handles {
                            black_box(map.get(black_box(*handle), &pin));
                        }
                    });
                }
            });
        });
}

#[divan::bench(types = [LocklessSlotMap<u32>, EpochSlotMap<u32>, DeferredSlotMap<u32>, ConcurrentSlotMap<u32>, LockedSlotMap<u32>], args = LENS)]
fn get_random<M: Map<u32>>(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| {
            let mut map: M = map(len);
            let pin = map.pin();
            let mut handles: Vec<_> = (0..len).map(|i| map.insert(i as u32, &pin)).collect();
            handles.shuffle(&mut rand::rng());
            map.flush_deferred();
            (map, handles)
        })
        .bench_values(|(map, handles)| {
            std::thread::scope(|s| {
                for _ in 0..NUM_THREADS {
                    s.spawn(|| {
                        let pin = map.pin();
                        for handle in &handles {
                            black_box(map.get(black_box(*handle), &pin));
                        }
                    });
                }
            });
        });
}

#[divan::bench(types = [LocklessSlotMap<u32>, EpochSlotMap<u32>, DeferredSlotMap<u32>, ConcurrentSlotMap<u32>, LockedSlotMap<u32>], args = LENS)]
fn insert<M: Map<u32>>(bencher: Bencher, len: usize) {
    bencher.with_inputs(|| map(len)).bench_values(|mut map: M| {
        std::thread::scope(|s| {
            for _ in 0..NUM_THREADS {
                s.spawn(|| {
                    let pin = map.pin();
                    for i in 0..len {
                        black_box(map.insert(black_box(i as u32), &pin));
                    }
                });
            }
        });
        map.flush_deferred();
    });
}

#[divan::bench(types = [LocklessSlotMap<u32>, EpochSlotMap<u32>, DeferredSlotMap<u32>, ConcurrentSlotMap<u32>, LockedSlotMap<u32>], args = LENS)]
fn remove<M: Map<u32>>(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| {
            let map: M = map(len);
            let pin = map.pin();
            let handles: Vec<_> = (0..len).map(|i| map.insert(i as u32, &pin)).collect();
            (map, handles)
        })
        .bench_values(|(mut map, handles)| {
            std::thread::scope(|s| {
                for _ in 0..NUM_THREADS {
                    s.spawn(|| {
                        let pin = map.pin();
                        for handle in &handles {
                            black_box(map.remove(black_box(*handle), &pin));
                        }
                    });
                }
            });
            map.flush_deferred();
        });
}

#[divan::bench(types = [LocklessSlotMap<u32>, EpochSlotMap<u32>, DeferredSlotMap<u32>, ConcurrentSlotMap<u32>, LockedSlotMap<u32>], args = LENS)]
fn mixed_operations<M: Map<u32>>(bencher: Bencher, len: usize) {
    #[derive(Clone, Copy)]
    enum Operation<H> {
        Insert(u32),
        Get(H),
        Remove(H),
    }

    bencher
        .with_inputs(|| {
            let map: M = map(len);

            let mut rng = rand::rng();
            let pin = map.pin();
            let initial_handles: Vec<_> =
                (0..len / 2).map(|i| map.insert(i as u32, &pin)).collect();

            // Pre-generate operations for each thread
            let thread_ops: Vec<Vec<Operation<M::Handle>>> = (0..NUM_THREADS)
                .map(|_| {
                    let mut ops = Vec::with_capacity(len);
                    for _ in 0..len {
                        let op = match rng.random_range(0..3) {
                            0 => Operation::Insert(rng.random()),
                            1 => Operation::Get(*initial_handles.choose(&mut rng).unwrap()),
                            2 => Operation::Remove(*initial_handles.choose(&mut rng).unwrap()),
                            _ => unreachable!(),
                        };
                        ops.push(op);
                    }
                    ops
                })
                .collect();

            (map, thread_ops)
        })
        .bench_values(|(map, thread_ops)| {
            std::thread::scope(|s| {
                for ops in thread_ops {
                    let map = &map;
                    s.spawn(move || {
                        let pin = map.pin();
                        for op in ops {
                            match op {
                                Operation::Insert(value) => {
                                    black_box(map.insert(black_box(value), &pin));
                                }
                                Operation::Get(handle) => {
                                    black_box(map.get(black_box(handle), &pin));
                                }
                                Operation::Remove(handle) => {
                                    black_box(map.remove(black_box(handle), &pin));
                                }
                            }
                        }
                    });
                }
            });
        });
}

// Wrapper struct for SlotMap with RwLock
struct LockedSlotMap<T> {
    inner: Arc<RwLock<slotmap::SlotMap<slotmap::DefaultKey, T>>>,
}

impl<T> LockedSlotMap<T> {
    fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(slotmap::SlotMap::with_key())),
        }
    }
}

trait Map<T>: Send + Sync {
    type Handle: Send + Sync + Copy;
    type Pin;

    fn new(len: usize, frees: usize) -> Self
    where
        Self: Sized;
    fn insert(&self, value: T, pin: &Self::Pin) -> Self::Handle;
    fn get(&self, handle: Self::Handle, pin: &Self::Pin) -> Option<T>
    where
        T: Copy;
    fn pin(&self) -> Self::Pin;
    fn remove(&self, handle: Self::Handle, pin: &Self::Pin) -> bool;
    fn flush_deferred(&mut self) {}
}

impl<T: Send + Sync> Map<T> for ConcurrentSlotMap<T> {
    type Handle = Handle<T>;
    type Pin = ();

    fn new(len: usize, frees: usize) -> Self {
        ConcurrentSlotMap::new(len, frees)
    }

    fn insert(&self, value: T, _: &Self::Pin) -> Self::Handle {
        self.insert(value).unwrap()
    }

    fn get(&self, handle: Self::Handle, _: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.get(handle).copied()
    }

    fn pin(&self) -> Self::Pin {}

    fn remove(&self, handle: Self::Handle, _: &Self::Pin) -> bool {
        self.free(handle).is_ok()
    }

    fn flush_deferred(&mut self) {
        self.flush();
    }
}

impl<T: Send + Sync> Map<T> for DeferredSlotMap<T> {
    type Handle = DeferredHandle<T>;
    type Pin = ();

    fn new(len: usize, frees: usize) -> Self {
        DeferredSlotMap::new(len, frees)
    }

    fn insert(&self, value: T, _: &Self::Pin) -> Self::Handle {
        self.insert(value).unwrap()
    }

    fn get(&self, handle: Self::Handle, _: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.get(handle).copied()
    }

    fn pin(&self) -> Self::Pin {}

    fn remove(&self, handle: Self::Handle, _: &Self::Pin) -> bool {
        self.free(handle).is_ok()
    }

    fn flush_deferred(&mut self) {
        self.flush();
    }
}

impl<T: Send + Sync> Map<T> for LockedSlotMap<T> {
    type Handle = slotmap::DefaultKey;
    type Pin = ();

    fn new(_len: usize, _frees: usize) -> Self {
        LockedSlotMap::new()
    }

    fn insert(&self, value: T, _: &Self::Pin) -> Self::Handle {
        self.inner.write().unwrap().insert(value)
    }

    fn get(&self, handle: Self::Handle, _: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.inner.read().unwrap().get(black_box(handle)).copied()
    }

    fn pin(&self) -> Self::Pin {}

    fn remove(&self, handle: Self::Handle, _: &Self::Pin) -> bool {
        self.inner.write().unwrap().remove(handle).is_some()
    }
}

impl<T: Send + Sync> Map<T> for LocklessSlotMap<T> {
    type Handle = rust_lockless_slotmap::SlotmapTicket;
    type Pin = ();

    fn new(len: usize, _: usize) -> Self {
        LocklessSlotMap::with_capacity(len.min(rust_lockless_slotmap::MAX_ELEMENTS_PER_BLOCK))
    }

    fn insert(&self, value: T, _: &Self::Pin) -> Self::Handle {
        self.insert(value)
    }

    fn get(&self, handle: Self::Handle, _: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.get(handle).map(|i| *i)
    }

    fn pin(&self) -> Self::Pin {}

    fn remove(&self, handle: Self::Handle, _: &Self::Pin) -> bool {
        self.erase(handle).is_some()
    }

    fn flush_deferred(&mut self) {}
}

impl<T: Send + Sync> Map<T> for EpochSlotMap<T> {
    type Handle = concurrent_slotmap::SlotId;
    type Pin = concurrent_slotmap::epoch::UniqueLocalHandle;

    fn new(len: usize, _: usize) -> Self {
        EpochSlotMap(concurrent_slotmap::SlotMap::new(len as u32))
    }

    fn insert(&self, value: T, pin: &Self::Pin) -> Self::Handle {
        self.0.insert(value, pin.pin())
    }

    fn get(&self, handle: Self::Handle, pin: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.0.get(handle, pin.pin()).map(|i| *i)
    }

    fn pin(&self) -> Self::Pin {
        self.0.global().register_local()
    }

    fn remove(&self, handle: Self::Handle, pin: &Self::Pin) -> bool {
        self.0.remove(handle, pin.pin()).is_some()
    }

    fn flush_deferred(&mut self) {}
}
