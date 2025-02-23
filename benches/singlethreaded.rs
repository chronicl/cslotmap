use cslotmap::{ConcurrentSlotMap, SlotHandle};
use divan::{Bencher, black_box};
use rand::seq::SliceRandom;

const LENS: &[usize] = &[1_000, 10_000, 20_000];

fn main() {
    divan::main();
}

#[divan::bench(types = [slotmap::SlotMap<slotmap::DefaultKey, u32>, ConcurrentSlotMap<u32>, ConcurrentSlotMapWithConcurrentOperations<u32>], threads = [1, 4, 16], args = LENS)]
fn get_sequential<M: Map<u32>>(bencher: Bencher, len: usize) {
    let mut map = M::new();
    let handles: Vec<_> = (0..len).map(|i| map.insert(i as u32)).collect();

    bencher.bench(|| {
        for handle in handles.iter() {
            black_box(map.get(black_box(*handle)));
        }
    })
}

#[divan::bench(types = [slotmap::SlotMap<slotmap::DefaultKey, u32>, ConcurrentSlotMap<u32>, ConcurrentSlotMapWithConcurrentOperations<u32>], threads = [1, 4, 16], args = LENS)]
fn get_random<M: Map<u32>>(bencher: Bencher, len: usize) {
    let mut map = M::new();
    let mut handles: Vec<_> = (0..len).map(|i| map.insert(i as u32)).collect();
    handles.shuffle(&mut rand::rng());

    bencher.bench(|| {
        for handle in handles.iter() {
            black_box(map.get(black_box(*handle)));
        }
    })
}

#[divan::bench(types = [slotmap::SlotMap<slotmap::DefaultKey, u32>, ConcurrentSlotMap<u32>, ConcurrentSlotMapWithConcurrentOperations<u32>], args = LENS)]
fn insert<M: Map<u32>>(bencher: Bencher, len: usize) {
    bencher.with_inputs(|| M::new()).bench_values(|mut map| {
        for i in 0..len {
            black_box(map.insert(black_box(i as u32)));
        }
    })
}

#[divan::bench(types = [slotmap::SlotMap<slotmap::DefaultKey, u32>, ConcurrentSlotMap<u32>, ConcurrentSlotMapWithConcurrentOperations<u32>], args = LENS)]
fn remove<M: Map<u32>>(bencher: Bencher, len: usize) {
    bencher
        .with_inputs(|| {
            let mut map = M::new();
            let handles: Vec<_> = (0..len).map(|i| map.insert(i as u32)).collect();
            (map, handles)
        })
        .bench_values(|(mut map, handles)| {
            for handle in handles.iter() {
                black_box(map.remove(black_box(*handle)));
            }
        })
}

trait Map<T>: Send + Sync {
    type Handle: Send + Sync + Copy;

    fn new() -> Self
    where
        Self: Sized;
    fn insert(&mut self, value: T) -> Self::Handle;
    fn get(&self, handle: Self::Handle) -> Option<T>
    where
        T: Copy;
    fn remove(&mut self, handle: Self::Handle) -> Option<T>;
}

impl<T: Send + Sync + Copy> Map<T> for ConcurrentSlotMap<T> {
    type Handle = SlotHandle;

    fn new() -> Self {
        ConcurrentSlotMap::new(0, 0)
    }

    fn insert(&mut self, value: T) -> Self::Handle {
        self.insert(value)
    }

    fn get(&self, handle: Self::Handle) -> Option<T> {
        self.get(handle).copied()
    }

    fn remove(&mut self, handle: Self::Handle) -> Option<T> {
        self.remove(handle)
    }
}

struct ConcurrentSlotMapWithConcurrentOperations<T>(ConcurrentSlotMap<T>);
impl<T: Send + Sync + Copy> Map<T> for ConcurrentSlotMapWithConcurrentOperations<T> {
    type Handle = SlotHandle;

    fn new() -> Self {
        ConcurrentSlotMapWithConcurrentOperations(ConcurrentSlotMap::new(20_000, 20_000))
    }

    fn insert(&mut self, value: T) -> Self::Handle {
        self.0.concurrent_insert(value).unwrap()
    }

    fn get(&self, handle: Self::Handle) -> Option<T> {
        self.0.get(handle).copied()
    }

    fn remove(&mut self, handle: Self::Handle) -> Option<T> {
        self.0.deferred_remove(handle).unwrap();
        None
    }
}

impl<T: Send + Sync + Copy> Map<T> for slotmap::SlotMap<slotmap::DefaultKey, T> {
    type Handle = slotmap::DefaultKey;

    fn new() -> Self {
        slotmap::SlotMap::with_key()
    }

    fn insert(&mut self, value: T) -> Self::Handle {
        self.insert(value)
    }

    fn get(&self, handle: Self::Handle) -> Option<T> {
        self.get(handle).copied()
    }

    fn remove(&mut self, handle: Self::Handle) -> Option<T> {
        self.remove(handle)
    }
}
