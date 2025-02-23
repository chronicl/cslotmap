use std::{any::type_name, fmt::Display, hint::black_box, sync::Arc, time::Duration};

use cslotmap::{ConcurrentSlotMap, SlotHandle};
use parking_lot::RwLock;
use rand::{
    Rng,
    seq::{IndexedRandom, SliceRandom},
};

const SAMPLE_COUNT: u32 = 100;
const THREADS: &[u32] = &[1, 4, 16];
const MAX_THREADS: u32 = 32;
const LENS: &[usize] = &[1000, 5_000, 20_000];

type M1 = ConcurrentSlotMap<u32>;
type M2 = LocklessSlotMap<u32>;
type M3 = EpochSlotMap<u32>;
type M4 = LockedSlotMap<u32>;

// Rolling my own benchmark system, because divan's support for multi-threading is not enough -
// particularly it's not possible to create one map for all threads to be used simultaneously without
// using the same map for all samples.
//
// This isn't pretty, but works...
fn main() {
    Report::print_header();

    macro_rules! bench {
        ($name:ident, $new:ident, $method:ident, $thread_count:ident, $len:ident, $($t:ty),*) => {
            $(
                bench::<$t, _>($name, SAMPLE_COUNT, $thread_count, $new::<$t>($len), $method::<$t>($len)).print();
            )*
        };
    }

    // get sequential
    for &len in LENS {
        for &thread_count in THREADS {
            let name = &format!("get seq {}", len);
            bench!(name, new_filled, get, thread_count, len, M1, M2, M3, M4);
        }
        println!()
    }

    // get random
    for &len in LENS {
        for &thread_count in THREADS {
            let name = &format!("get rand {}", len);
            bench!(name, new_shuffled, get, thread_count, len, M1, M2, M3, M4);
        }
        println!()
    }

    // insert
    for &len in LENS {
        for &thread_count in THREADS {
            let name = &format!("insert {}", len);
            bench!(name, new_insert, insert, thread_count, len, M1, M2, M3, M4);
        }
        println!()
    }

    // TODO: this removes a lot of the same elements (each thread removes the same)
    // remove
    for &len in LENS {
        for &thread_count in THREADS {
            let name = &format!("remove {}", len);
            bench!(name, new_filled, remove, thread_count, len, M1, M2, M3, M4);
        }
        println!()
    }

    // mixed operations
    for &len in LENS {
        for &thread_count in THREADS {
            let name = &format!("mixed ops {}", len);
            #[rustfmt::skip]
            bench!(name, new_mixed_operations, mixed_operations, thread_count, len, M1, M2, M3, M4);
        }
        println!()
    }
}

fn new_insert<M: Map<u32>>(len: usize) -> impl Fn() -> M {
    move || M::new(len * MAX_THREADS as usize, len * MAX_THREADS as usize)
}

fn new_filled<M: Map<u32>>(len: usize) -> impl Fn() -> (M, Vec<M::Handle>) {
    move || {
        let mut map = M::new(len, len);
        let pin = map.pin();
        let mut handles = Vec::new();
        for i in 0..len as u32 {
            handles.push(map.insert(i, &pin));
        }
        map.flush_deferred();
        (map, handles)
    }
}

fn new_shuffled<M: Map<u32>>(len: usize) -> impl Fn() -> (M, Vec<M::Handle>) {
    move || {
        let (map, mut handles) = new_filled(len)();
        handles.shuffle(&mut rand::rng());
        (map, handles)
    }
}

#[derive(Clone, Copy)]
enum Operation<H> {
    Insert(u32),
    Get(H),
    Remove(H),
}

fn new_mixed_operations<M: Map<u32>>(
    len: usize,
) -> impl Fn() -> (M, Vec<Vec<Operation<M::Handle>>>) {
    move || {
        let map: M = M::new(len * MAX_THREADS as usize, len * MAX_THREADS as usize);

        let mut rng = rand::rng();
        let pin = map.pin();
        let initial_handles: Vec<_> = (0..len / 2).map(|i| map.insert(i as u32, &pin)).collect();

        // Pre-generate operations for each thread
        let thread_ops: Vec<Vec<Operation<M::Handle>>> = (0..32)
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
    }
}

#[allow(clippy::type_complexity)]
fn mixed_operations<M: Map<u32>>(_: usize) -> impl Fn(&(M, Vec<Vec<Operation<M::Handle>>>), u32) {
    move |(map, ops), thread_id| {
        let ops = &ops[thread_id as usize];
        let pin = map.pin();
        for op in ops {
            match op {
                Operation::Insert(value) => {
                    black_box(map.insert(black_box(*value), &pin));
                }
                Operation::Get(handle) => {
                    black_box(map.get(black_box(*handle), &pin));
                }
                Operation::Remove(handle) => {
                    black_box(map.remove(black_box(*handle), &pin));
                }
            }
        }
    }
}

fn get<M: Map<u32>>(_: usize) -> impl Fn(&(M, Vec<M::Handle>), u32) {
    |(map, handles), _| {
        let pin = map.pin();
        for handle in handles {
            black_box(map.get(black_box(*handle), &pin));
        }
    }
}

fn insert<M: Map<u32>>(len: usize) -> impl Fn(&M, u32) {
    move |map, _| {
        let pin = map.pin();
        for i in 0..len {
            black_box(map.insert(black_box(i as u32), &pin));
        }
    }
}

fn remove<M: Map<u32>>(_: usize) -> impl Fn(&(M, Vec<M::Handle>), u32) {
    move |(map, handles), _| {
        let pin = map.pin();
        for handle in handles {
            black_box(map.remove(black_box(*handle), &pin));
        }
    }
}

pub struct Report {
    type_name: String,
    operation_name: String,
    samples: Vec<Duration>,
    thread_count: u32,
}

impl Report {
    pub fn mean(&self) -> Duration {
        let sum: Duration = self.samples.iter().sum();
        sum / self.samples.len() as u32
    }

    pub fn median(&self) -> Duration {
        let mut sorted = self.samples.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    pub fn header() -> String {
        format!(
            "{: <24} | {: <14} | {: >7} | {: >11} | {: >11} | {: >11} | {: >11} | {: >8}",
            "Type", "Operation", "Threads", "Mean", "Median", "Max", "Min", "Samples"
        )
    }

    pub fn print_header() {
        println!("{}", Report::header());
    }

    pub fn print(&self) {
        println!("{}", self);
    }
}

impl Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{: <24} | {: <14} | {: >7} | {: >11.3?} | {: >11.3?} | {: >11.3?} | {: >11.3?} | {: >8}",
            &self.type_name[..24.min(self.type_name.len())],
            self.operation_name,
            self.thread_count,
            self.mean(),
            self.median(),
            self.samples.iter().max().unwrap(),
            self.samples.iter().min().unwrap(),
            self.samples.len()
        )
    }
}

fn bench<T, I: Send + Sync>(
    operation_name: impl ToString,
    sample_count: u32,
    thread_count: u32,
    input: impl Fn() -> I,
    f: impl Fn(&I, u32) + Send + Sync,
) -> Report {
    let parts = type_name::<T>().split("::");
    let mut ty = String::new();
    let mut type_name_started = false;
    for part in parts {
        type_name_started |= part.contains("<");
        if type_name_started {
            ty.push_str(part);
        }
    }

    if ty.is_empty() {
        ty = type_name::<T>().split("::").last().unwrap().into();
    }

    Report {
        type_name: ty,
        operation_name: operation_name.to_string(),
        samples: (0..sample_count)
            .map(|_| one_sample_threaded(thread_count, &input, &f))
            .collect(),
        thread_count,
    }
}

fn one_sample_threaded<I: Send + Sync>(
    thread_count: u32,
    input: impl Fn() -> I,
    f: impl Fn(&I, u32) + Send + Sync,
) -> Duration {
    let i = input();
    let time_spent: Duration = std::thread::scope(|s| {
        let handles: Vec<_> = (0..thread_count)
            .map(|j| {
                let i = &i;
                let f = &f;
                s.spawn(move || {
                    let now = std::time::Instant::now();
                    f(i, j);
                    now.elapsed()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    time_spent / thread_count
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
    type Handle = SlotHandle;
    type Pin = ();

    fn new(len: usize, frees: usize) -> Self {
        ConcurrentSlotMap::new(len, frees)
    }

    fn insert(&self, value: T, _: &Self::Pin) -> Self::Handle {
        self.concurrent_insert(value).unwrap()
    }

    fn get(&self, handle: Self::Handle, _: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.get(handle).copied()
    }

    fn pin(&self) -> Self::Pin {}

    fn remove(&self, handle: Self::Handle, _: &Self::Pin) -> bool {
        self.deferred_remove(handle).is_ok()
    }

    fn flush_deferred(&mut self) {
        self.flush();
    }
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

impl<T: Send + Sync> Map<T> for LockedSlotMap<T> {
    type Handle = slotmap::DefaultKey;
    type Pin = ();

    fn new(_len: usize, _frees: usize) -> Self {
        LockedSlotMap::new()
    }

    fn insert(&self, value: T, _: &Self::Pin) -> Self::Handle {
        self.inner.write().insert(value)
    }

    fn get(&self, handle: Self::Handle, _: &Self::Pin) -> Option<T>
    where
        T: Copy,
    {
        self.inner.read().get(black_box(handle)).copied()
    }

    fn pin(&self) -> Self::Pin {}

    fn remove(&self, handle: Self::Handle, _: &Self::Pin) -> bool {
        self.inner.write().remove(handle).is_some()
    }
}

type LocklessSlotMap<T> = rust_lockless_slotmap::LocklessSlotmap<T, parking_lot::RawRwLock>;
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

// to rename it in bench
pub struct EpochSlotMap<T>(concurrent_slotmap::SlotMap<T>);
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
