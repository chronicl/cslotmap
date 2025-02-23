// Testing slightly different versions of BumpVec.
// Performance was similar (ignoring BumpVec3 because it's unsafe), but for single threaded write BumpVec1
// is faster so that's what is used in this crate.
fn main() {
    divan::main();
}

const THREADS: &[usize] = &[0, 1, 4, 16];

const LENS: &[usize] = &[1_000, 20_000, 100_000];

#[divan::bench_group]
mod single_threaded {
    use divan::Bencher;
    use std::hint::black_box;

    use crate::{BumpVec1, BumpVec2, BumpVec3, LENS, VecTrait};

    #[divan::bench(types = [Vec<u32>, BumpVec1<u32>, BumpVec2<u32>, BumpVec3<u32>], args = LENS)]
    fn read<T: VecTrait<u32> + Sync>(bencher: Bencher, len: usize) {
        let v = T::new(len);
        bencher.bench(move || {
            for i in 0..len {
                black_box(v.read(black_box(i)));
            }
        });
    }

    #[divan::bench(types = [Vec<u32>, BumpVec1<u32>, BumpVec2<u32>, BumpVec3<u32>], args = LENS)]
    fn write<T: VecTrait<u32> + Sync>(bencher: Bencher, len: usize) {
        bencher.with_inputs(|| T::new(len)).bench_values(|mut v| {
            for i in 0..len {
                v.write(black_box(i), black_box(10));
            }
        });
    }
}

#[divan::bench_group(threads = THREADS)]
mod multi_threaded {
    use divan::Bencher;
    use std::{hint::black_box, sync::RwLock};

    use crate::{BumpVec1, BumpVec2, BumpVec3, ConcurrentVecTrait, LENS};

    #[divan::bench(types = [RwLock<Vec<u32>>, BumpVec1<u32>, BumpVec2<u32>, BumpVec3<u32>], args = LENS)]
    fn read<T: ConcurrentVecTrait<u32> + Sync>(bencher: Bencher, len: usize) {
        let v = T::new(len);
        bencher.bench(|| {
            for i in 0..len {
                black_box(v.read(black_box(i)));
            }
        });
    }

    #[divan::bench(types = [RwLock<Vec<u32>>, BumpVec1<u32>, BumpVec2<u32>, BumpVec3<u32>], args = LENS)]
    fn write<T: ConcurrentVecTrait<u32> + Sync>(bencher: Bencher, len: usize) {
        let v = T::new(len);
        bencher.bench(|| {
            for i in 0..len {
                v.write(black_box(i), black_box(10));
            }
        });
    }
}

mod atomic_protection {
    use std::{
        hint::black_box,
        sync::{Mutex, atomic::AtomicBool},
    };

    use divan::Bencher;

    use crate::{AtomicProtected, MutexProtected, Unprotected};

    const LENS: &[usize] = &[1_000, 20_000, 400_000];

    #[divan::bench(args = LENS)]
    fn mutex_protected(bencher: Bencher, len: usize) {
        let c = MutexProtected {
            v: Mutex::new(vec![0; len]),
        };
        bencher.bench(|| {
            for i in 0..len {
                c.read(black_box(i));
            }
        });
    }

    #[divan::bench(args = LENS)]
    fn protected(bencher: Bencher, len: usize) {
        let c = AtomicProtected {
            v: vec![0; len],
            atomic: AtomicBool::new(false),
        };
        bencher.bench(|| {
            for i in 0..len {
                c.read(black_box(i));
            }
        });
    }

    #[divan::bench(args = LENS)]
    fn unprotected(bencher: Bencher, len: usize) {
        let c = Unprotected { v: vec![0; len] };
        bencher.bench(|| {
            for i in 0..len {
                c.read(black_box(i));
            }
        });
    }
}

pub struct MutexProtected {
    v: Mutex<Vec<u32>>,
}

impl MutexProtected {
    fn read(&self, i: usize) -> u32 {
        self.v.lock().unwrap()[i]
    }
}

pub struct AtomicProtected {
    v: Vec<u32>,
    atomic: AtomicBool,
}

impl AtomicProtected {
    fn read(&self, i: usize) -> &u32 {
        self.atomic.load(Ordering::Acquire);
        &self.v[i]
    }
}

pub struct Unprotected {
    v: Vec<u32>,
}

impl Unprotected {
    fn read(&self, i: usize) -> &u32 {
        &self.v[i]
    }
}

use std::sync::{Mutex, RwLock};
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};

pub struct BumpVec1<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
    guards: Box<[AtomicBool]>,
    allocated: AtomicUsize,
}

impl<T> BumpVec1<T> {
    pub fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        let mut guards = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(UnsafeCell::new(MaybeUninit::uninit()));
            guards.push(AtomicBool::new(false));
        }
        Self {
            data: data.into_boxed_slice(),
            guards: guards.into_boxed_slice(),
            allocated: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, value: T) -> Option<usize> {
        let index = self.allocated.fetch_add(1, Ordering::Relaxed);
        if index >= self.data.len() {
            return None;
        }

        unsafe {
            (*self.data[index].get()).as_mut_ptr().write(value);
        }
        self.guards[index].store(true, Ordering::Release);

        Some(index)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        let allocated = self.allocated.load(Ordering::Relaxed);

        if index >= allocated || index >= self.data.len() {
            return None;
        }

        if self.guards[index].load(Ordering::Acquire) {
            Some(unsafe { (*self.data[index].get()).assume_init_ref() })
        } else {
            None
        }
    }
}

unsafe impl<T: Send> Sync for BumpVec1<T> {}

impl<T> Drop for BumpVec1<T> {
    fn drop(&mut self) {
        let len = self.allocated.load(Ordering::Relaxed);
        for i in 0..len.min(self.data.len()) {
            // Sync 1
            // This should always be the case. The only time the guard isn't true for i < len.min(self.data.len())
            // is in the push method, which takes &self, but here we have &mut self, so push must not be running at the same time
            assert!(self.guards[i].load(Ordering::Acquire));
            let data = self.data[i].get_mut();
            unsafe {
                (*data).assume_init_drop();
            }
        }
    }
}

pub struct BumpVec2<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
    guards: Box<[AtomicU64]>,
    allocated: AtomicUsize,
}

impl<T> BumpVec2<T> {
    pub fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        let num_guard_words = capacity.div_ceil(64);
        let mut guards = Vec::with_capacity(num_guard_words);
        for _ in 0..capacity {
            data.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        for _ in 0..num_guard_words {
            guards.push(AtomicU64::new(0));
        }
        Self {
            data: data.into_boxed_slice(),
            guards: guards.into_boxed_slice(),
            allocated: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, value: T) -> Option<usize> {
        let index = self.allocated.fetch_add(1, Ordering::Relaxed);
        if index >= self.data.len() {
            return None;
        }

        unsafe {
            (*self.data[index].get()).as_mut_ptr().write(value);
        }

        let (guard_word, guard_bit) = Self::guard_access(index);
        // Sync 1: synchronize above write
        self.guards[guard_word].fetch_or(guard_bit, Ordering::Release);

        Some(index)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        let allocated = self.allocated.load(Ordering::Relaxed);

        if index >= allocated || index >= self.data.len() {
            return None;
        }

        let (guard_word, guard_bit) = Self::guard_access(index);
        // Sync 1: synchronize below read with write in push
        if (self.guards[guard_word].load(Ordering::Acquire) & guard_bit) != 0 {
            Some(unsafe { (*self.data[index].get()).assume_init_ref() })
        } else {
            None
        }
    }

    fn guard_access(index: usize) -> (usize, u64) {
        let guard_word = index / 64;
        let guard_bit = 1u64 << (index % 64);
        (guard_word, guard_bit)
    }
}

unsafe impl<T: Send + Sync> Sync for BumpVec2<T> {}

impl<T> Drop for BumpVec2<T> {
    fn drop(&mut self) {
        let len = self.allocated.load(Ordering::Relaxed);
        for i in 0..len.min(self.data.len()) {
            let (guard_word, guard_bit) = Self::guard_access(i);

            // Sync 1
            // This should always be the case. The only time the guard isn't true for i < len.min(self.data.len())
            // is in the push method, which takes &self, but here we have &mut self, so push must not be running at the same time
            assert!((self.guards[guard_word].load(Ordering::Acquire) & guard_bit) != 0);
            let data = self.data[i].get_mut();
            unsafe {
                (*data).assume_init_drop();
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Handle {
    index: usize,
}

// WARNING: This version of BumpVec is not safe, because it may write and read the same value at the same time.
pub struct BumpVec3<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// This is the amount of elements allocated if less than data.len(), otherwise there is data.len() elements allocated.
    allocated: AtomicUsize,
    fence: AtomicBool,
}

impl<T> BumpVec3<T> {
    pub fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(UnsafeCell::new(MaybeUninit::uninit()));
        }
        Self {
            data: data.into_boxed_slice(),
            allocated: AtomicUsize::new(0),
            fence: AtomicBool::new(false),
        }
    }

    // The Handle is essentially just an index, but by restricting access to the vec via the handle we can guarantee
    // that the value has been written before being accessed.
    pub fn push(&self, value: T) -> Option<Handle> {
        let index = self.allocated.fetch_add(1, Ordering::Relaxed);
        if index >= self.data.len() {
            return None;
        }

        unsafe {
            (*self.data[index].get()).as_mut_ptr().write(value);
        }
        // Sync 1: synchronizing above write
        self.fence.store(true, Ordering::Release);

        Some(Handle { index })
    }

    /// # SAFETY
    /// The handle must have been obtained from the same `BumpVec`.
    pub unsafe fn get(&self, handle: Handle) -> Option<&T> {
        let index = handle.index;
        let allocated = self.allocated.load(Ordering::Relaxed);

        if index >= allocated || index >= self.data.len() {
            return None;
        }

        // Sync 1: synchronizing write in `push` with below read
        self.fence.load(Ordering::Acquire);
        Some(unsafe { (*self.data[index].get()).assume_init_ref() })
    }
}

unsafe impl<T: Send + Sync> Sync for BumpVec3<T> {}

impl<T> Drop for BumpVec3<T> {
    fn drop(&mut self) {
        let len = self.allocated.load(Ordering::Relaxed);
        // Sync 1: synchronizing write in `push`
        self.fence.load(Ordering::Acquire);
        for i in 0..len.min(self.data.len()) {
            let data = self.data[i].get_mut();
            // SAFETY:
            // len.min(self.data.len()) is guaranteed to be the length of the initialized data.
            // The only time where this is not the case is in `push`, but that method takes a reference of
            // the BumpVec, so doesn't drop it.
            unsafe {
                (*data).assume_init_drop();
            }
        }
    }
}

trait VecTrait<T> {
    fn new(size: usize) -> Self;
    fn read(&self, i: usize) -> Option<&T>;
    fn write(&mut self, i: usize, value: T);
}

trait ConcurrentVecTrait<T> {
    fn new(size: usize) -> Self;
    fn read(&self, i: usize) -> Option<T>;
    fn write(&self, i: usize, value: T);
}

impl<T: Default + Clone> VecTrait<T> for Vec<T> {
    fn new(size: usize) -> Self {
        vec![Default::default(); size]
    }

    fn read(&self, i: usize) -> Option<&T> {
        self.get(i)
    }

    fn write(&mut self, i: usize, value: T) {
        self[i] = value;
    }
}

impl<T> VecTrait<T> for BumpVec1<T> {
    fn new(size: usize) -> Self {
        BumpVec1::<T>::new(size)
    }

    fn read(&self, i: usize) -> Option<&T> {
        self.get(i)
    }

    fn write(&mut self, _: usize, value: T) {
        let _ = self.push(value);
    }
}

impl<T> VecTrait<T> for BumpVec2<T> {
    fn new(size: usize) -> Self {
        BumpVec2::<T>::new(size)
    }

    fn read(&self, i: usize) -> Option<&T> {
        self.get(i)
    }

    fn write(&mut self, _: usize, value: T) {
        let _ = self.push(value);
    }
}

impl<T> VecTrait<T> for BumpVec3<T> {
    fn new(size: usize) -> Self {
        BumpVec3::<T>::new(size)
    }

    fn read(&self, i: usize) -> Option<&T> {
        // SAFETY: Using index i as handle is safe since access pattern matches other implementations
        unsafe { self.get(Handle { index: i }) }
    }

    fn write(&mut self, _: usize, value: T) {
        let _ = self.push(value);
    }
}

impl<T: Copy> ConcurrentVecTrait<T> for BumpVec1<T> {
    fn new(size: usize) -> Self {
        BumpVec1::<T>::new(size)
    }

    fn read(&self, i: usize) -> Option<T> {
        self.get(i).copied()
    }

    fn write(&self, _: usize, value: T) {
        let _ = self.push(value);
    }
}

impl<T: Copy> ConcurrentVecTrait<T> for BumpVec2<T> {
    fn new(size: usize) -> Self {
        BumpVec2::<T>::new(size)
    }

    fn read(&self, i: usize) -> Option<T> {
        self.get(i).copied()
    }

    fn write(&self, _: usize, value: T) {
        let _ = self.push(value);
    }
}

impl<T: Copy> ConcurrentVecTrait<T> for BumpVec3<T> {
    fn new(size: usize) -> Self {
        BumpVec3::<T>::new(size)
    }

    fn read(&self, i: usize) -> Option<T> {
        // SAFETY: Using index i as handle is safe since access pattern matches other implementations
        unsafe { self.get(Handle { index: i }).copied() }
    }

    fn write(&self, _: usize, value: T) {
        let _ = self.push(value);
    }
}

impl<T: Default + Copy> ConcurrentVecTrait<T> for RwLock<Vec<T>> {
    fn new(size: usize) -> Self {
        RwLock::new(vec![Default::default(); size])
    }

    fn read(&self, i: usize) -> Option<T> {
        self.read().unwrap().get(i).copied()
    }

    fn write(&self, i: usize, value: T) {
        self.write().unwrap()[i] = value;
    }
}
