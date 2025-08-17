/// # A slot map with support for concurrent lock-less operations.
/// Aside from offering the usual slot map API, this slot map offers two concurrent lock-less
/// operations: [`concurrent_insert`](ConcurrentSlotMap::concurrent_insert) and
/// [`deferred_remove`](ConcurrentSlotMap::deferred_remove).
///
/// Once a concurrent method has been used, the slot map is marked as dirty and other write
/// methods like [`insert`](ConcurrentSlotMap::insert) and
/// [`remove`](ConcurrentSlotMap::remove) become unavailable until
/// [`flush`](ConcurrentSlotMap::flush) is called. Read methods like
/// [`get`](ConcurrentSlotMap::get) and [`iter`](ConcurrentSlotMap::iter) remain available.
///
/// However, there is a configurable limit to how many `concurrent_insert` and
/// `deferred_remove` operations can occur in between each `flush`. The limit must be
/// configured when creating the slot map with [`ConcurrentSlotMap::new`] and can later be
/// changed via [`ConcurrentSlotMap::set_concurrent_insert_capacity`] and
/// [`ConcurrentSlotMap::set_deferred_remove_capacity`].
///
/// ## Performance
/// The non-concurrent methods are comparable in performance to `SlotMap` from the [`slotmap`](https://docs.rs/slotmap/latest/slotmap/) crate.
///
/// In single-threaded usage the concurrent methods are roughly 2 to 4 times slower than their
/// non-concurrent counterparts.
///
/// I have benchmarked against other concurrent slotmap implementations (which have different
/// trade-offs) and this implementation has better performance by 2-20 times across the board.
///
/// These measurements are heavily hardware dependent. Benchmark code can be found in the
/// github repo.
use bumpvec::BumpVec;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub mod bumpvec;

type Version = u32;

/// # A slot map with support for concurrent lock-less operations.
/// Aside from offering the usual slot map API, this slot map offers two concurrent lock-less
/// operations: [`concurrent_insert`](ConcurrentSlotMap::concurrent_insert) and
/// [`deferred_remove`](ConcurrentSlotMap::deferred_remove).
///
///
/// Once a concurrent method has been used, the slot map is marked as dirty and other write methods
/// like [`insert`](ConcurrentSlotMap::insert) and [`remove`](ConcurrentSlotMap::remove) become
/// unavailable until [`flush`](ConcurrentSlotMap::flush) is called. Read methods like
/// [`get`](ConcurrentSlotMap::get) and [`iter`](ConcurrentSlotMap::iter) remain available.
///
/// However, there is a configurable limit to how many `concurrent_insert` and `deferred_remove`
/// operations can occur in between each `flush`. The limit must be configured when creating the
/// slot map with [`ConcurrentSlotMap::new`] and can later be changed via
/// [`ConcurrentSlotMap::set_concurrent_insert_capacity`] and
/// [`ConcurrentSlotMap::set_deferred_remove_capacity`].
///
/// ## Performance
/// The non-concurrent methods are comparable in performance to `SlotMap` from the [`slotmap`](https://docs.rs/slotmap/latest/slotmap/) crate.
///
/// In single-threaded usage the concurrent methods are roughly 2 to 4 times slower than their
/// non-concurrent counterparts.
///
/// I have benchmarked against other concurrent slotmap implementations (which have different
/// trade-offs) and this implementation has better performance by 2-20 times across the board.
///
/// These measurements are heavily hardware dependent. Benchmark code can be found in the github
/// repo.
///
/// ### Safety
///
/// This is meant as a quick overview of how safety is achieved:
///
/// While calling the concurrent operations, non-concurrent **write** methods can't be used, because
/// the concurrent methods operate on a shared reference (&self) and the non-concurrent write methods
/// operate on an exclusive reference (&mut self).
///
/// While calling the concurrent operations, non-concurrent **read** methods can be used, because
/// both operate on a shared reference (&self). The methods are safe because
/// - `deferred_remove`s are stored in a list that the non-concurrent read methods don't access.
///   The removals are applied on the next `flush(&mut self)` call, at which point they become observable
///   by the non-concurrent read methods.
/// - `concurrent_insert` writes to the same slots the non-concurrent read methods read from.
///   Each slot is secured by an `AtomicBool`. Non-concurrent read methods on a slot only succeed
///   if the `AtomicBool` is true, at which point the slot is guaranteed to contain data.
///   Generation checks happen non-atomically after the atomic occupied check.
#[derive(Debug)]
pub struct ConcurrentSlotMap<T> {
    slots: Vec<Slot<T>>,

    free: Vec<u32>,
    frees_used: AtomicUsize,

    deferred_slots: BumpVec<T>,
    deferred_frees: BumpVec<u32>,
    dirty: AtomicBool,
}

// Concurrent API
impl<T> ConcurrentSlotMap<T> {
    /// Inserts a new item into the map. The item is immediately accessible by all threads.
    pub fn concurrent_insert(&self, value: T) -> Result<SlotHandle, OutOfSpace> {
        self.dirty.store(true, Ordering::Relaxed);

        let (index, version) = if let Some(index) = self.reserve_free_slot() {
            // SAFETY: index was reserved via `reserve_free_slot`. self is marked as dirty in this
            // method.
            let generation = unsafe { self.write_to_reserved_slot(index, value) };
            (index, generation)
        } else if let Some(index) = self.deferred_slots.push(value) {
            // deferred_slots will be appeneded to slots upon calling flush, so the handle index is
            // actually self.slots.len() + index
            ((self.slots.len() + index) as u32, 0)
        } else {
            return Err(OutOfSpace);
        };

        Ok(SlotHandle::new(index, version))
    }

    /// Removals are deferred until the next `flush` call.
    /// The removed items can be obtained by calling `flush_with` instead of `flush`.
    pub fn deferred_remove(&self, handle: SlotHandle) -> Result<(), OutOfSpace> {
        self.dirty.store(true, Ordering::Relaxed);

        if self.handle_is_valid(handle) && self.deferred_frees.push(handle.index).is_none() {
            return Err(OutOfSpace);
        }

        Ok(())
    }

    pub fn flush(&mut self) {
        self.flush_with(|_, _| {}, |_, _| {}, |_| {});
    }

    pub fn flush_with<FR>(
        &mut self,
        mut reused_fn: impl FnMut(SlotHandle, &T),
        mut new_fn: impl FnMut(SlotHandle, &T),
        free_fn: impl FnOnce(&mut dyn Iterator<Item = (usize, T)>) -> FR,
    ) -> FR {
        self.dirty.store(false, Ordering::Relaxed);

        let Self {
            slots,
            free,
            frees_used,
            deferred_slots,
            deferred_frees,
            ..
        } = self;

        // Removing reused frees from `free` and updating the non-atomic
        let used = frees_used.load(Ordering::Relaxed);
        frees_used.store(0, Ordering::Relaxed);
        for _ in 0..used.min(free.len()) {
            let index = free.pop().unwrap();
            let slot = &mut slots[index as usize];
            // SAFETY: We just popped it from `free`, which means it was reused/occupied since last
            // flush.
            unsafe { slot.confirm_write() };
            let handle = SlotHandle::new(index, slot.version());
            reused_fn(handle, slot.get_unchecked())
        }

        // Move from deferred slots to slots to make space for future shared insertions
        deferred_slots.clear(|values| {
            for value in values {
                let index = slots.len() as u32;
                slots.push(Slot::new_occupied(value));
                let slot = &slots[index as usize];
                let handle = SlotHandle::new(index, slot.version());
                new_fn(handle, slot.get_unchecked());
            }
        });

        // Appying deferred frees
        deferred_frees.clear(|indices| {
            let mut values = indices.flat_map(|index| {
                let value = slots[index as usize].free();
                // Only freeing if the slot wasn't already free
                if value.is_some() {
                    free.push(index);
                }
                value.map(|v| (index as usize, v))
            });

            let r = free_fn(&mut values);

            values.count();

            r
        })
    }

    fn reserve_free_slot(&self) -> Option<u32> {
        let frees_used = self.frees_used.fetch_add(1, Ordering::Relaxed);
        let len = self.free.len();
        (frees_used < len).then(|| self.free[len - 1 - frees_used])
    }

    /// Returns the generation of the slot.
    /// ## SAFETY
    /// index must have previously been reserved via `reserve_free_slot`, so that no other
    /// shared write method can possibly use the same free slot.
    /// self must be marked as dirty, so that no exclusive write method can possibly use the free
    /// slot, because exclusive write methods can't be used as long as self is dirty.
    unsafe fn write_to_reserved_slot(&self, index: u32, value: T) -> Version {
        let slot = &self.slots[index as usize];
        assert!(slot.is_free());

        // SAFETY: See above.
        // Reads are secured by slot.occupied_atomic
        // TODO: maybe have to free AtomicU32 here
        unsafe { slot.write(value) };

        slot.version()
    }
}

// Standard slot map API
impl<T> ConcurrentSlotMap<T> {
    /// Create a new map. `concurrent_insert_capacity` and `deferred_remove_capacity` determine how
    /// many times `concurrent_insert` and `deferred_remove` without returning `OutOfMemory` in
    /// between `flush`es.
    pub fn new(concurrent_insert_capacity: usize, deferred_remove_capacity: usize) -> Self {
        Self {
            slots: Default::default(),

            free: Default::default(),
            frees_used: AtomicUsize::new(0),

            deferred_slots: BumpVec::new(concurrent_insert_capacity),
            deferred_frees: BumpVec::new(deferred_remove_capacity),
            dirty: AtomicBool::new(false),
        }
    }

    /// Get a reference to an item from the map.
    pub fn get(&self, handle: SlotHandle) -> Option<&T> {
        let index = handle.index();
        if index >= self.slots.len() && handle.version == 0 {
            self.deferred_slots.get(index - self.slots.len())
        } else {
            let slot = &self.slots[index];
            slot.get(handle.version)
        }
    }

    /// Insert a new item into the map.
    /// ### Panics
    /// A shared write method like `insert_sync` has been called since the last `flush`.
    pub fn insert(&mut self, value: T) -> SlotHandle {
        self.assert_not_dirty();

        let index = if let Some(index) = self.free.pop() {
            // SAFETY: since we got the index from the free list, the slot should be free.
            unsafe { self.slots[index as usize].write_mut(value) };
            index
        } else {
            let index = self.slots.len();
            self.slots.push(Slot::new_occupied(value));
            index as u32
        };

        let version = self.slots[index as usize].version();

        SlotHandle::new(index, version)
    }

    /// Remove an item from the map.
    /// ### Panics
    /// A shared write method like `insert_sync` has been called since the last `flush`.
    pub fn remove(&mut self, handle: SlotHandle) -> Option<T> {
        self.assert_not_dirty();

        if let Some(slot) = self.slots.get_mut(handle.index()) {
            if slot.version() == handle.version {
                return slot.free();
            }
        }

        None
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots
            .iter()
            .filter_map(|slot| slot.get(slot.version()))
            .chain(self.deferred_slots.iter())
    }

    fn handle_is_valid(&self, handle: SlotHandle) -> bool {
        if handle.index < self.slots.len() as u32 {
            let slot = &self.slots[handle.index()];
            slot.version() == handle.version && slot.is_occupied()
        } else {
            let deferred_index = handle.index as usize - self.slots.len();
            // All deferred slots are version 0
            deferred_index < self.deferred_slots.len() && handle.version == 0
        }
    }

    fn assert_not_dirty(&self) {
        if self.dirty.load(Ordering::Relaxed) {
            panic!(
                "A non-concurrent write method like `insert` or `remove` was called \
                and no flush happened since the last concurrent write method like `concurrent_insert` or `deferred_remove`."
            );
        }
    }
}

unsafe impl<T: Send + Sync> Sync for ConcurrentSlotMap<T> {}

#[derive(Debug, Clone, Copy)]
pub struct SlotHandle {
    index: u32,
    version: Version,
}

impl SlotHandle {
    pub fn index(&self) -> usize {
        self.index as usize
    }
    pub fn version(&self) -> Version {
        self.version
    }
}

impl SlotHandle {
    pub const PLACEHOLDER: SlotHandle = SlotHandle::new(u32::MAX, Version::MAX);

    const fn new(index: u32, version: Version) -> SlotHandle {
        SlotHandle { index, version }
    }

    /// Should rarely be used. The handles should be obtained from the slot map when inserting
    /// instead.
    pub const fn _new(index: u32, version: Version) -> SlotHandle {
        SlotHandle::new(index, version)
    }
}

use slot::Slot;

// Putting Slot into an extra module to restrict field access and method usage.
mod slot {
    use std::{
        cell::UnsafeCell,
        mem::MaybeUninit,
        sync::atomic::{AtomicBool, Ordering},
    };

    use super::Version;

    // A slot, which represents storage for a value and a current version.
    // Can be occupied or vacant.
    #[derive(Debug)]
    pub struct Slot<T> {
        value: UnsafeCell<MaybeUninit<T>>,
        version: Version,
        occupied: bool,
        occupied_atomic: AtomicBool,
    }

    impl<T> Slot<T> {
        pub fn new_occupied(value: T) -> Self {
            Self {
                value: UnsafeCell::new(MaybeUninit::new(value)),
                version: 0,
                occupied: true,
                occupied_atomic: AtomicBool::new(true),
            }
        }

        pub fn get(&self, version: Version) -> Option<&T> {
            (self.version == version && self.is_occupied()).then(|| self.get_unchecked())
        }

        pub(crate) fn get_unchecked(&self) -> &T {
            unsafe { (*self.value.get()).assume_init_ref() }
        }

        pub fn version(&self) -> Version {
            self.version
        }

        /// SAFETY: self must be free and no further calls to `write` must happen until the slot is
        /// `free`d again.
        pub unsafe fn write(&self, value: T) {
            // Note:
            // The caller guarantees self is free, so `self.content` is uninit, which means
            // overwritting it below is fine. SAFETY
            // Aside from `write`, which the caller guarantees to not call again, there is only two
            // methods that access access `self.content`: `free` and `get`, which
            // protect against reading uninitialized values via `self.is_occupied`.
            unsafe { (*self.value.get()).write(value) };
            self.occupied_atomic.store(true, Ordering::Release);
        }

        /// Should be called to shortcircuit an atomic load in future calls to `Self::is_occupied`.
        /// # SAFETY
        /// Must be occupied.
        pub unsafe fn confirm_write(&mut self) {
            self.occupied = true;
        }

        /// # SAFETY
        /// Must be free.
        // Although technically not unsafe, this would not drop any overwritten `self.value` which
        // we want to avoid.
        pub unsafe fn write_mut(&mut self, value: T) {
            self.value.get_mut().write(value);
            self.occupied = true;
        }

        /// Frees the slot for later reuse.
        pub fn free(&mut self) -> Option<T> {
            if self.is_occupied() {
                self.version += 1;
                self.occupied = false;
                self.occupied_atomic = AtomicBool::new(false);

                Some(unsafe {
                    std::mem::replace(self.value.get_mut(), MaybeUninit::uninit()).assume_init()
                })
            } else {
                None
            }
        }

        // Is this slot occupied?
        #[inline(always)]
        pub fn is_occupied(&self) -> bool {
            self.occupied || self.occupied_atomic.load(Ordering::Acquire)
        }

        #[inline(always)]
        pub fn is_free(&self) -> bool {
            !self.is_occupied()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutOfSpace;
impl std::fmt::Display for OutOfSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Out of space in a slot map fixed-sized array")
    }
}
impl std::error::Error for OutOfSpace {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_concurrent_insert() {
        let mut map = ConcurrentSlotMap::new(100, 100);
        let m = &map;

        let handles: [SlotHandle; 10] = thread::scope(|s| {
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                .map(|i| s.spawn(move || m.concurrent_insert(i).unwrap()))
                .map(|t| t.join().unwrap())
        });

        // Verify all insertions were successful
        for handle in handles {
            let value = *map.get(handle).unwrap();
            assert!(value < 10);
        }

        map.flush();
    }

    #[test]
    fn test_basic_insert_get() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.concurrent_insert(42).unwrap();
        assert_eq!(*map.get(handle).unwrap(), 42);
        map.flush();
    }

    #[test]
    fn test_multiple_inserts() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let h1 = map.concurrent_insert(1).unwrap();
        let h2 = map.concurrent_insert(2).unwrap();
        let h3 = map.concurrent_insert(3).unwrap();

        assert_eq!(*map.get(h1).unwrap(), 1);
        assert_eq!(*map.get(h2).unwrap(), 2);
        assert_eq!(*map.get(h3).unwrap(), 3);

        map.flush();
    }

    #[test]
    fn test_insert_free_reuse() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let h1 = map.concurrent_insert(1).unwrap();

        map.deferred_remove(h1).unwrap();
        map.flush();

        println!("{:#?}", map);

        let h2 = map.concurrent_insert(2).unwrap();
        println!("{:#?}", map);
        assert_eq!(*map.get(h2).unwrap(), 2);
        assert!(map.get(h1).is_none());
        assert!(h1.index == h2.index);

        map.flush();
    }

    #[test]
    fn test_out_of_space() {
        let mut map = ConcurrentSlotMap::new(2, 2);
        let h1 = map.concurrent_insert(1).unwrap();
        let h2 = map.concurrent_insert(2).unwrap();
        assert!(map.concurrent_insert(3).is_err());

        map.deferred_remove(h1).unwrap();
        map.deferred_remove(h2).unwrap();
        map.flush();

        let h3 = map.concurrent_insert(3).unwrap();
        assert_eq!(*map.get(h3).unwrap(), 3);
    }

    #[test]
    fn test_free_invalid_handle() {
        let mut map = ConcurrentSlotMap::<u32>::new(10, 10);
        let invalid_handle = SlotHandle::_new(99, 0);
        assert!(map.deferred_remove(invalid_handle).is_ok()); // Should be ok since invalid handles are ignored
        map.flush();
    }

    #[test]
    fn test_mixed_operations() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let h1 = map.concurrent_insert(1).unwrap();
        let h2 = map.concurrent_insert(2).unwrap();

        assert_eq!(*map.get(h1).unwrap(), 1);
        map.deferred_remove(h1).unwrap();

        let h3 = map.concurrent_insert(3).unwrap();
        assert_eq!(*map.get(h2).unwrap(), 2);
        assert_eq!(*map.get(h3).unwrap(), 3);

        map.flush();

        assert!(map.get(h1).is_none());
    }

    #[test]
    fn test_flush_behavior() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let h1 = map.concurrent_insert(1).unwrap();
        map.deferred_remove(h1).unwrap();

        // Handle should still be valid before flush
        assert!(map.get(h1).is_some());

        map.flush();

        // Handle should be invalid after flush
        assert!(map.get(h1).is_none());

        let h2 = map.concurrent_insert(2).unwrap();
        assert_eq!(*map.get(h2).unwrap(), 2);
    }

    #[test]
    fn test_version_handling() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let h1 = map.concurrent_insert(1).unwrap();
        map.deferred_remove(h1).unwrap();
        map.flush();

        let h2 = map.concurrent_insert(2).unwrap();
        // Even if we reuse the same slot, the handle version should prevent access via old handle
        assert!(map.get(h1).is_none());
        assert_eq!(*map.get(h2).unwrap(), 2);
    }

    #[test]
    #[should_panic]
    fn test_exclusive_after_shared_without_flush() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let _ = map.concurrent_insert(1).unwrap();
        // This should panic because we didn't flush after insert_sync
        map.insert(2);
    }

    #[test]
    fn test_exclusive_after_shared_with_flush() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let _ = map.concurrent_insert(1).unwrap();
        map.flush();
        // This should work fine because we flushed
        let _ = map.insert(2);
    }

    #[test]
    fn test_generation_validity() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle1 = map.insert(1);
        map.remove(handle1);
        let handle2 = map.insert(2); // Reuses the same slot

        assert!(map.get(handle1).is_none()); // Old handle should be invalid
        assert_eq!(*map.get(handle2).unwrap(), 2); // New handle should be valid
    }

    #[test]
    #[should_panic]
    fn test_shared_then_exclusive_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let _ = map.concurrent_insert(1).unwrap();
        let _ = map.insert(2);
    }

    #[test]
    #[should_panic]
    fn test_shared_then_free_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.concurrent_insert(1).unwrap();
        map.remove(handle);
    }

    #[test]
    #[should_panic]
    fn test_deferred_free_then_free_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.insert(1);
        map.deferred_remove(handle).unwrap();
        map.remove(handle);
    }

    #[test]
    #[should_panic]
    fn test_deferred_free_then_insert_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.insert(1);
        map.deferred_remove(handle).unwrap();
        map.insert(2);
    }

    #[test]
    fn test_exclusive_then_shared_is_ok() {
        let mut map = ConcurrentSlotMap::new(10, 10);

        // First do exclusive writes
        let handle1 = map.insert(1);
        let handle2 = map.insert(2);
        map.remove(handle1);

        // Then do shared writes
        let handle3 = map.concurrent_insert(3).unwrap();
        map.deferred_remove(handle2).unwrap();

        // Verify everything worked
        assert!(map.get(handle1).is_none());
        assert_eq!(*map.get(handle2).unwrap(), 2);
        assert_eq!(*map.get(handle3).unwrap(), 3);

        map.flush();

        // After flush
        assert!(map.get(handle1).is_none());
        assert!(map.get(handle2).is_none());
        assert_eq!(*map.get(handle3).unwrap(), 3);
    }

    // TODO: use loom for more guaranteed tests
    #[test]
    fn test_concurrent_stress_mixed_operations() {
        use rand::prelude::*;

        const NUM_THREADS: usize = 4;
        const OPS_PER_THREAD: usize = 1000;
        const SIZE: usize = 100;

        #[derive(Clone, Copy)]
        enum Operation {
            Insert(u32),
            Free(usize),
            Get(usize),
        }

        let mut map = ConcurrentSlotMap::<u32>::new(SIZE, SIZE);
        let mut rng = rand::rng();

        let thread_ops: Vec<Vec<Operation>> = (0..NUM_THREADS)
            .map(|_| {
                let mut ops = Vec::with_capacity(OPS_PER_THREAD);
                for _ in 0..OPS_PER_THREAD {
                    let op = match rng.random_range(0..3) {
                        0 => Operation::Insert(rng.random()),
                        1 => Operation::Free(rng.random_range(0..SIZE)),
                        2 => Operation::Get(rng.random_range(0..SIZE)),
                        _ => unreachable!(),
                    };
                    ops.push(op);
                }
                ops
            })
            .collect();

        for _ in 0..3 {
            let map_ref = &map;
            thread::scope(|s| {
                for ops in thread_ops.clone() {
                    s.spawn(move || {
                        for op in ops {
                            match op {
                                Operation::Insert(value) => {
                                    if let Ok(handle) = map_ref.concurrent_insert(value) {
                                        assert_eq!(*map_ref.get(handle).unwrap(), value);
                                    }
                                }
                                Operation::Free(handle_idx) => {
                                    let handle = SlotHandle::_new(handle_idx as u32, 0);
                                    let _ = map_ref.deferred_remove(handle);
                                }
                                Operation::Get(handle_idx) => {
                                    let handle = SlotHandle::_new(handle_idx as u32, 0);
                                    let _ = map_ref.get(handle);
                                }
                            }
                        }
                    });
                }
            });

            map.flush();
        }
    }
}
