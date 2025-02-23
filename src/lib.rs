use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
};

use bitset::Bitset;
use bumpvec::BumpVec;

mod bitset;
mod bumpvec;
pub mod new;

/// ## A (mostly) lock-less concurrent slot map.
///
/// Inserting a value into this map allocates a slot in a `Vec` and returns a `Handle` by which the value
/// can be retrieved.
/// Removing a value from this map, opens up it's slot for reuse by future insertions.
/// To invalidate old handles to the same slot, each slot has an associated generation counter which is incremented upon freeing it.
///
/// ### Concurrency
/// To avoid requiring locking mechanisms for concurrent usage, an intermediate fixed size array is used.
/// Upon inserting a value, it is first stored in the fixed size array and only upon calling `ConcurrentSlotMap::flush(&mut self)`
/// will the value be moved to a dynamically sized `Vec`.
/// TODO: If the fixed size array is full a Arc<Mutex<Vec<_>>> is used as intermediate storage instead, which requires locking, so the slot map is only
/// lock-less if `allocations <= fixed size array length` inbetween flushes.
///
/// Freeing a value concurrently also uses a fixed size array, but the free is deferred until `ConcurrentSlotMap::flush(&mut self)` is called.
/// Only non-deferred free slots are used when inserting new values.
///
/// The size of the fixed size array can be configured upon ConcurrentSlotMap creation via `ConcurrentSlotMap::new`
/// or modified later via `ConcurrentSlotMap::set_fixed_size_array_sizes`.
///
/// ### Method overview
/// The methods can be divided into read methods and write methods. The read methods only require a shared reference (&self) and may be used in
/// single and multi-threaded contexts. The write methods can further be divided into shared (&self) and exclusive (&mut self) write methods.
/// The exclusive write methods are designed for high performance single-threaded workloads, they are
/// - `insert`
/// - `free`
///
/// whereas the shared write methods are designed for high performance multi-threaded workloads, they are
/// - `insert_sync`
/// - `deferred_free`
///
/// **However**, they can not be freely combined. When calling a shared write method, Self is marked as dirty and exclusive write methods
/// become unavailable until calling `flush`, which marks Self as clean. But, calling exclusive write methods before shared write methods is valid.
/// - 🚫 shared write -> exclusive write
/// - ✅ shared write -> flush -> exclusive write
/// - ✅ exclusive write -> shared write
///
/// ### Usage in render graph
/// This slot map is used for storing render resources like textures and buffers and assigning them an index in a **bindless descriptor array**.
///
/// At the start of a frame the slot map represents the bindless descriptor array of the previous frame, during frame preparation we modify
/// the slot map with new allocations and frees and once the preparation is done **and** the previous frame has finished rendering we apply
/// the allocations and frees to the bindless descriptor array to then start rendering the current frame. Note that the slot map now represents
/// the current frames bindless descriptor array and we can start preparing the next frame following the same steps.
pub struct ConcurrentSlotMap<T> {
    // These fields require exclusive (single-threaded, &mut) write access.
    // They will only be updated upon calling flush or if Self is currently
    // not dirty (is flushed) and methods like `insert` or `free` are used.
    slots: Vec<UnsafeCell<T>>,
    free: Vec<u32>,
    free_bitset: Bitset,
    generations: Vec<u32>,

    // These fields allow for lock-free concurrent write access. See methods like `deferred_free` and `insert_sync`.
    /// These extend the slots in `slots`, they all have generation 0 and are newly allocated since the last flush.
    deferred_slots: BumpVec<T>,
    /// Only resources that were freed before the last flush are reused.
    /// Here we keep track of how many of the slots in `free` (from the end to start) we've used since the last flush.
    frees_used: AtomicUsize,
    /// Newly freed resources are stored in `deferred_frees` and will become available after the next flush.
    /// MAYBE: This could use a lighter version of BumpVec - one that doesn't allow read access.
    deferred_frees: BumpVec<u32>,
    /// Used for synchronizing reuse of slots. This is only accessed for newly (after last flush) allocated resources that reused a slot.
    /// Once a new value has been written to a reused slot, the highest bit of this u32 is set to indicate that the slot is now valid and
    /// to synchronize with reads (`get`). Is incremented alongside `generations`.
    atomic_generations: Vec<AtomicU32>,

    /// Tracks whether any shared write (`insert_sync`, `deferred_free`) has happened since the last flush.
    /// MAYBE: `deferred_free` shouldn't set dirty.
    dirty: AtomicBool,
}

unsafe impl<T: Send + Sync> Sync for ConcurrentSlotMap<T> {}

#[derive(Debug, Clone, Copy)]
pub struct OutOfSpace;
impl std::fmt::Display for OutOfSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Out of space in a slot map fixed-sized array")
    }
}
impl std::error::Error for OutOfSpace {}

impl<T> ConcurrentSlotMap<T> {
    const ATOMIC_GENERATION_USED_BIT: u32 = 1 << 31;
    const ATOMIC_GENERATION_MASK: u32 = Self::ATOMIC_GENERATION_USED_BIT - 1;

    pub fn new(allocations_capacity: usize, deferred_frees_capacity: usize) -> Self {
        Self {
            slots: Default::default(),
            free: Default::default(),
            free_bitset: Default::default(),
            generations: Default::default(),

            deferred_slots: BumpVec::new(allocations_capacity),
            frees_used: AtomicUsize::new(0),
            deferred_frees: BumpVec::new(deferred_frees_capacity),
            atomic_generations: Default::default(),

            dirty: AtomicBool::new(false),
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        // if not in range of self.slots, it's a deferred slot and deferred slots always have generation 0
        if handle.index >= self.slots.len() as u32 && handle.generation == 0 {
            self.deferred_slots
                .get(handle.index as usize - self.slots.len())
        } else if self.generation_is_valid(handle) {
            // SAFETY: self.generation_is_valid synchronizes any potential write via `insert_sync`, which can only happen once every flush and
            // there is no other shared write method that writes to self.slots, so we can safely get
            // a reference here.
            Some(unsafe { &*self.slots[handle.index as usize].get() })
        } else {
            None
        }
    }

    pub fn insert(&self, value: T) -> Result<Handle<T>, OutOfSpace> {
        let (index, generation) = if let Some(index) = self.reserve_free_slot() {
            // SAFETY: index was reserved via `reserve_free_slot`. self is marked as dirty in this method.
            let generation = unsafe { self.write_to_reserved_slot(index, value) };
            (index, generation)
        } else if let Some(index) = self.deferred_slots.push(value) {
            // deferred_slots will be appeneded to slots upon calling flush, so the handle index is actually self.slots.len() + index
            ((self.slots.len() + index) as u32, 0)
        } else {
            return Err(OutOfSpace);
        };

        self.dirty.store(true, Ordering::Relaxed);

        Ok(Handle::new(index, generation))
    }

    /// Frees are deferred until next call to `flush`.
    ///
    /// If the handle is invalid, returns Ok(()), but it's (non-existent) data is not gonna be freed.
    pub fn free(&self, handle: Handle<T>) -> Result<(), OutOfSpace> {
        if self.generation_is_valid(handle) && self.deferred_frees.push(handle.index).is_none() {
            return Err(OutOfSpace);
        }
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// ### Panics
    /// A shared write method like `insert_sync` has been called since the last `flush`.
    pub fn insert_mut(&mut self, value: T) -> Handle<T> {
        self.assert_not_dirty();

        let index = if let Some(index) = self.free.pop() {
            self.free_bitset.unset(index);
            *self.slots[index as usize].get_mut() = value;
            index
        } else {
            let index = self.slots.len();
            self.slots.push(UnsafeCell::new(value));
            self.generations.push(0);
            self.atomic_generations.push(AtomicU32::new(0));
            index as u32
        };

        let generation = self.generations[index as usize];

        Handle::new(index, generation)
    }

    /// ### Panics
    /// A shared write method like `insert_sync` has been called since the last `flush`.
    pub fn free_mut(&mut self, handle: Handle<T>) -> bool {
        self.assert_not_dirty();

        let was_not_already_free = self.free_bitset.set(handle.index);
        if was_not_already_free {
            self.free.push(handle.index);
            let i = handle.index as usize;
            self.generations[i] += 1;
            self.atomic_generations[i].store(self.generations[i], Ordering::Relaxed);
        }
        was_not_already_free
    }

    fn reserve_free_slot(&self) -> Option<u32> {
        let frees_used = self.frees_used.fetch_add(1, Ordering::Relaxed);
        let len = self.free.len();
        (frees_used < len).then(|| self.free[len - 1 - frees_used])
    }

    /// Returns the generation of the slot.
    /// SAFETY:
    /// index must have previously been reserved via `reserve_free_slot`, so that no other
    /// shared write method can possibly use the same free slot.
    /// self must be marked as dirty, so that no exclusive write method can possibly use the free slot, because
    /// exclusive write methods can't be used as long as self is dirty.
    unsafe fn write_to_reserved_slot(&self, index: u32, value: T) -> u32 {
        assert!(self.is_free(index));

        // SAFETY: See above.
        // Reads are secured by self.atomic_generations.
        unsafe { self.slots[index as usize].get().write(value) };
        // or 1 << 32 and release Ordering for synchronization in `generation_is_valid`
        self.atomic_generations[index as usize]
            .fetch_or(Self::ATOMIC_GENERATION_USED_BIT, Ordering::Release)
    }

    /// ### Panics
    /// A shared write method like `insert_sync` has been called since the last `flush`.
    pub fn set_fixed_size_array_sizes(
        &mut self,
        allocations_per_frame_capacity: usize,
        deferred_frees_capacity: usize,
    ) {
        self.assert_not_dirty();

        self.deferred_slots = BumpVec::new(allocations_per_frame_capacity);
        self.deferred_frees = BumpVec::new(deferred_frees_capacity);
    }

    pub fn flush(&mut self) {
        // All of the fields need to be updated
        // 1.  slots: Vec<UnsafeCell<T>>           fill with deferred_slots
        // 2.  free: Vec<u32>                      remove last `frees_used` elements then fill with deferred frees
        // 3.  free_bitset:                        unset last `frees_used` elements from `free` then fill with deferred frees
        // 4.  generations: Vec<u32>               set to 0 for the values from deferred_slots and incremented for deferred frees
        // 5.  deferred_slots: BumpVec<T>          clear
        // 6.  frees_used: AtomicUsize             set to 0
        // 7.  deferred_frees: BumpVec<u32>        clear
        // 8.  atomic_generations: Vec<AtomicU32>  set to 0 for the values from deferred_slots and incremented for deferred frees
        // 9.  dirty: AtomicBool                   set to false

        if !self.dirty.load(Ordering::Relaxed) {
            return;
        }
        // 9. done
        self.dirty.store(false, Ordering::Relaxed);

        {
            let Self {
                slots,
                deferred_slots,
                generations,
                atomic_generations,
                ..
            } = self;

            // 5. done
            deferred_slots.clear(|values| {
                for value in values {
                    // 1. done
                    slots.push(UnsafeCell::new(value));
                    // 4. partially done
                    generations.push(0);
                    // 8. partially done
                    atomic_generations.push(AtomicU32::new(0));
                }
            })
        }

        {
            let Self {
                free,
                free_bitset,
                frees_used,
                generations,
                atomic_generations,
                deferred_frees,
                ..
            } = self;

            // reusing free slots
            let used = frees_used.load(Ordering::Relaxed);
            // 6. done
            frees_used.store(0, Ordering::Relaxed);
            for _ in 0..used.min(free.len()) {
                // 2. partially done
                let index = free.pop().unwrap();
                // 3. partially done
                free_bitset.unset(index);
            }

            // applying deferred frees
            // 7. done
            deferred_frees.clear(|indices| {
                for index in indices {
                    // 3. done
                    if free_bitset.set(index) {
                        // 2. done
                        free.push(index);
                        // 4. done
                        generations[index as usize] += 1;
                        // This needs to be set to generations[index] instead of just incremented
                        // because it may stil contain the ATOMIC_GENERATION_USED_BIT, which generations[index] doesn't.
                        // 8. done
                        atomic_generations[index as usize]
                            .store(generations[index as usize], Ordering::Relaxed);
                    }
                }
            });
        }
    }

    pub fn len_last_flush(&self) -> usize {
        self.slots.len()
    }

    fn assert_not_dirty(&self) {
        if self.dirty.load(Ordering::Relaxed) {
            panic!(
                "An exclusive (single-threaded, &mut) write method like `insert` was called \
                    and no flush happened since the last concurrent write method like `insert_sync`."
            );
        }
    }

    fn is_free(&self, index: u32) -> bool {
        self.free_bitset.get(index)
    }

    // isn't free but it's being reused

    fn generation_is_valid(&self, handle: Handle<T>) -> bool {
        let generation = if self.is_free(handle.index) {
            // Possibly reused free slot since last flush
            let generation = self.atomic_generations[handle.index as usize].load(Ordering::Acquire);
            if generation & Self::ATOMIC_GENERATION_USED_BIT == 0 {
                return false;
            }
            generation & Self::ATOMIC_GENERATION_MASK
        } else {
            // Is not free, so `self.generations` will be valid for it until flush
            let len = self.slots.len() as u32;
            if handle.index < len {
                self.generations[handle.index as usize]
            } else if handle.index - len < self.deferred_slots.len() as u32 {
                0
            } else {
                return false;
            }
        };

        handle.generation == generation
    }

    #[allow(dead_code)]
    fn check(&mut self) {
        self.assert_not_dirty();

        for &index in self.free.iter() {
            assert!(self.free_bitset.get(index));
        }

        let mut i = 0;
        while i < self.slots.len() {
            let bit = self.free_bitset.get(i as u32);
            if bit {
                assert!(
                    self.free.contains(&(i as u32)),
                    "Index {} was marked as free in free_bitset but not found in free list",
                    i
                );
            }
            i += 1;
        }

        let mut free_counts = vec![0; self.slots.len()];
        for &index in self.free.iter() {
            free_counts[index as usize] += 1;
            assert_eq!(
                free_counts[index as usize], 1,
                "Found duplicate index {} in free list",
                index
            );
        }

        for &index in self.free.iter() {
            let generation = self.atomic_generations[index as usize].load(Ordering::Relaxed);
            assert_eq!(
                generation & Self::ATOMIC_GENERATION_USED_BIT,
                0,
                "Found ATOMIC_GENERATION_USED_BIT set for free index {}",
                index
            );
        }
    }
}

#[derive(Debug)]
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _type: PhantomData<T>,
}

// Manual impl to avoid dependent on T: Copy from derive
impl<T> Copy for Handle<T> {}
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T> Send for Handle<T> {}
unsafe impl<T> Sync for Handle<T> {}

impl<T> Handle<T> {
    pub const PLACEHOLDER: Handle<T> = Handle::new(u32::MAX, u32::MAX);

    const fn new(index: u32, generation: u32) -> Handle<T> {
        Handle {
            index,
            generation,
            _type: PhantomData::<T>,
        }
    }

    /// Should rarely be used. The handles should be obtained from the slot map when inserting instead.
    pub const fn _new(index: u32, generation: u32) -> Handle<T> {
        Handle::new(index, generation)
    }
}

impl<T> From<Handle<T>> for RawHandle {
    fn from(handle: Handle<T>) -> Self {
        RawHandle {
            index: handle.index,
            generation: handle.generation,
        }
    }
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct RawHandle {
    index: u32,
    generation: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_basic_insert_get() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.insert_mut(42);
        assert_eq!(*map.get(handle).unwrap(), 42);
    }

    #[test]
    fn test_free_and_reuse() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle1 = map.insert_mut(1);
        let handle2 = map.insert_mut(2);

        map.free_mut(handle1);
        let handle3 = map.insert_mut(3);

        assert!(map.get(handle1).is_none()); // Original handle should be invalid
        assert_eq!(*map.get(handle2).unwrap(), 2); // Untouched slot should remain valid
        assert_eq!(*map.get(handle3).unwrap(), 3); // New value in reused slot
    }

    #[test]
    fn test_concurrent_insert() {
        let mut map = ConcurrentSlotMap::new(100, 100);
        let m = &map;

        let handles: [Handle<u32>; 10] = thread::scope(|s| {
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                .map(|i| s.spawn(move || m.insert(i).unwrap()))
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
    fn test_concurrent_insert_and_get() {
        // Mostly a test for miri.
        let mut map = ConcurrentSlotMap::<u32>::new(1000, 1000);

        thread::scope(|s| {
            let handles: Vec<_> = (0..10)
                .map(|i| {
                    let closure: Box<dyn Fn() + Send + Sync> = match i % 2 {
                        0 => Box::new(|| {
                            (0..100)
                                .map(|j| {
                                    map.insert(j).unwrap();
                                })
                                .count();
                        }),
                        1 => Box::new(|| {
                            (0..1000)
                                .rev()
                                .map(|j| {
                                    map.get(Handle::_new(j, 0));
                                })
                                .count();
                        }),
                        _ => unreachable!(),
                    };
                    s.spawn(closure)
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }
        });

        map.flush();
        assert_eq!(map.len_last_flush(), 500);
    }

    #[test]
    fn test_deferred_free() {
        let mut map = ConcurrentSlotMap::new(10, 10);

        let handle = map.insert(123).unwrap();
        assert_eq!(*map.get(handle).unwrap(), 123);

        map.free(handle).unwrap();
        // Value should still be accessible until flush
        assert_eq!(*map.get(handle).unwrap(), 123);

        map.flush();
        // After flush, handle should be invalid
        assert!(map.get(handle).is_none());
    }

    #[test]
    fn test_capacity_limits() {
        let mut map = ConcurrentSlotMap::new(2, 2);

        // Test allocation capacity
        let _h1 = map.insert(1).unwrap();
        let _h2 = map.insert(2).unwrap();
        assert!(map.insert(3).is_err()); // Should fail as capacity is exceeded

        map.flush();

        // Test deferred free capacity
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();
        map.free(h1).unwrap();
        map.free(h2).unwrap();
        // Should fail as deferred free capacity is exceeded.
        // Note that freeing the same item twice is not deduplicated.
        assert!(map.free(h2).is_err());
    }

    #[test]
    fn test_set_fixed_size_array_sizes() {
        let mut map = ConcurrentSlotMap::new(2, 2);
        let h1 = map.insert_mut(1);

        map.set_fixed_size_array_sizes(4, 4);

        // Verify existing data remains valid
        assert_eq!(*map.get(h1).unwrap(), 1);

        // Verify new capacities work
        let _h2 = map.insert(2).unwrap();
        let _h3 = map.insert(3).unwrap();
        let _h4 = map.insert(4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_exclusive_after_shared_without_flush() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let _ = map.insert(1).unwrap();
        // This should panic because we didn't flush after insert_sync
        map.insert_mut(2);
    }

    #[test]
    fn test_exclusive_after_shared_with_flush() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let _ = map.insert(1).unwrap();
        map.flush();
        // This should work fine because we flushed
        let _ = map.insert_mut(2);
    }

    #[test]
    fn test_generation_validity() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle1 = map.insert_mut(1);
        map.free_mut(handle1);
        let handle2 = map.insert_mut(2); // Reuses the same slot

        assert!(map.get(handle1).is_none()); // Old handle should be invalid
        assert_eq!(*map.get(handle2).unwrap(), 2); // New handle should be valid
    }

    #[test]
    #[should_panic]
    fn test_shared_then_exclusive_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let _ = map.insert(1).unwrap();
        let _ = map.insert_mut(2);
    }

    #[test]
    #[should_panic]
    fn test_shared_then_free_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.insert(1).unwrap();
        map.free_mut(handle);
    }

    #[test]
    #[should_panic]
    fn test_deferred_free_then_free_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.insert_mut(1);
        map.free(handle).unwrap();
        map.free_mut(handle);
    }

    #[test]
    #[should_panic]
    fn test_deferred_free_then_insert_should_panic() {
        let mut map = ConcurrentSlotMap::new(10, 10);
        let handle = map.insert_mut(1);
        map.free(handle).unwrap();
        map.insert_mut(2);
    }

    #[test]
    fn test_exclusive_then_shared_is_ok() {
        let mut map = ConcurrentSlotMap::new(10, 10);

        // First do exclusive writes
        let handle1 = map.insert_mut(1);
        let handle2 = map.insert_mut(2);
        map.free_mut(handle1);

        // Then do shared writes
        let handle3 = map.insert(3).unwrap();
        map.free(handle2).unwrap();

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

    #[test]
    fn test_insertions() {
        let mut map = ConcurrentSlotMap::new(3200, 3200);
        for i in 0..1000 {
            map.insert(i).unwrap();
        }
        map.flush();
        assert!(map.len_last_flush() == 1000);
    }

    // MAYBE: Use loom instead for proper permutation testing.
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
                                    if let Ok(handle) = map_ref.insert(value) {
                                        assert_eq!(*map_ref.get(handle).unwrap(), value);
                                    }
                                }
                                Operation::Free(handle_idx) => {
                                    let handle = Handle::_new(handle_idx as u32, 0);
                                    let _ = map_ref.free(handle);
                                }
                                Operation::Get(handle_idx) => {
                                    let handle = Handle::_new(handle_idx as u32, 0);
                                    let _ = map_ref.get(handle);
                                }
                            }
                        }
                    });
                }
            });

            map.flush();
            map.check();
        }
    }
}
