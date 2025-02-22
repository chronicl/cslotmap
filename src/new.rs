use std::cell::UnsafeCell;
use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::bumpvec::BumpVec;

type Version = u16;

#[derive(Debug)]
pub struct DeferredSlotMap<T> {
    slots: Vec<Slot<T>>,
    free_head: AtomicU32,

    deferred_slots: BumpVec<T>,
    deferred_frees: BumpVec<u32>,
    dirty: AtomicBool,
}

impl<T> DeferredSlotMap<T> {
    const INVALID_FREE: u32 = u32::MAX;

    pub fn new(allocation_capacity: usize, deferred_frees_capacity: usize) -> Self {
        Self {
            slots: Default::default(),
            free_head: AtomicU32::new(Self::INVALID_FREE),

            deferred_slots: BumpVec::new(allocation_capacity),
            deferred_frees: BumpVec::new(deferred_frees_capacity),
            dirty: AtomicBool::new(false),
        }
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let index = handle.index();
        if index >= self.slots.len() && handle.version == 0 {
            self.deferred_slots.get(index - self.slots.len())
        } else {
            let slot = &self.slots[index];
            if slot.version == handle.version {
                match slot.get() {
                    Occupied(value) => Some(value),
                    _ => None,
                }
            } else {
                None
            }
        }
    }

    pub fn insert(&self, value: T) -> Result<Handle<T>, OutOfSpace> {
        let (index, version) = if let Some(index) = self.reserve_free_slot() {
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

        Ok(Handle::new(index, version))
    }

    fn reserve_free_slot(&self) -> Option<u32> {
        let mut curr = self.free_head.load(Ordering::Relaxed);
        while curr != Self::INVALID_FREE {
            let old = curr;
            assert!(self.slots[curr as usize].is_free());
            match self.free_head.compare_exchange_weak(
                old,
                unsafe { self.slots[curr as usize].u.next_free },
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(curr);
                }
                Err(new) => curr = new,
            }
        }
        None
    }

    /// Returns the generation of the slot.
    /// SAFETY:
    /// index must have previously been reserved via `reserve_free_slot`, so that no other
    /// shared write method can possibly use the same free slot.
    /// self must be marked as dirty, so that no exclusive write method can possibly use the free slot, because
    /// exclusive write methods can't be used as long as self is dirty.
    unsafe fn write_to_reserved_slot(&self, index: u32, value: T) -> Version {
        let slot = &self.slots[index as usize];
        assert!(slot.is_free());

        // SAFETY: See above.
        // Reads are secured by slot.occupied_atomic
        // TODO: maybe have to free AtomicU32 here
        unsafe { (*slot.u.value).get().write(value) };
        slot.occupied_atomic.store(true, Ordering::Release);

        slot.version
    }

    fn handle_is_valid(&self, handle: Handle<T>) -> bool {
        if handle.index < self.slots.len() as u32 {
            let slot = &self.slots[handle.index()];
            slot.version == handle.version && slot.is_occupied()
        } else {
            let deferred_index = handle.index as usize - self.slots.len();
            // All deferred slots are version 0
            deferred_index < self.deferred_slots.len() && handle.version == 0
        }
    }

    pub fn free(&self, handle: Handle<T>) -> Result<(), OutOfSpace> {
        if self.handle_is_valid(handle) && self.deferred_frees.push(handle.index).is_none() {
            return Err(OutOfSpace);
        }

        self.dirty.store(true, Ordering::Relaxed);

        Ok(())
    }

    pub fn flush(&mut self) {
        self.flush_with(|_| {})
    }

    pub fn flush_with<R>(&mut self, free_fn: impl FnOnce(&mut dyn Iterator<Item = T>) -> R) -> R {
        self.dirty.store(false, Ordering::Relaxed);

        let Self {
            slots,
            free_head,
            deferred_slots,
            deferred_frees,
            ..
        } = self;

        // Move from deferred slots to slots to make space for future shared insertions
        deferred_slots.clear(|values| {
            for value in values {
                slots.push(Slot::new_occupied(value));
            }
        });

        let head = &mut free_head.load(Ordering::Relaxed);
        // Appying deferred frees
        let r = deferred_frees.clear(|indices| {
            let mut values = indices.flat_map(|index| {
                let slot = &mut slots[index as usize];
                // The same index maybe occur multiple times in deferred frees, so it could already be freed here
                if slot.is_occupied_mut() {
                    let value = unsafe { ManuallyDrop::take(&mut slot.u.value) };

                    slot.u.next_free = *head;
                    *head = index;
                    slot.version += 1;
                    slot.occupied = false;
                    slot.occupied_atomic = AtomicBool::new(false);

                    Some(value.into_inner())
                } else {
                    None
                }
            });

            let r = free_fn(&mut values);

            values.count();

            r
        });
        *free_head = AtomicU32::new(*head);

        r
    }
}

unsafe impl<T: Send + Sync> Sync for DeferredSlotMap<T> {}

#[derive(Debug)]
pub struct Handle<T> {
    index: u32,
    version: Version,
    _type: PhantomData<T>,
}

impl<T> Handle<T> {
    fn index(&self) -> usize {
        self.index as usize
    }
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
    pub const PLACEHOLDER: Handle<T> = Handle::new(u32::MAX, Version::MAX);

    const fn new(index: u32, version: Version) -> Handle<T> {
        Handle {
            index,
            version,
            _type: PhantomData::<T>,
        }
    }

    /// Should rarely be used. The handles should be obtained from the slot map when inserting instead.
    pub const fn _new(index: u32, version: Version) -> Handle<T> {
        Handle::new(index, version)
    }
}

// A slot, which represents storage for a value and a current version.
// Can be occupied or vacant.
struct Slot<T> {
    u: SlotUnion<T>,
    version: Version,
    occupied: bool,
    occupied_atomic: AtomicBool,
}

// Storage inside a slot or metadata for the freelist when vacant.
union SlotUnion<T> {
    value: ManuallyDrop<UnsafeCell<T>>,
    next_free: u32,
}

// Safe API to read a slot.
enum SlotContent<'a, T: 'a> {
    Occupied(&'a T),
    Vacant(u32),
}

use self::SlotContent::{Occupied, Vacant};

impl<T> Slot<T> {
    pub fn new_occupied(value: T) -> Self {
        Self {
            u: SlotUnion {
                value: ManuallyDrop::new(UnsafeCell::new(value)),
            },
            version: 0,
            occupied: true,
            occupied_atomic: AtomicBool::new(true),
        }
    }

    // Is this slot occupied?
    #[inline(always)]
    pub fn is_occupied(&self) -> bool {
        self.occupied || self.occupied_atomic.load(Ordering::Acquire)
    }

    pub fn is_occupied_mut(&mut self) -> bool {
        self.occupied
    }

    #[inline(always)]
    pub fn is_free(&self) -> bool {
        !self.is_occupied()
    }

    pub fn get(&self) -> SlotContent<T> {
        if self.is_occupied() {
            Occupied(unsafe { &*self.u.value.get() })
        } else {
            Vacant(unsafe { self.u.next_free })
        }
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        if core::mem::needs_drop::<T>() && self.is_occupied() {
            // This is safe because we checked that we're occupied.
            unsafe {
                ManuallyDrop::drop(&mut self.u.value);
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Slot<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let mut builder = fmt.debug_struct("Slot");
        builder.field("version", &self.version);
        match self.get() {
            Occupied(value) => builder.field("value", value).finish(),
            Vacant(next_free) => builder.field("next_free", &next_free).finish(),
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
        let mut map = DeferredSlotMap::new(100, 100);
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
    fn test_basic_insert_get() {
        let mut map = DeferredSlotMap::new(10, 10);
        let handle = map.insert(42).unwrap();
        assert_eq!(*map.get(handle).unwrap(), 42);
        map.flush();
    }

    #[test]
    fn test_multiple_inserts() {
        let mut map = DeferredSlotMap::new(10, 10);
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();
        let h3 = map.insert(3).unwrap();

        assert_eq!(*map.get(h1).unwrap(), 1);
        assert_eq!(*map.get(h2).unwrap(), 2);
        assert_eq!(*map.get(h3).unwrap(), 3);

        map.flush();
    }

    #[test]
    fn test_insert_free_reuse() {
        let mut map = DeferredSlotMap::new(10, 10);
        let h1 = map.insert(1).unwrap();

        map.free(h1).unwrap();
        map.flush();

        println!("{:#?}", map);

        let h2 = map.insert(2).unwrap();
        println!("{:#?}", map);
        assert_eq!(*map.get(h2).unwrap(), 2);
        assert!(map.get(h1).is_none());
        assert!(h1.index == h2.index);

        map.flush();
    }

    #[test]
    fn test_out_of_space() {
        let mut map = DeferredSlotMap::new(2, 2);
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();
        assert!(map.insert(3).is_err());

        map.free(h1).unwrap();
        map.free(h2).unwrap();
        map.flush();

        let h3 = map.insert(3).unwrap();
        assert_eq!(*map.get(h3).unwrap(), 3);
    }

    #[test]
    fn test_free_invalid_handle() {
        let mut map = DeferredSlotMap::<u32>::new(10, 10);
        let invalid_handle = Handle::_new(99, 0);
        assert!(map.free(invalid_handle).is_ok()); // Should be ok since invalid handles are ignored
        map.flush();
    }

    #[test]
    fn test_mixed_operations() {
        let mut map = DeferredSlotMap::new(10, 10);
        let h1 = map.insert(1).unwrap();
        let h2 = map.insert(2).unwrap();

        assert_eq!(*map.get(h1).unwrap(), 1);
        map.free(h1).unwrap();

        let h3 = map.insert(3).unwrap();
        assert_eq!(*map.get(h2).unwrap(), 2);
        assert_eq!(*map.get(h3).unwrap(), 3);

        map.flush();

        assert!(map.get(h1).is_none());
    }

    #[test]
    fn test_flush_behavior() {
        let mut map = DeferredSlotMap::new(10, 10);
        let h1 = map.insert(1).unwrap();
        map.free(h1).unwrap();

        // Handle should still be valid before flush
        assert!(map.get(h1).is_some());

        map.flush();

        // Handle should be invalid after flush
        assert!(map.get(h1).is_none());

        let h2 = map.insert(2).unwrap();
        assert_eq!(*map.get(h2).unwrap(), 2);
    }

    #[test]
    fn test_version_handling() {
        let mut map = DeferredSlotMap::new(10, 10);
        let h1 = map.insert(1).unwrap();
        map.free(h1).unwrap();
        map.flush();

        let h2 = map.insert(2).unwrap();
        // Even if we reuse the same slot, the handle version should prevent access via old handle
        assert!(map.get(h1).is_none());
        assert_eq!(*map.get(h2).unwrap(), 2);
    }

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

        let mut map = DeferredSlotMap::<u32>::new(SIZE, SIZE);
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
        }
    }
}
