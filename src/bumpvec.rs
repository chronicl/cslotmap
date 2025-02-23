use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

// TODO: should be renamed
pub struct BumpVec<T> {
    data: Box<[UnsafeCell<MaybeUninit<T>>]>,
    guards: Box<[AtomicBool]>,
    max_allocated: AtomicUsize,
}

impl<T> std::fmt::Debug for BumpVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BumpVec {{ len: {} }}", self.len())
    }
}

impl<T> BumpVec<T> {
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
            max_allocated: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, value: T) -> Option<usize> {
        let index = self.max_allocated.fetch_add(1, Ordering::Relaxed);
        if index >= self.data.len() {
            return None;
        }

        unsafe {
            (*self.data[index].get()).as_mut_ptr().write(value);
        }
        // Sync 1: synchronize above write
        self.guards[index].store(true, Ordering::Release);

        Some(index)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            return None;
        }

        // Sync 1: synchronize below read with write in push
        if self.guards[index].load(Ordering::Acquire) {
            Some(unsafe { (*self.data[index].get()).assume_init_ref() })
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.max_allocated
            .load(Ordering::Relaxed)
            .min(self.data.len())
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len()).filter_map(|i| self.get(i))
    }

    pub fn clear<R>(&mut self, values_fn: impl FnOnce(&mut dyn Iterator<Item = T>) -> R) -> R {
        let mut iter = (0..self.len()).map(|i| {
            // Sync 1
            self.guards[i].load(Ordering::Acquire);
            let value = std::mem::replace(self.data[i].get_mut(), MaybeUninit::uninit());
            // I belive this could be Relaxed, because when setting the guard to false,
            // no read of the data is performed when the guard is loaded anywhere.
            self.guards[i].store(false, Ordering::Release);
            unsafe { value.assume_init() }
        });
        let r = values_fn(&mut iter);
        iter.count();
        self.max_allocated.store(0, Ordering::Relaxed);
        r
    }
}

unsafe impl<T: Send> Sync for BumpVec<T> {}

impl<T> Drop for BumpVec<T> {
    fn drop(&mut self) {
        let len = self.max_allocated.load(Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_new() {
        let vec = BumpVec::<i32>::new(5);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_push_and_get() {
        let vec = BumpVec::new(3);

        assert_eq!(vec.push(1), Some(0));
        assert_eq!(vec.push(2), Some(1));
        assert_eq!(vec.push(3), Some(2));
        assert_eq!(vec.push(4), None); // Should fail, capacity exceeded

        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&3));
        assert_eq!(vec.get(3), None); // Out of bounds
    }

    #[test]
    fn test_len() {
        let vec = BumpVec::new(5);
        assert_eq!(vec.len(), 0);

        vec.push(1);
        assert_eq!(vec.len(), 1);

        vec.push(2);
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut vec = BumpVec::new(3);
        vec.push(1);
        vec.push(2);
        vec.push(3);

        let mut collected = Vec::new();
        vec.clear(|iter| collected.extend(iter));

        assert_eq!(collected, vec![1, 2, 3]);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_thread_safety() {
        let vec = Arc::new(BumpVec::new(1000));
        let mut handles = vec![];

        for i in 0..10 {
            let vec_clone = Arc::clone(&vec);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let value = i * 100 + j;
                    vec_clone.push(value);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(vec.len() <= 1000);

        // Verify that all successfully pushed values can be retrieved
        let mut count = 0;
        for i in 0..vec.len() {
            if vec.get(i).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, vec.len());
    }

    #[test]
    fn test_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct DropCounter;
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        {
            let vec = BumpVec::new(3);
            vec.push(DropCounter);
            vec.push(DropCounter);
            vec.push(DropCounter);
        } // vec gets dropped here

        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_out_of_bounds() {
        let vec = BumpVec::new(3);
        vec.push(1);

        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.get(999), None);
    }

    #[test]
    fn test_capacity() {
        let vec = BumpVec::<i32>::new(0);
        assert_eq!(vec.push(1), None);

        let vec = BumpVec::new(1);
        assert_eq!(vec.push(1), Some(0));
        assert_eq!(vec.push(2), None);

        let vec = BumpVec::new(3);
        assert_eq!(vec.push(1), Some(0));
        assert_eq!(vec.push(2), Some(1));
        assert_eq!(vec.push(3), Some(2));
        assert_eq!(vec.push(4), None);
    }
}
