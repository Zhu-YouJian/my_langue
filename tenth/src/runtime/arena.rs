use super::limits;

/// Arena allocator for tensor data.
/// Pre-allocates a pool of f64 values and hands out slices.
/// All allocations are batch-freed when the arena is reset.
///
/// When compiled with `mem-debug` feature, arena allocations update
/// the global live counters so that `:mem` in the REPL reflects
/// actual arena usage.
pub struct Arena {
    pool: Vec<f64>,
    offset: usize,
    capacity: usize,
    /// Tracked bytes for limits integration
    tracked_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaSlice {
    /// Offset into the arena pool
    pub offset: usize,
    /// Number of f64 elements
    pub len: usize,
}

impl Arena {
    /// Create a new arena with the given capacity (number of f64 elements).
    pub fn new(capacity: usize) -> Self {
        Arena {
            pool: vec![0.0; capacity],
            offset: 0,
            capacity,
            tracked_bytes: 0,
        }
    }

    /// Allocate `count` f64 elements from the arena.
    /// Returns None if insufficient space.
    pub fn alloc(&mut self, count: usize) -> Option<ArenaSlice> {
        let start = self.offset;
        if start + count > self.capacity {
            return None;
        }
        self.offset = start + count;
        let byte_size = count * std::mem::size_of::<f64>();
        self.tracked_bytes += byte_size;
        limits::inc_arena_bytes(byte_size);
        Some(ArenaSlice { offset: start, len: count })
    }

    /// Write data into an allocated slice.
    pub fn write(&mut self, slice: &ArenaSlice, data: &[f64]) {
        let start = slice.offset;
        let end = start + slice.len;
        assert!(data.len() == slice.len, "data length mismatch");
        self.pool[start..end].copy_from_slice(data);
    }

    /// Get a mutable reference to the data at the given slice.
    pub fn get_mut(&mut self, slice: &ArenaSlice) -> &mut [f64] {
        let start = slice.offset;
        &mut self.pool[start..start + slice.len]
    }

    /// Get a shared reference to the data at the given slice.
    pub fn get(&self, slice: &ArenaSlice) -> &[f64] {
        let start = slice.offset;
        &self.pool[start..start + slice.len]
    }

    /// Reset the arena, freeing all allocations.
    /// Also decrements the global counter.
    pub fn reset(&mut self) {
        limits::dec_arena_bytes(self.tracked_bytes);
        self.tracked_bytes = 0;
        self.offset = 0;
    }

    /// Run a closure within an arena scope. All allocations within
    /// the scope are automatically freed when the scope exits.
    pub fn scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Arena) -> R,
    {
        let saved = self.offset;
        let saved_tracked = self.tracked_bytes;
        let result = f(self);
        // Roll back the global counter for scoped allocations
        let diff = self.tracked_bytes - saved_tracked;
        limits::dec_arena_bytes(diff);
        self.tracked_bytes = saved_tracked;
        self.offset = saved;
        result
    }

    /// Bytes used / total capacity.
    pub fn usage(&self) -> (usize, usize) {
        (self.offset * std::mem::size_of::<f64>(), self.capacity * std::mem::size_of::<f64>())
    }

    /// Number of remaining f64 slots.
    pub fn remaining(&self) -> usize {
        self.capacity - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc_and_read() {
        let mut arena = Arena::new(1024);
        let s = arena.alloc(10).unwrap();
        assert_eq!(s.len, 10);
        assert_eq!(s.offset, 0);

        arena.write(&s, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(arena.get(&s)[0], 1.0);
        assert_eq!(arena.get(&s)[9], 10.0);
    }

    #[test]
    fn test_arena_scope_reset() {
        let mut arena = Arena::new(1024);
        let s1 = arena.alloc(10).unwrap();
        assert_eq!(s1.offset, 0);

        arena.scope(|a| {
            let s2 = a.alloc(20).unwrap();
            assert_eq!(s2.offset, 10);
            a.write(&s2, &vec![42.0; 20]);
        });

        // After scope, offset should be back to 10
        let s3 = arena.alloc(10).unwrap();
        assert_eq!(s3.offset, 10);
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = Arena::new(5);
        assert!(arena.alloc(6).is_none());
        assert!(arena.alloc(5).is_some());
    }

    #[test]
    fn test_arena_usage() {
        let mut arena = Arena::new(100);
        assert_eq!(arena.remaining(), 100);
        arena.alloc(30).unwrap();
        assert_eq!(arena.remaining(), 70);
    }
}
