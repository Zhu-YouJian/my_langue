//! Runtime resource limits — a safety net for memory management.
//!
//! All memory-affecting operations pass through this layer so that
//! runaway allocations, REPL bloat, and unbounded growth can be
//! detected early and either capped or reported.
//!
//! In `mem-strict` mode (feature flag), limits are enforced as hard errors.
//! In `mem-debug` mode, every allocation is tracked with a counter.
//! In default (release) mode, limits are soft warnings only.

use std::sync::atomic::{AtomicUsize, Ordering};

// ── Configuration ──────────────────────────────────────────────────────────

/// Tunable limits for a Tenth runtime session.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum total bytes the arena can hand out (tensor data).
    /// Default: 256 MiB
    pub max_arena_bytes: usize,

    /// Maximum number of variables tracked in a REPL session.
    /// Default: 10_000
    pub max_variables: usize,

    /// Maximum accumulated function / method / trait definitions in REPL.
    /// Default: 5_000
    pub max_accumulated_defs: usize,

    /// Maximum number of f64 elements in a single tensor allocation.
    /// Default: 256 × 1024 × 1024  (~2 GiB)
    pub max_tensor_elements: usize,

    /// When true, every arena alloc / tensor creation increments a global
    /// counter visible via `LiveCounter::snapshot()`.
    pub track_allocations: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            max_arena_bytes: 256 * 1024 * 1024,       // 256 MiB
            max_variables: 10_000,
            max_accumulated_defs: 5_000,
            max_tensor_elements: 256 * 1024 * 1024,   // ~2 GiB (f64)
            track_allocations: cfg!(feature = "mem-debug"),
        }
    }
}

impl MemoryConfig {
    /// Tiny footprint for unit tests.
    pub fn test_small() -> Self {
        MemoryConfig {
            max_arena_bytes: 16 * 1024 * 1024,        // 16 MiB
            max_variables: 1_000,
            max_accumulated_defs: 500,
            max_tensor_elements: 4 * 1024 * 1024,     // 4M elements
            track_allocations: true,
        }
    }

    /// No practical limits (use with care).
    pub fn unbounded() -> Self {
        MemoryConfig {
            max_arena_bytes: usize::MAX,
            max_variables: usize::MAX,
            max_accumulated_defs: usize::MAX,
            max_tensor_elements: usize::MAX,
            track_allocations: true,
        }
    }
}

// ── Live counters (lock-free for minimal overhead) ─────────────────────────

/// Global allocation counters.
pub struct LiveCounter;

impl LiveCounter {
    pub fn snapshot() -> Counters {
        Counters {
            arena_alloc_bytes: ARENA_BYTES.load(Ordering::Relaxed),
            tensor_count: TENSOR_COUNT.load(Ordering::Relaxed),
            variable_count: VAR_COUNT.load(Ordering::Relaxed),
        }
    }

    pub fn reset() {
        ARENA_BYTES.store(0, Ordering::Relaxed);
        TENSOR_COUNT.store(0, Ordering::Relaxed);
        VAR_COUNT.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Counters {
    pub arena_alloc_bytes: usize,
    pub tensor_count: usize,
    pub variable_count: usize,
}

// ── Atomic counters (feature-gated writes) ─────────────────────────────────

static ARENA_BYTES: AtomicUsize = AtomicUsize::new(0);
static TENSOR_COUNT: AtomicUsize = AtomicUsize::new(0);
static VAR_COUNT: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn inc_arena_bytes(n: usize) {
    ARENA_BYTES.fetch_add(n, Ordering::Relaxed);
}

#[inline]
pub fn dec_arena_bytes(n: usize) {
    ARENA_BYTES.fetch_sub(n, Ordering::Relaxed);
}

#[inline]
pub fn inc_tensor_count() {
    TENSOR_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn dec_tensor_count() {
    TENSOR_COUNT.fetch_sub(1, Ordering::Relaxed);
}

#[inline]
pub fn inc_var_count() {
    VAR_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn dec_var_count() {
    VAR_COUNT.fetch_sub(1, Ordering::Relaxed);
}

// ── Guard checks ───────────────────────────────────────────────────────────

/// Check that `requested` arena bytes won't exceed the config cap.
/// Returns `Ok(())` or a descriptive error.
pub fn check_arena_alloc(config: &MemoryConfig, current: usize, requested: usize) -> Result<(), String> {
    if current.saturating_add(requested) > config.max_arena_bytes {
        return Err(format!(
            "arena limit exceeded: {}/{} bytes (requested +{})",
            current, config.max_arena_bytes, requested
        ));
    }
    Ok(())
}

/// Check that the number of variables stays within bounds.
pub fn check_var_count(config: &MemoryConfig, current: usize) -> Result<(), String> {
    if current >= config.max_variables {
        return Err(format!(
            "variable count limit reached: {} (max {})",
            current, config.max_variables
        ));
    }
    Ok(())
}

/// Check that the accumulated definitions count stays within bounds.
pub fn check_def_count(config: &MemoryConfig, current: usize) -> Result<(), String> {
    if current >= config.max_accumulated_defs {
        return Err(format!(
            "definition limit reached: {} (max {})",
            current, config.max_accumulated_defs
        ));
    }
    Ok(())
}

/// Check tensor element count before allocation.
pub fn check_tensor_elements(config: &MemoryConfig, elements: usize) -> Result<(), String> {
    if elements > config.max_tensor_elements {
        return Err(format!(
            "tensor too large: {} elements (max {})",
            elements, config.max_tensor_elements
        ));
    }
    Ok(())
}

// ── Convenience wrapper ────────────────────────────────────────────────────

/// Owned guard that holds a `MemoryConfig` and can be passed through the
/// interpreter / REPL to enforce limits uniformly.
#[derive(Debug, Clone)]
pub struct RuntimeLimits {
    pub config: MemoryConfig,
}

impl RuntimeLimits {
    pub fn new(config: MemoryConfig) -> Self {
        if config.track_allocations {
            LiveCounter::reset();
        }
        RuntimeLimits { config }
    }

    pub fn default_strict() -> Self {
        RuntimeLimits::new(MemoryConfig::default())
    }

    pub fn test_small() -> Self {
        RuntimeLimits::new(MemoryConfig::test_small())
    }

    /// Returns the current live counters snapshot.
    pub fn snapshot(&self) -> Counters {
        LiveCounter::snapshot()
    }

    /// Check arena allocation against configured limit.
    pub fn guard_arena(&self, current_bytes: usize, requested: usize) -> Result<(), String> {
        check_arena_alloc(&self.config, current_bytes, requested)
    }

    /// Check variable count.
    pub fn guard_vars(&self, current: usize) -> Result<(), String> {
        check_var_count(&self.config, current)
    }

    /// Check definition count.
    pub fn guard_defs(&self, current: usize) -> Result<(), String> {
        check_def_count(&self.config, current)
    }

    /// Check tensor element count.
    pub fn guard_tensor(&self, elements: usize) -> Result<(), String> {
        check_tensor_elements(&self.config, elements)
    }
}
