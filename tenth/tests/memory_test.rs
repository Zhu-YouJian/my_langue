/// Memory / resource-limit tests.
///
/// Run with:
///   cargo test --features mem-debug memory
///   cargo test --features mem-strict memory
///
/// For leak checking on Linux/macOS:
///   cargo test --features mem-debug memory -- --nocapture
///   valgrind --leak-check=full ./target/debug/tenth_test

#[cfg(test)]
mod limits_tests {
    use tenth::runtime::limits::*;
    use tenth::runtime::arena::Arena;

    // ── MemoryConfig ───────────────────────────────────────────────────

    #[test]
    fn defaults_are_sensible() {
        let cfg = MemoryConfig::default();
        assert!(cfg.max_arena_bytes > 0);
        assert!(cfg.max_variables > 0);
        assert!(cfg.max_tensor_elements > 0);
    }

    #[test]
    fn test_small_is_tight() {
        let cfg = MemoryConfig::test_small();
        assert_eq!(cfg.max_arena_bytes, 16 * 1024 * 1024);
        assert_eq!(cfg.max_variables, 1_000);
        assert_eq!(cfg.max_tensor_elements, 4 * 1024 * 1024);
    }

    #[test]
    fn unbounded_allows_large() {
        let cfg = MemoryConfig::unbounded();
        assert!(cfg.max_arena_bytes > 1024 * 1024 * 1024);
    }

    // ── Guard checks ───────────────────────────────────────────────────

    #[test]
    fn test_arena_under_limit() {
        let cfg = MemoryConfig { max_arena_bytes: 1024, ..MemoryConfig::default() };
        assert!(check_arena_alloc(&cfg, 0, 512).is_ok());
        assert!(check_arena_alloc(&cfg, 512, 512).is_ok());
    }

    #[test]
    fn test_arena_over_limit() {
        let cfg = MemoryConfig { max_arena_bytes: 1024, ..MemoryConfig::default() };
        assert!(check_arena_alloc(&cfg, 0, 2048).is_err());
        assert!(check_arena_alloc(&cfg, 1000, 100).is_err());
    }

    #[test]
    fn test_var_count_limit() {
        let cfg = MemoryConfig { max_variables: 10, ..MemoryConfig::default() };
        assert!(check_var_count(&cfg, 5).is_ok());
        assert!(check_var_count(&cfg, 9).is_ok());
        assert!(check_var_count(&cfg, 10).is_err());
        assert!(check_var_count(&cfg, 100).is_err());
    }

    #[test]
    fn test_def_count_limit() {
        let cfg = MemoryConfig { max_accumulated_defs: 100, ..MemoryConfig::default() };
        assert!(check_def_count(&cfg, 50).is_ok());
        assert!(check_def_count(&cfg, 100).is_err());
    }

    #[test]
    fn test_tensor_elements_limit() {
        let cfg = MemoryConfig { max_tensor_elements: 1024, ..MemoryConfig::default() };
        assert!(check_tensor_elements(&cfg, 512).is_ok());
        assert!(check_tensor_elements(&cfg, 1024).is_ok());
        assert!(check_tensor_elements(&cfg, 1025).is_err());
    }

    // ── RuntimeLimits wrapper ──────────────────────────────────────────

    #[test]
    fn runtime_limits_guard_methods() {
        let limits = RuntimeLimits::test_small();
        assert!(limits.guard_arena(0, 1024).is_ok());
        assert!(limits.guard_vars(500).is_ok());
        assert!(limits.guard_defs(200).is_ok());
        // Exceed each
        assert!(limits.guard_arena(16 * 1024 * 1024 + 1, 1).is_err());
        assert!(limits.guard_vars(1001).is_err());
        assert!(limits.guard_defs(501).is_err());
    }

    // ── Arena + limits integration ─────────────────────────────────────

    #[test]
    fn arena_tracks_bytes() {
        LiveCounter::reset();
        let snap0 = LiveCounter::snapshot();
        // Note: global counters may be affected by parallel tests.
        // We check the delta, not absolute value.

        let mut arena = Arena::new(1024);
        let before = LiveCounter::snapshot().arena_alloc_bytes;
        let s = arena.alloc(100).unwrap();
        // 100 f64 = 800 bytes
        let after = LiveCounter::snapshot().arena_alloc_bytes;
        assert_eq!(after - before, 800);

        arena.write(&s, &vec![1.0; 100]);
        assert_eq!(arena.get(&s)[0], 1.0);
    }

    #[test]
    fn arena_reset_decrements_counter() {
        LiveCounter::reset();
        let mut arena = Arena::new(1024);
        let before = LiveCounter::snapshot().arena_alloc_bytes;
        arena.alloc(50).unwrap();   // 400 bytes
        arena.alloc(50).unwrap();   // 400 bytes
        let after_alloc = LiveCounter::snapshot().arena_alloc_bytes;
        assert_eq!(after_alloc - before, 800);

        arena.reset();
        let after_reset = LiveCounter::snapshot().arena_alloc_bytes;
        assert!(after_reset <= before + 10, "reset should bring counter back near baseline");
    }

    #[test]
    fn arena_scope_rolls_back_counter() {
        LiveCounter::reset();
        let mut arena = Arena::new(1024);
        let base = LiveCounter::snapshot().arena_alloc_bytes;
        let a1 = arena.alloc(10).unwrap(); // 80 bytes permanent
        let after_perm = LiveCounter::snapshot().arena_alloc_bytes;
        assert_eq!(after_perm - base, 80);

        arena.scope(|a| {
            let inner_base = LiveCounter::snapshot().arena_alloc_bytes;
            a.alloc(20).unwrap(); // 160 bytes temporary
            let inner_after = LiveCounter::snapshot().arena_alloc_bytes;
            assert_eq!(inner_after - inner_base, 160);
        });

        // After scope, counter should roll back to permanent-only level
        let after_scope = LiveCounter::snapshot().arena_alloc_bytes;
        assert_eq!(after_scope, after_perm);
    }

    #[test]
    fn arena_overflow_returns_none() {
        let mut arena = Arena::new(5);
        assert!(arena.alloc(6).is_none());
        assert!(arena.alloc(5).is_some());
    }
}

#[cfg(test)]
mod interpreter_limits_tests {
    use tenth::runtime::interpreter::Interpreter;
    use tenth::runtime::limits::*;
    use tenth::hir::hir::*;
    use std::collections::HashMap;

    fn empty_program() -> HirProgram {
        HirProgram {
            functions: vec![],
            generic_funcs: vec![],
            main_expr: None,
            modules: HashMap::new(),
            uses: vec![],
            methods: HashMap::new(),
            structs: HashMap::new(),
            generic_structs: HashMap::new(),
            enums: HashMap::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
        }
    }

    #[test]
    fn interpreter_accepts_limits() {
        let prog = empty_program();
        let limits = RuntimeLimits::test_small();
        let interp = Interpreter::with_limits(&prog, limits);
        assert!(interp.limits.is_some());
    }

    #[test]
    fn interpreter_without_limits_has_none() {
        let prog = empty_program();
        let interp = Interpreter::new(&prog);
        assert!(interp.limits.is_none());
    }

    #[test]
    fn make_tensor_within_limits() {
        let prog = empty_program();
        let limits = RuntimeLimits::new(MemoryConfig {
            max_tensor_elements: 1000,
            ..MemoryConfig::test_small()
        });
        let interp = Interpreter::with_limits(&prog, limits);
        // 10 elements is fine
        let tensor = interp.make_tensor(vec![1.0; 10], vec![10]);
        assert!(tensor.is_ok());
    }

    #[test]
    fn make_tensor_exceeds_limits() {
        let prog = empty_program();
        let limits = RuntimeLimits::new(MemoryConfig {
            max_tensor_elements: 5,
            ..MemoryConfig::test_small()
        });
        let interp = Interpreter::with_limits(&prog, limits);
        let tensor = interp.make_tensor(vec![1.0; 100], vec![100]);
        assert!(tensor.is_err());
    }
}
