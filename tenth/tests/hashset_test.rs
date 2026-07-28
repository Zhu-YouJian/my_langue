//! HashSet 行为测试。
//!
//! 验证 std/collections/hashset.th 中所有函数的语义正确性：
//! - 基础：new / insert / contains / len / is_empty / remove / to_array / clear
//! - 集合运算：from_array / set_union / intersection / difference / is_subset
//!
//! 运行时不支持 `use` 加载 .th 模块（参考 date_test.rs / duration_test.rs 模式），
//! 故在此内联 hashset.th 的全部实现，与 tenth/std/collections/hashset.th 保持同步。

use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

/// Run source through lexer → parser → HIR → interpreter.
fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// 内联 std/collections/hashset.th 的全部实现，供测试调用。
/// 与 tenth/std/collections/hashset.th 保持同步。
const HASHSET_HELPERS: &str = r#"
    struct HashSet { inner: HashMap }

    fn new() -> HashSet {
        HashSet { inner: HashMap::new() }
    }

    fn insert(set: HashSet, value) -> HashSet {
        set.inner.insert(value, ());
        set
    }

    fn contains(set: HashSet, value) -> bool {
        set.inner.contains_key(value)
    }

    fn remove(set: HashSet, value) -> HashSet {
        set.inner.remove(value);
        set
    }

    fn len(set: HashSet) -> i64 {
        set.inner.len()
    }

    fn is_empty(set: HashSet) -> bool {
        set.inner.is_empty()
    }

    fn to_array(set: HashSet) -> Vec {
        set.inner.keys()
    }

    fn clear(set: HashSet) -> HashSet {
        HashSet { inner: HashMap::new() }
    }

    fn from_array(arr: Vec) -> HashSet {
        let n = arr.len();
        let mut set = new();
        for i in 0..n {
            set = insert(set, arr.get(i));
        };
        set
    }

    fn set_union(a: HashSet, b: HashSet) -> HashSet {
        let arr_a = to_array(a);
        let arr_b = to_array(b);
        let mut result = new();
        let na = arr_a.len();
        for i in 0..na {
            result = insert(result, arr_a.get(i));
        };
        let nb = arr_b.len();
        for i in 0..nb {
            result = insert(result, arr_b.get(i));
        };
        result
    }

    fn intersection(a: HashSet, b: HashSet) -> HashSet {
        let arr_a = to_array(a);
        let na = arr_a.len();
        let mut result = new();
        for i in 0..na {
            let v = arr_a.get(i);
            if contains(b, v) {
                result = insert(result, v);
            };
        };
        result
    }

    fn difference(a: HashSet, b: HashSet) -> HashSet {
        let arr_a = to_array(a);
        let na = arr_a.len();
        let mut result = new();
        for i in 0..na {
            let v = arr_a.get(i);
            if !contains(b, v) {
                result = insert(result, v);
            };
        };
        result
    }

    fn is_subset(a: HashSet, b: HashSet) -> bool {
        let arr_a = to_array(a);
        let na = arr_a.len();
        let mut all_in = true;
        for i in 0..na {
            if !contains(b, arr_a.get(i)) {
                all_in = false;
            };
        };
        all_in
    }
"#;

/// 取解释器返回值中的 i64。
fn as_i64(v: Option<Value>) -> i64 {
    match v {
        Some(Value::Int(n, _)) => n,
        other => panic!("期望 Some(Int(_))，got {:?}", other),
    }
}

/// 取解释器返回值中的 bool。
fn as_bool(v: Option<Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => b,
        other => panic!("期望 Some(Bool(_))，got {:?}", other),
    }
}

// ─── 基础函数：new / insert / contains ──────────────────────────────────

#[test]
fn test_hashset_new_insert_contains() {
    let src = format!(
        r#"{}
        let s = new();
        s = insert(s, "a");
        s = insert(s, "b");
        s = insert(s, "c");
        let has_a = contains(s, "a");
        let has_d = contains(s, "d");
        has_a && !has_d
        "#,
        HASHSET_HELPERS
    );
    assert!(as_bool(run_code(&src).unwrap()), "contains(a)=true, contains(d)=false");
}

// ─── len / is_empty ───────────────────────────────────────────────────

#[test]
fn test_hashset_len_is_empty_empty() {
    // 空 set: len==0, is_empty==true
    let src = format!("{}\n        is_empty(new())", HASHSET_HELPERS);
    assert!(as_bool(run_code(&src).unwrap()), "empty set is_empty should be true");

    let src = format!("{}\n        len(new())", HASHSET_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), 0, "empty set len should be 0");
}

#[test]
fn test_hashset_len_is_empty_nonempty() {
    // 插入后: len==1, is_empty==false
    let src = format!(
        r#"{}
        let s = new();
        s = insert(s, "x");
        is_empty(s)
        "#,
        HASHSET_HELPERS
    );
    assert!(!as_bool(run_code(&src).unwrap()), "non-empty set is_empty should be false");

    let src = format!(
        r#"{}
        let s = new();
        s = insert(s, "x");
        len(s)
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 1, "1-element set len should be 1");
}

// ─── remove ──────────────────────────────────────────────────────────

#[test]
fn test_hashset_remove() {
    let src = format!(
        r#"{}
        let s = new();
        s = insert(s, "a");
        s = remove(s, "a");
        let still_has = contains(s, "a");
        let n = len(s);
        !still_has && n == 0
        "#,
        HASHSET_HELPERS
    );
    assert!(as_bool(run_code(&src).unwrap()), "after remove(a): contains==false, len==0");
}

// ─── to_array ────────────────────────────────────────────────────────

#[test]
fn test_hashset_to_array_len() {
    let src = format!(
        r#"{}
        let s = new();
        s = insert(s, "a");
        s = insert(s, "b");
        s = insert(s, "c");
        let arr = to_array(s);
        arr.len()
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 3, "to_array len should be 3");
}

// ─── clear ───────────────────────────────────────────────────────────

#[test]
fn test_hashset_clear() {
    let src = format!(
        r#"{}
        let s = new();
        s = insert(s, "a");
        s = insert(s, "b");
        s = clear(s);
        len(s)
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 0, "after clear: len==0");
}

// ─── from_array（去重）──────────────────────────────────────────────

#[test]
fn test_hashset_from_array_dedup() {
    // 从 ["a","b","a","c"] 创建，len==3（去重）
    let src = format!(
        r#"{}
        let arr = Vec::new();
        arr.push("a");
        arr.push("b");
        arr.push("a");
        arr.push("c");
        let s = from_array(arr);
        len(s)
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 3, "from_array should dedup to 3");
}

// ─── union ───────────────────────────────────────────────────────────

#[test]
fn test_hashset_union() {
    // {1,2,3} ∪ {3,4,5} == {1,2,3,4,5}, len==5
    let src = format!(
        r#"{}
        let a_arr = Vec::new();
        a_arr.push(1);
        a_arr.push(2);
        a_arr.push(3);
        let b_arr = Vec::new();
        b_arr.push(3);
        b_arr.push(4);
        b_arr.push(5);
        let a = from_array(a_arr);
        let b = from_array(b_arr);
        let u = set_union(a, b);
        len(u)
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 5, "set_union {{1,2,3}} ∪ {{3,4,5}} should have 5 elements");
}

// ─── intersection ────────────────────────────────────────────────────

#[test]
fn test_hashset_intersection() {
    // {1,2,3} ∩ {3,4,5} == {3}, len==1
    let src = format!(
        r#"{}
        let a_arr = Vec::new();
        a_arr.push(1);
        a_arr.push(2);
        a_arr.push(3);
        let b_arr = Vec::new();
        b_arr.push(3);
        b_arr.push(4);
        b_arr.push(5);
        let a = from_array(a_arr);
        let b = from_array(b_arr);
        let i = intersection(a, b);
        len(i)
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 1, "intersection {{1,2,3}} ∩ {{3,4,5}} should have 1 element");
}

// ─── difference ──────────────────────────────────────────────────────

#[test]
fn test_hashset_difference() {
    // {1,2,3} - {3,4,5} == {1,2}, len==2
    let src = format!(
        r#"{}
        let a_arr = Vec::new();
        a_arr.push(1);
        a_arr.push(2);
        a_arr.push(3);
        let b_arr = Vec::new();
        b_arr.push(3);
        b_arr.push(4);
        b_arr.push(5);
        let a = from_array(a_arr);
        let b = from_array(b_arr);
        let d = difference(a, b);
        len(d)
        "#,
        HASHSET_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 2, "difference {{1,2,3}} - {{3,4,5}} should have 2 elements");
}

// ─── is_subset ───────────────────────────────────────────────────────

#[test]
fn test_hashset_is_subset_true() {
    // {1,2} ⊆ {1,2,3} == true
    let src = format!(
        r#"{}
        let a_arr = Vec::new();
        a_arr.push(1);
        a_arr.push(2);
        let b_arr = Vec::new();
        b_arr.push(1);
        b_arr.push(2);
        b_arr.push(3);
        let a = from_array(a_arr);
        let b = from_array(b_arr);
        is_subset(a, b)
        "#,
        HASHSET_HELPERS
    );
    assert!(as_bool(run_code(&src).unwrap()), "{{1,2}} ⊆ {{1,2,3}} should be true");
}

#[test]
fn test_hashset_is_subset_false() {
    // {1,2,3} ⊆ {1,2} == false
    let src = format!(
        r#"{}
        let a_arr = Vec::new();
        a_arr.push(1);
        a_arr.push(2);
        a_arr.push(3);
        let b_arr = Vec::new();
        b_arr.push(1);
        b_arr.push(2);
        let a = from_array(a_arr);
        let b = from_array(b_arr);
        is_subset(a, b)
        "#,
        HASHSET_HELPERS
    );
    assert!(!as_bool(run_code(&src).unwrap()), "{{1,2,3}} ⊆ {{1,2}} should be false");
}
