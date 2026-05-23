# Phase 3A: 类型系统深化 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 2 解释器基础上实现泛型系统、trait 系统和所有权/借用模型，产出类型系统完备的解释器。

**Architecture:** 延续 Pipeline：Lexer → Parser(AST) → HIR Lowering + Type Check → Interpreter。泛型通过"调用处实例化"策略实现（非编译期 monomorphization）。所有权在解释器中动态检查。

**Tech Stack:** Rust 2024 edition, ndarray, rustyline.

---

## 文件结构（Phase 3A 新增/修改）

```
tenth/src/
├── parser/
│   ├── ast.rs          # 新增: GenericParams, TraitDef, Ref/Move 表达式
│   └── parser.rs       # 新增: 解析泛型参数、trait、&/move 语法
├── hir/
│   ├── types.rs        # 新增: TypeParam, RefType, TraitRef 类型变体
│   ├── hir.rs          # 新增: GenericFnDef, TraitDef, HirRef/Move
│   └── lower.rs        # 新增: 泛型实例化、trait 约束检查、借用检查
├── runtime/
│   ├── value.rs        # 新增: Ref/Moved 标记
│   └── interpreter.rs  # 新增: 引用/解引用、所有权检查
```

---

### Task 1: 泛型函数

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/types.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Create: `tenth/tests/generic_test.rs`

**语法目标:**
```tenth
fn identity<T>(x: T) -> T { x }
identity<i32>(42)   // => 42
identity<f64>(3.14) // => 3.14
```

- [ ] **Step 1: 扩展 AST 节点**

在 `ast.rs` 中添加泛型参数定义：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: Ident,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericArgs {
    pub args: Vec<TypeAnnotation>,
}
```

修改 `ItemKind::Function` 添加泛型参数：
```rust
Function {
    name: Ident,
    generics: Vec<GenericParam>,
    params: Vec<Param>,
    return_type: Option<TypeAnnotation>,
    body: Expr,
},
```

在 `ExprKind` 中添加泛型调用：
```rust
GenericCall {
    func: Box<Expr>,
    generics: Vec<TypeAnnotation>,
    args: Vec<Expr>,
},
```

- [ ] **Step 2: Parser 解析泛型语法**

解析 `<T>` 尖括号泛型参数。在 `parse_item` 的 `Fn` 分支中，函数名之后检测 `<`：
```rust
TokenKind::Fn => {
    self.advance();
    let name = self.expect_ident()?;
    let generics = self.parse_generic_params()?;
    self.expect(TokenKind::LParen)?;
    // ... existing param parsing ...
}
```

添加 `parse_generic_params` 方法：
```rust
fn parse_generic_params(&mut self) -> TenthResult<Vec<GenericParam>> {
    if !matches!(self.peek_kind(), TokenKind::Lt) {
        return Ok(Vec::new());
    }
    self.advance();
    let mut params = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::Gt) {
        let name = self.expect_ident()?;
        params.push(GenericParam { name });
        if !matches!(self.peek_kind(), TokenKind::Comma) {
            break;
        }
        self.advance();
    }
    self.expect(TokenKind::Gt)?;
    Ok(params)
}
```

解析泛型调用 `func<T>(args)`。在 parse_postfix 中检测 `<` 后跟类型注解：
```rust
TokenKind::Lt if self.is_generic_args() => {
    self.advance();
    let mut generics = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::Gt) {
        generics.push(self.parse_type()?);
        if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
        self.advance();
    }
    self.expect(TokenKind::Gt)?;
    self.expect(TokenKind::LParen)?;
    let mut args = Vec::new();
    if !matches!(self.peek_kind(), TokenKind::RParen) {
        args = self.parse_arg_list()?;
    }
    self.expect(TokenKind::RParen)?;
    expr = Expr {
        kind: ExprKind::GenericCall { func: Box::new(expr), generics, args },
        span: expr.span.clone(),
    };
}
```

- [ ] **Step 3: 扩展 HIR 类型系统**

在 `types.rs` 中添加：
```rust
pub enum Type {
    // ... existing variants ...
    TypeParam { name: String },
    Ref(Box<Type>),
    MutRef(Box<Type>),
}
```

- [ ] **Step 4: 扩展 HIR 节点**

在 `hir.rs` 中：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirFnDef {
    pub name: String,
    pub generics: Vec<String>,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: HirExpr,
    pub span: Span,
}
```

添加泛型调用：
```rust
// HirExprKind
GenericCall {
    func: Box<HirExpr>,
    generics: Vec<Type>,
    args: Vec<HirExpr>,
    ret_ty: Type,
},
```

- [ ] **Step 5: Lowerer 泛型实例化**

在 `lower_program` 中处理泛型函数定义，记录 generics 信息。

在 `lower_expr` 中处理 `GenericCall`：将类型参数代入函数签名，生成实例化后的 body。

实现 `instantiate_generic` 方法：
```rust
fn instantiate_generic(
    &self, fn_def: &HirFnDef, type_args: &[Type],
) -> TenthResult<(Vec<(String, Type)>, Type, HirExpr)> {
    // 构建类型替换映射
    let mut type_map: HashMap<String, Type> = HashMap::new();
    for (gen, ty_arg) in fn_def.generics.iter().zip(type_args.iter()) {
        type_map.insert(gen.clone(), ty_arg.clone());
    }
    // 替换参数类型
    let params: Vec<(String, Type)> = fn_def.params.iter()
        .map(|(n, t)| (n.clone(), substitute_type(t, &type_map)))
        .collect();
    let ret_ty = substitute_type(&fn_def.return_type, &type_map);
    Ok((params, ret_ty, fn_def.body.clone()))
}
```

- [ ] **Step 6: 解释器支持**

在 `call_named_fn` 中，如果函数有泛型参数，检查调用时是否提供了类型参数（通过 GenericCall 传递）。

修改 `eval_call` 以处理 `GenericCall`：实例化泛型函数后调用。

- [ ] **Step 7: 编写测试**

```rust
#[test]
fn test_generic_identity() {
    let src = "fn identity<T>(x: T) -> T { x }; identity<i32>(42)";
    // assert result is 42
}

#[test]
fn test_generic_function() {
    let src = "fn add<T>(a: T, b: T) -> T { a + b }; add<f64>(1.5, 2.5)";
    // assert result is 4.0
}
```

- [ ] **Step 8: 编译与测试验证**

```bash
cd /workspace/tenth && cargo build && cargo test
```

- [ ] **Step 9: Commit**

---

### Task 2: 泛型结构体

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/value.rs`
- Modify: `tenth/tests/generic_test.rs`

**语法目标:**
```tenth
struct Pair<T, U> { first: T, second: U }
let p = Pair<i32, f64> { first: 42, second: 3.14 };
p.first + p.second
```

- [ ] **Step 1: 扩展 AST**

修改 `ItemKind::StructDef` 添加泛型参数：
```rust
StructDef {
    name: Ident,
    generics: Vec<GenericParam>,
    fields: Vec<StructField>,
},
```

修改 `ExprKind::StructLiteral`：
```rust
StructLiteral {
    name: Ident,
    generics: Vec<TypeAnnotation>,
    fields: Vec<(Ident, Expr)>,
},
```

- [ ] **Step 2: Parser 解析泛型结构体**

在 struct 定义解析中添加 `<T, U>` 参数。

在 struct 字面量解析中支持 `Pair<i32, f64> { }` 语法。

- [ ] **Step 3: HIR 和 Lowerer**

`HirStructDef` 添加 `generics: Vec<String>`。
`StructLiteral` lowering 时代入泛型参数。

- [ ] **Step 4: 解释器**

`Value::Struct` 添加泛型参数信息。

- [ ] **Step 5: 编写测试**

- [ ] **Step 6: 验证 & Commit**

---

### Task 3: Trait 定义与实现

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/types.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Create: `tenth/tests/trait_test.rs`

**语法目标:**
```tenth
trait Add {
    fn add(self, other: Self) -> Self;
}

impl Add for Point {
    fn add(self, other: Point) -> Point {
        Point { x: self.x + other.x, y: self.y + other.y }
    }
}

let p = Point { x: 1.0, y: 2.0 };
p.add(Point { x: 3.0, y: 4.0 })
```

- [ ] **Step 1: 扩展 AST**

在 `ItemKind` 中添加：
```rust
TraitDef {
    name: Ident,
    generics: Vec<GenericParam>,
    methods: Vec<TraitMethod>,
},
```

添加 `TraitMethod`：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<TypeAnnotation>,
}
```

修改 `ItemKind::Impl` 支持 trait：
```rust
Impl {
    type_name: Ident,
    generics: Vec<GenericParam>,
    trait_name: Option<Ident>,
    functions: Vec<Item>,
},
```

- [ ] **Step 2: Parser**

解析 `trait Name { fn method(...) -> ...; }`。
解析 `impl TraitName for TypeName { fn method(...) -> ... { body } }`。

- [ ] **Step 3: 扩展 HIR/Type**

`Type` 中添加 trait 引用（用于类型约束）。

- [ ] **Step 4: Lowerer**

在 `Lowerer` 中维护 trait 方法注册表。
`impl Trait for Type` 将方法注册到 trait 的分发表中。

- [ ] **Step 5: 解释器**

当调用 trait 方法时，通过分发表查找具体实现。
```
p.add(q) → 查找 impl Add for Point → 找到 add 方法 → 调用
```

- [ ] **Step 6: 编写测试**

- [ ] **Step 7: 验证 & Commit**

---

### Task 4: 泛型约束（Trait Bounds）

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/tests/generic_test.rs`

**语法目标:**
```tenth
fn sum<T: Add>(a: T, b: T) -> T { a.add(b) }
```

- [ ] **Step 1: 扩展 AST**

`GenericParam` 添加约束：
```rust
pub struct GenericParam {
    pub name: Ident,
    pub bounds: Vec<Ident>,
}
```

- [ ] **Step 2: Parser 解析 `T: Trait`**

```rust
fn parse_generic_params(&mut self) -> TenthResult<Vec<GenericParam>> {
    // ... existing ...
    let mut bounds = Vec::new();
    if matches!(self.peek_kind(), TokenKind::Colon) {
        self.advance();
        bounds.push(self.expect_ident()?);
        while matches!(self.peek_kind(), TokenKind::Plus) {
            self.advance();
            bounds.push(self.expect_ident()?);
        }
    }
    params.push(GenericParam { name, bounds });
}
```

- [ ] **Step 3: Lowerer 约束检查**

在 `lower_expr` 的 `GenericCall` 处理中，验证传入的类型参数满足 trait 约束。

- [ ] **Step 4: 编写测试**

- [ ] **Step 5: 验证 & Commit**

---

### Task 5: 内置 Trait（Display, Eq, Clone）

**Files:**
- Modify: `tenth/src/runtime/value.rs`
- Modify: `tenth/src/runtime/interpreter.rs`

实现三个内置 trait，无需语法定义：
- `Display`: 自动实现于所有 Value（已有 Display trait for Rust）
- `Eq`: `a == b` 自动支持
- `Clone`: 值自动可复制（当前已是 Clone）

主要是注册这些 trait 名称，使其可以被引用。

- [ ] **Step 1: 注册内置 trait**

在 Lowerer 和 Interpreter 的初始化中注册 `Display`, `Eq`, `Clone`。

- [ ] **Step 2: 验证 & Commit**

---

### Task 6: 引用与借用

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/lexer/token.rs`
- Modify: `tenth/src/lexer/lexer.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/value.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Create: `tenth/tests/ownership_test.rs`

**语法目标:**
```tenth
let x = 42;
let r = &x;       // 不可变引用
println(*r);      // 解引用

let mut y = 10;
let m = &mut y;   // 可变引用
*m = 20;
y                 // => 20
```

- [ ] **Step 1: 添加 `&` 和 `*` token**

确认 `TokenKind::Ref`（`&`）和 `TokenKind::Deref`（`*`）已存在。检查 lexer 中 `&` 和 `*` 的 token 处理。

- [ ] **Step 2: 扩展 AST**

在 `ExprKind` 中添加：
```rust
Ref(Box<Expr>),
MutRef(Box<Expr>),
Deref(Box<Expr>),
```

- [ ] **Step 3: Parser 解析 `&`、`&mut`、`*`**

在 `parse_prefix` 或 `parse_primary` 中：
```rust
TokenKind::Ref => {
    self.advance();
    if matches!(self.peek_kind(), TokenKind::Mut) {
        self.advance();
        ExprKind::MutRef(Box::new(self.parse_primary()?))
    } else {
        ExprKind::Ref(Box::new(self.parse_primary()?))
    }
}
TokenKind::Star => {
    self.advance();
    ExprKind::Deref(Box::new(self.parse_primary()?))
}
```

- [ ] **Step 4: 扩展 HIR**

在 `HirExprKind` 中添加：
```rust
Ref(Box<HirExpr>),
MutRef(Box<HirExpr>),
Deref(Box<HirExpr>),
```

- [ ] **Step 5: Lowerer**

直接传递引用表达式。

- [ ] **Step 6: 解释器**

在 `eval_expr` 中：
```rust
HirExprKind::Ref(inner) => {
    let val = self.eval_expr(inner)?.ok_or_else(...)?;
    Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
}

HirExprKind::Deref(inner) => {
    let val = self.eval_expr(inner)?.ok_or_else(...)?;
    match &val {
        Value::Ref(rc) => Ok(Some(rc.borrow().clone())),
        _ => Err(...),
    }
}
```

在 `eval_stmt` 的 `Let` 中实现借用的所有权语义：`let r = &x` 创建共享引用，`x` 仍可用但不可被移动。通过引用计数跟踪引用状态。

- [ ] **Step 7: 编写测试**

- [ ] **Step 8: 验证 & Commit**

---

### Task 7: 移动语义

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Modify: `tenth/tests/ownership_test.rs`

**语法目标:**
```tenth
let x = Point { x: 1.0, y: 2.0 };
let y = move x;   // 所有权转移
// x 在此之后不可用
y.x               // => 1.0
```

- [ ] **Step 1: AST 扩展**

在 `StmtKind::Let` 或 `ExprKind` 中添加 move 标记。

`move` 关键字解析。

- [ ] **Step 2: Parser**

在 `let` 语句解析中识别 `move` 关键字：
```rust
// let move x = expr;
// 或在表达式中：let y = move x;
```

- [ ] **Step 3: 解释器**

在变量赋值时检查所有权：
- 如果使用 `move`，将值从源变量移出，源变量标记为无效
- 后续对源变量的访问报运行时错误："value moved"

实现方式：在 `variables` HashMap 中使用 `Option<Value>`，移动后将值设为 `None`。

- [ ] **Step 4: 编写测试**

- [ ] **Step 5: 验证 & Commit**

---

### Task 8: borrow checker（基础版）

**Files:**
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Modify: `tenth/tests/ownership_test.rs`

在 Lowerer 中实现基础借用检查：
- 不可变引用 `&x` 不消耗 `x`
- 可变引用 `&mut x` 独占 `x`
- 不可同时存在可变引用和其他引用
- 被 move 的变量不可再使用

- [ ] **Step 1: 实现所有权跟踪**

在 Lowerer 中为每个作用域维护所有权状态表：
```rust
struct OwnershipState {
    // variable_name -> status
    status: HashMap<String, OwnershipStatus>,
}

enum OwnershipStatus {
    Owned,
    SharedRef(usize),  // 引用计数
    ExclusiveRef,
    Moved,
}
```

- [ ] **Step 2: 在 lower_expr 中插入检查**

- 变量使用前检查其所有权状态
- 移动后标记为 Moved
- 创建引用时更新引用计数

- [ ] **Step 3: 编写测试**

验证以下场景报编译期错误：
```
let x = 42;
let y = move x;
x  // error: x has been moved
```

```
let mut x = 42;
let r1 = &x;
let r2 = &mut x;  // error: cannot borrow x as mutable while shared
```

- [ ] **Step 4: 验证 & Commit**

---

### Task 9: 全量验收

- [ ] 运行 `cd /workspace/tenth && cargo test` 验证所有测试通过
- [ ] REPL 验收：泛型函数、trait 方法、引用/解引用
- [ ] 提交最终状态

---

## Phase 3A 完成标准

- [ ] 泛型函数：`fn id<T>(x:T)->T` 与 `id<i32>(42)` 调用
- [ ] 泛型结构体：`struct Pair<T,U>` 与 `Pair<i32,f64>{...}`
- [ ] trait 定义：`trait Add { fn add(self, other:Self)->Self }`
- [ ] trait 实现：`impl Add for Point { ... }` 与 `.add()` 分发
- [ ] trait 约束：`fn sum<T:Add>(a:T, b:T)->T`
- [ ] 引用：`&x`, `&mut x`, `*r`
- [ ] 移动：`let y = move x` 后 x 不可用
- [ ] 基础借用检查：不可同时 mutable + shared
- [ ] 所有测试通过（预计 50+ 个测试）