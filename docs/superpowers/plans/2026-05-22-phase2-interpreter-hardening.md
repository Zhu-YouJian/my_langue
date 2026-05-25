# Phase 2: 夯实解释器 实施计划

> **✅ COMPLETED** — 2026-05-22 ~ 2026-05-22  
> 交付物：StructDef/EnumDef/Impl/Match 全管线打通（AST→HIR→Interpreter），模块系统 + use 导入，泛型函数/泛型结构体基础（GenericParam/GenericCall），Trait 定义与 impl 方法分发，内置 trait (Display/Eq/Clone)。  
> 4 项 struct_test + 5 项 enum_test + 5 项 generic_test + 4 项 trait_test + 2 项 module_test，全部通过。  
> 偏差：实际交付超出计划——泛型、trait、bounds 也在 Phase 2 中实现（计划在 Phase 3A）。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 树遍历解释器基础上，补齐 Tenth 语言的核心类型系统（struct/enum/match/impl）、模块系统和标准库，使其达到可用于编写非平凡程序的成熟度。

**Architecture:** 延续 Phase 1 的管线：Lexer → Parser(AST) → HIR Lowering + Type Check → Interpreter。每个新特性沿完整管线打通。全 CPU 解释执行，不涉及 MLIR/LLVM。

**Tech Stack:** Rust 2024 edition, `ndarray` 用于张量计算, `rustyline` 用于 REPL 交互。`rand` + `rand_distr` 用于随机张量。

**自举说明:** Phase 2 完成后，解释器应能运行 Tenth 编写的自举编译器雏形（v1 的词法/语法分析器）。

---

## 文件结构（Phase 2 新增/修改）

```
tenth/
├── src/
│   ├── parser/
│   │   ├── ast.rs              # 新增: StructDef, EnumDef, Impl, Match, Pattern 等节点
│   │   └── parser.rs           # 新增: parse_struct/parse_enum/parse_match/parse_impl
│   ├── hir/
│   │   ├── types.rs            # 新增: Named/Struct/Enum 类型变体
│   │   ├── hir.rs              # 新增: StructLit, EnumLit, Match, HirStructDef 等
│   │   └── lower.rs            # 新增: lower_struct/lower_enum/lower_match/lower_impl
│   ├── runtime/
│   │   ├── value.rs            # 新增: Struct/Enum 运行时值变体
│   │   └── interpreter.rs      # 新增: eval_struct/eval_match/eval_enum
│   ├── stdlib/                 # 新增目录: 标准库模块
│   │   ├── mod.rs
│   │   ├── tensor_ops.rs       # 扩展张量操作 (matmul, softmax, etc.)
│   │   └── random.rs           # 随机数生成 (rand, randn)
│   └── lib.rs                  # 修改: 导出 stdlib 模块
```

---

### Task 1: 结构体定义与实例化

**Files:**
- Modify: `tenth/src/parser/ast.rs` (~5行新增)
- Modify: `tenth/src/parser/parser.rs` (~80行新增)
- Modify: `tenth/src/hir/types.rs` (~10行新增)
- Modify: `tenth/src/hir/hir.rs` (~25行新增)
- Modify: `tenth/src/hir/lower.rs` (~50行新增)
- Modify: `tenth/src/runtime/value.rs` (~15行新增)
- Modify: `tenth/src/runtime/interpreter.rs` (~30行新增)
- Create: `tenth/tests/struct_test.rs`

**语法目标:**
```tenth
struct Point {
    x: f64,
    y: f64,
}

let p = Point { x: 3.0, y: 4.0 };
p.x + p.y  // => 7.0
```

- [ ] **Step 1: 扩展 AST 节点**

在 `tenth/src/parser/ast.rs` 的 `ItemKind` 中新增：
```rust
StructDef {
    name: Ident,
    fields: Vec<StructField>,
},
```

在 `ItemKind` 上方新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
}
```

在 `ExprKind` 中新增：
```rust
StructLiteral {
    name: Ident,
    fields: Vec<(Ident, Expr)>,
},
```

- [ ] **Step 2: 实现 Parser 解析结构体定义**

在 `tenth/src/parser/parser.rs` 的 `parse_item` 方法中添加 `TokenKind::Struct` 分支：

```rust
TokenKind::Struct => {
    self.advance();
    let name = self.expect_ident()?;
    self.expect(TokenKind::LBrace)?;
    let mut fields = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::RBrace) {
        let field_name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let type_ann = self.parse_type_annotation()?;
        fields.push(StructField { name: field_name, type_ann });
        if !matches!(self.peek_kind(), TokenKind::Comma) {
            break;
        }
        self.advance();
    }
    self.expect(TokenKind::RBrace)?;
    Ok(Item { kind: ItemKind::StructDef { name, fields }, span })
}
```

- [ ] **Step 3: 实现 Parser 解析结构体字面量**

在 `parse_primary` 中：当看到 `TokenKind::Identifier(name)` 且后跟 `TokenKind::LBrace` 时，解析结构体字面量：

```rust
TokenKind::Identifier(ref name) if self.peek_next_is(TokenKind::LBrace) => {
    let ident = Ident { name: name.clone(), span: token.span.clone() };
    self.advance(); // skip ident
    self.advance(); // skip {
    let mut fields = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::RBrace) {
        let field_name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let value = self.parse_expr()?;
        fields.push((field_name, value));
        if !matches!(self.peek_kind(), TokenKind::Comma) {
            break;
        }
        self.advance();
    }
    self.expect(TokenKind::RBrace)?;
    ExprKind::StructLiteral { name: ident, fields }
}
```

需要在 `Parser` 中添加辅助方法 `peek_next_is`：
```rust
fn peek_next_is(&self, kind: TokenKind) -> bool {
    self.tokens.get(self.pos + 1).map(|t| t.kind == kind).unwrap_or(false)
}
```

- [ ] **Step 4: 扩展 HIR 类型系统**

在 `tenth/src/hir/types.rs` 的 `Type` 枚举中新增：
```rust
Struct {
    name: String,
    fields: Vec<(String, Type)>,
},
```

添加构造函数：
```rust
pub fn struct_(name: &str, fields: Vec<(String, Type)>) -> Self {
    Type::Struct { name: name.to_string(), fields }
}
```

- [ ] **Step 5: 扩展 HIR 节点**

在 `tenth/src/hir/hir.rs` 的 `HirExprKind` 中新增：
```rust
StructLiteral {
    name: String,
    fields: Vec<(String, HirExpr)>,
    ty: Type,
},
```

在文件末尾新增结构体定义 HIR 节点：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirStructDef {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirProgram {
    pub functions: Vec<HirFnDef>,
    pub structs: Vec<HirStructDef>,
    pub main_expr: Option<HirExpr>,
}
```

- [ ] **Step 6: 实现 Lowerer 处理结构体**

在 `tenth/src/hir/lower.rs` 的 `lower_program` 中，先收集所有结构体定义：

```rust
pub fn lower_program(&mut self, program: &Program) -> TenthResult<HirProgram> {
    // First pass: collect struct definitions
    for item in &program.items {
        if let ItemKind::StructDef { name, fields } = &item.kind {
            let hir_fields: Vec<(String, Type)> = fields.iter()
                .map(|f| (f.name.name.clone(), Type::from_annotation(&f.type_ann)))
                .collect();
            self.structs.insert(name.name.clone(), hir_fields.clone());
        }
    }
    // ... rest of lowering
}
```

在 `Lowerer` 结构体中新增：
```rust
structs: HashMap<String, Vec<(String, Type)>>,
```

在 `lower_expr` 中添加：
```rust
ExprKind::StructLiteral { name, fields } => {
    let lowered_fields: Vec<(String, HirExpr)> = fields.iter()
        .map(|(n, e)| Ok((n.name.clone(), self.lower_expr(e)?)))
        .collect::<TenthResult<_>>()?;
    let field_types = self.structs.get(&name.name)
        .cloned()
        .unwrap_or_default();
    let ty = Type::Struct { name: name.name.clone(), fields: field_types };
    (HirExprKind::StructLiteral { name: name.name.clone(), fields: lowered_fields, ty: ty.clone() }, ty)
}
```

- [ ] **Step 7: 扩展运行时值**

在 `tenth/src/runtime/value.rs` 的 `Value` 枚举中新增：
```rust
Struct {
    name: String,
    fields: Vec<(String, Value)>,
},
```

在 `Display` 实现中添加：
```rust
Value::Struct { name, fields } => {
    write!(f, "{} {{", name)?;
    for (i, (n, v)) in fields.iter().enumerate() {
        if i > 0 { write!(f, ", ")?; }
        write!(f, "{}: {}", n, v)?;
    }
    write!(f, "}}")
}
```

在 `type_of` 中添加：
```rust
Value::Struct { name, fields } => Type::Struct {
    name: name.clone(),
    fields: fields.iter().map(|(n, v)| (n.clone(), v.type_of())).collect(),
},
```

- [ ] **Step 8: 解释器执行结构体**

在 `tenth/src/runtime/interpreter.rs` 的 `eval_expr` 中添加：
```rust
HirExprKind::StructLiteral { name, fields, .. } => {
    let mut vals = Vec::new();
    for (fname, fexpr) in fields {
        let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError {
            message: format!("struct field '{}' is void", fname),
        })?;
        vals.push((fname.clone(), v));
    }
    Ok(Some(Value::Struct { name: name.clone(), fields: vals }))
}
```

- [ ] **Step 9: 字段访问支持**

修改 `eval_expr` 中 `HirExprKind::Field` 的处理：
```rust
HirExprKind::Field { target, field } => {
    let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
        message: "field access target is void".into(),
    })?;
    match &t {
        Value::Struct { fields, .. } => {
            fields.iter()
                .find(|(n, _)| n == field)
                .map(|(_, v)| Some(v.clone()))
                .ok_or_else(|| TenthError::RuntimeError {
                    message: format!("struct has no field '{}'", field),
                })
        }
        _ => Ok(Some(Value::Unit)),
    }
}
```

- [ ] **Step 10: 编写测试**

创建 `tenth/tests/struct_test.rs`：

```rust
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let functions = hir.functions.clone();
    let mut interpreter = Interpreter::new(functions);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

#[test]
fn test_struct_definition_and_use() {
    let src = r#"
    struct Point { x: f64, y: f64 }
    let p = Point { x: 3.0, y: 4.0 };
    p.x + p.y
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 7.0).abs() < 1e-10),
        v => panic!("expected Float(7.0), got {:?}", v),
    }
}

#[test]
fn test_nested_struct() {
    let src = r#"
    struct Point { x: f64, y: f64 }
    struct Rect { top_left: Point, bottom_right: Point }
    let r = Rect {
        top_left: Point { x: 0.0, y: 10.0 },
        bottom_right: Point { x: 10.0, y: 0.0 },
    };
    r.top_left.y + r.bottom_right.x
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 20.0).abs() < 1e-10),
        v => panic!("expected Float(20.0), got {:?}", v),
    }
}
```

- [ ] **Step 11: 编译与测试验证**

```bash
cd /workspace/tenth && cargo test struct_test -- --nocapture
```

- [ ] **Step 12: Commit**

```bash
git add tenth/src/parser/ast.rs tenth/src/parser/parser.rs tenth/src/hir/types.rs tenth/src/hir/hir.rs tenth/src/hir/lower.rs tenth/src/runtime/value.rs tenth/src/runtime/interpreter.rs tenth/tests/struct_test.rs
git commit -m "feat(hir): add struct definition and literal support"
```

---

### Task 2: Impl 块与方法调用

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Modify: `tenth/tests/struct_test.rs`

**语法目标:**
```tenth
struct Point { x: f64, y: f64 }

impl Point {
    fn dist(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

let p = Point { x: 3.0, y: 4.0 };
p.dist()  // => 5.0
```

- [ ] **Step 1: 扩展 AST**

在 `ItemKind` 中新增：
```rust
Impl {
    type_name: Ident,
    methods: Vec<Item>,  // 复用 Function ItemKind
},
```

- [ ] **Step 2: 实现 Parser 解析 impl 块**

在 `parse_item` 中添加 `TokenKind::Impl` 分支：
```rust
TokenKind::Impl => {
    self.advance();
    let type_name = self.expect_ident()?;
    self.expect(TokenKind::LBrace)?;
    let mut methods = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::RBrace) {
        methods.push(self.parse_item()?);
    }
    self.expect(TokenKind::RBrace)?;
    Ok(Item { kind: ItemKind::Impl { type_name, methods }, span })
}
```

- [ ] **Step 3: 扩展 HIR**

在 `hir.rs` 中新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirImpl {
    pub type_name: String,
    pub methods: Vec<HirFnDef>,
}
```

在 `HirProgram` 中新增：
```rust
pub impls: Vec<HirImpl>,
```

- [ ] **Step 4: 实现 Lowerer**

在 `lower_program` 中收集 impl 块，构建方法注册表。`Lowerer` 中新增：
```rust
methods: HashMap<String, HashMap<String, HirFnDef>>,  // type_name -> method_name -> fn_def
```

Method call lowering：当遇到 `MethodCall { receiver, method, args }` 时，先检查 `self.methods`：

```rust
ExprKind::MethodCall { receiver, method, args } => {
    let recv = self.lower_expr(receiver)?;
    let lowered_args: Vec<HirExpr> = args.iter()
        .map(|a| self.lower_expr(a))
        .collect::<TenthResult<_>>()?;
    let ret_ty = Type::Unknown; // will be resolved from impl
    (HirExprKind::MethodCall {
        receiver: Box::new(recv),
        method: method.name.clone(),
        args: lowered_args,
        ret_ty: ret_ty.clone(),
    }, ret_ty)
}
```

- [ ] **Step 5: 解释器支持 impl 方法**

在 `Interpreter` 中新增方法注册表：
```rust
pub struct Interpreter {
    pub variables: HashMap<String, Value>,
    functions: Vec<HirFnDef>,
    methods: HashMap<String, HashMap<String, HirFnDef>>,  // type -> method -> fn
}
```

在 `new` 中初始化 `methods`。在 `execute_program` 中注册 impl 方法。

修改 `eval_method`：在 `Value::Struct` 分支中查找 impl 注册的方法：

```rust
Value::Struct { name: type_name, fields } => {
    if let Some(type_methods) = self.methods.get(type_name) {
        if let Some(method_fn) = type_methods.get(method) {
            let body = method_fn.body.clone();
            let mut saved = HashMap::new();
            // bind self
            self.variables.insert("self".to_string(), recv.clone());
            // bind params
            for ((pname, _), arg) in method_fn.params.iter().skip(1).zip(args.iter()) {
                saved.insert(pname.clone(), self.variables.get(pname).cloned());
                self.variables.insert(pname.clone(), arg.clone());
            }
            let result = self.eval_expr(&body);
            self.variables.remove("self");
            for (n, v) in saved {
                if let Some(val) = v {
                    self.variables.insert(n, val);
                } else {
                    self.variables.remove(&n);
                }
            }
            return result.transpose().unwrap_or(Ok(Value::Unit));
        }
    }
    Err(TenthError::RuntimeError {
        message: format!("method '{}' not found on type '{}'", method, type_name),
    })
}
```

- [ ] **Step 6: 编写测试**

在 `struct_test.rs` 中添加：
```rust
#[test]
fn test_impl_method_call() {
    let src = r#"
    struct Point { x: f64, y: f64 }
    impl Point {
        fn dist(self) -> f64 {
            (self.x * self.x + self.y * self.y).sqrt()
        }
    }
    let p = Point { x: 3.0, y: 4.0 };
    p.dist()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 5.0).abs() < 1e-10),
        v => panic!("expected Float(5.0), got {:?}", v),
    }
}
```

- [ ] **Step 7: 编译与测试验证**

```bash
cd /workspace/tenth && cargo test struct_test -- --nocapture
```

- [ ] **Step 8: Commit**

```bash
git add tenth/src/parser/ast.rs tenth/src/parser/parser.rs tenth/src/hir/hir.rs tenth/src/hir/lower.rs tenth/src/runtime/interpreter.rs tenth/tests/struct_test.rs
git commit -m "feat(interpreter): add impl blocks and method dispatch"
```

---

### Task 3: 枚举类型

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/types.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/value.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Create: `tenth/tests/enum_test.rs`

**语法目标:**
```tenth
enum Option {
    Some(value: i32),
    None,
}

let x = Option::Some(value: 42);
let y = Option::None;
```

- [ ] **Step 1: 扩展 AST**

在 `ItemKind` 中新增：
```rust
EnumDef {
    name: Ident,
    variants: Vec<EnumVariant>,
},
```

新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: Ident,
    pub fields: Vec<StructField>,  // 复用 StructField
}
```

在 `ExprKind` 中新增：
```rust
EnumLiteral {
    enum_name: Ident,
    variant: Ident,
    fields: Vec<(Ident, Expr)>,
},
```

- [ ] **Step 2: 实现 Parser 解析枚举**

在 `parse_item` 中添加 `TokenKind::Enum` 分支：
```rust
TokenKind::Enum => {
    self.advance();
    let name = self.expect_ident()?;
    self.expect(TokenKind::LBrace)?;
    let mut variants = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::RBrace) {
        let variant_name = self.expect_ident()?;
        let mut fields = Vec::new();
        if matches!(self.peek_kind(), TokenKind::LParen) {
            self.advance();
            while !matches!(self.peek_kind(), TokenKind::RParen) {
                let fname = self.expect_ident()?;
                self.expect(TokenKind::Colon)?;
                let ftype = self.parse_type_annotation()?;
                fields.push(StructField { name: fname, type_ann: ftype });
                if !matches!(self.peek_kind(), TokenKind::Comma) {
                    break;
                }
                self.advance();
            }
            self.expect(TokenKind::RParen)?;
        }
        variants.push(EnumVariant { name: variant_name, fields });
        if !matches!(self.peek_kind(), TokenKind::Comma) {
            break;
        }
        self.advance();
    }
    self.expect(TokenKind::RBrace)?;
    Ok(Item { kind: ItemKind::EnumDef { name, variants }, span })
}
```

- [ ] **Step 3: 实现 Parser 解析枚举构造**

在 `parse_primary` 中处理 `Ident::` 模式：
```rust
TokenKind::Identifier(ref name) if self.peek_next_is(TokenKind::ColonColon) => {
    let enum_name = Ident { name: name.clone(), span: token.span.clone() };
    self.advance(); // skip enum name
    self.advance(); // skip ::
    let variant = self.expect_ident()?;
    let mut fields = Vec::new();
    if matches!(self.peek_kind(), TokenKind::LParen) {
        self.advance();
        while !matches!(self.peek_kind(), TokenKind::RParen) {
            let fname = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let val = self.parse_expr()?;
            fields.push((fname, val));
            if !matches!(self.peek_kind(), TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        self.expect(TokenKind::RParen)?;
    }
    ExprKind::EnumLiteral { enum_name, variant, fields }
}
```

需要在 token.rs 中添加 `ColonColon` token（`::`）。查看现有 token 定义是否有 `ColonColon`：

```rust
// 在 TokenKind 中添加（如果没有）
ColonColon,
```

在 lexer.rs 的字符匹配中添加 `: :` 的识别：
```rust
':' if self.peek() == Some(':') => {
    self.advance();
    TokenKind::ColonColon
}
```

- [ ] **Step 4: 扩展 HIR**

在 `HirExprKind` 中新增：
```rust
EnumLiteral {
    enum_name: String,
    variant: String,
    fields: Vec<(String, HirExpr)>,
    ty: Type,
},
```

在 `Type` 中新增：
```rust
Enum {
    name: String,
    variants: Vec<(String, Vec<(String, Type)>)>,
},
```

在 `HirProgram` 中新增：
```rust
pub enums: Vec<HirEnumDef>,
```

新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirEnumDef {
    pub name: String,
    pub variants: Vec<(String, Vec<(String, Type)>)>,
}
```

- [ ] **Step 5: Lowerer 处理枚举**

在 `Lowerer` 中新增 `enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>`。

在 `lower_program` 中收集枚举定义。

枚举字面量 lowering：
```rust
ExprKind::EnumLiteral { enum_name, variant, fields } => {
    let lowered_fields = fields.iter()
        .map(|(n, e)| Ok((n.name.clone(), self.lower_expr(e)?)))
        .collect::<TenthResult<_>>()?;
    let variants = self.enums.get(&enum_name.name).cloned().unwrap_or_default();
    let ty = Type::Enum { name: enum_name.name.clone(), variants };
    (HirExprKind::EnumLiteral {
        enum_name: enum_name.name.clone(),
        variant: variant.name.clone(),
        fields: lowered_fields,
        ty: ty.clone(),
    }, ty)
}
```

- [ ] **Step 6: 运行时值**

在 `Value` 中新增：
```rust
Enum {
    enum_name: String,
    variant: String,
    fields: Vec<(String, Value)>,
},
```

`Display` 实现：
```rust
Value::Enum { enum_name, variant, fields } => {
    if fields.is_empty() {
        write!(f, "{}::{}", enum_name, variant)
    } else {
        write!(f, "{}::{}({})", enum_name, variant,
            fields.iter().map(|(n, v)| format!("{}: {}", n, v)).collect::<Vec<_>>().join(", "))
    }
}
```

- [ ] **Step 7: 解释器执行枚举**

在 `eval_expr` 中添加：
```rust
HirExprKind::EnumLiteral { enum_name, variant, fields, .. } => {
    let mut vals = Vec::new();
    for (fname, fexpr) in fields {
        let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError {
            message: format!("enum field '{}' is void", fname),
        })?;
        vals.push((fname.clone(), v));
    }
    Ok(Some(Value::Enum {
        enum_name: enum_name.clone(),
        variant: variant.clone(),
        fields: vals,
    }))
}
```

- [ ] **Step 8: 编写测试**

创建 `tenth/tests/enum_test.rs`：

```rust
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let functions = hir.functions.clone();
    let mut interpreter = Interpreter::new(functions);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

#[test]
fn test_enum_simple() {
    let src = r#"
    enum Color { Red, Green, Blue }
    Color::Red
    "#;
    let result = run(src).unwrap();
    assert!(result.is_some());
}

#[test]
fn test_enum_with_data() {
    let src = r#"
    enum Option { Some(value: i32), None }
    Option::Some(value: 42)
    "#;
    let result = run(src).unwrap();
    assert!(result.is_some());
}
```

- [ ] **Step 9: 编译与测试验证**

```bash
cd /workspace/tenth && cargo test enum_test -- --nocapture
```

- [ ] **Step 10: Commit**

```bash
git add tenth/src/ lexer/token.rs tenth/src/lexer/lexer.rs tenth/src/parser/ast.rs tenth/src/parser/parser.rs tenth/src/hir/types.rs tenth/src/hir/hir.rs tenth/src/hir/lower.rs tenth/src/runtime/value.rs tenth/src/runtime/interpreter.rs tenth/tests/enum_test.rs
git commit -m "feat(interpreter): add enum type support"
```

---

### Task 4: Match 表达式与模式匹配

**Files:**
- Modify: `tenth/src/parser/ast.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Modify: `tenth/tests/enum_test.rs`

**语法目标:**
```tenth
enum Option { Some(value: i32), None }

let x = Option::Some(value: 42);
match x {
    Option::Some(value: v) => v * 2,
    Option::None => 0,
}
// => 84
```

- [ ] **Step 1: 扩展 AST**

新增 Pattern 类型：
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Ident(Ident),
    Literal(Literal),
    Struct {
        name: Ident,
        fields: Vec<(Ident, Pattern)>,
    },
    Enum {
        enum_name: Ident,
        variant: Ident,
        fields: Vec<(Ident, Pattern)>,
    },
}
```

在 `ExprKind` 中新增：
```rust
Match {
    target: Box<Expr>,
    arms: Vec<MatchArm>,
},
```

新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Expr>>,
    pub body: Expr,
}
```

- [ ] **Step 2: 实现 Parser 解析 match 表达式**

在 `parse_primary` 中添加 `TokenKind::Match` 分支：
```rust
TokenKind::Match => {
    self.advance();
    let target = self.parse_expr()?;
    self.expect(TokenKind::LBrace)?;
    let mut arms = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::RBrace) {
        let pattern = self.parse_pattern()?;
        let guard = if matches!(self.peek_kind(), TokenKind::If) {
            self.advance();
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;
        arms.push(MatchArm { pattern, guard, body });
        if !matches!(self.peek_kind(), TokenKind::Comma) {
            break;
        }
        self.advance();
    }
    self.expect(TokenKind::RBrace)?;
    ExprKind::Match { target: Box::new(target), arms }
}
```

需要添加 `FatArrow` token（`=>`）。检查 lexer：

实现 `parse_pattern` 方法：
```rust
fn parse_pattern(&mut self) -> TenthResult<Pattern> {
    match self.peek_kind() {
        TokenKind::Underscore => {
            self.advance();
            Ok(Pattern::Wildcard)
        }
        TokenKind::IntLiteral(n) => {
            self.advance();
            Ok(Pattern::Literal(Literal::Int(n)))
        }
        TokenKind::FloatLiteral(n) => {
            self.advance();
            Ok(Pattern::Literal(Literal::Float(n)))
        }
        TokenKind::Identifier(ref name) => {
            let ident = Ident { name: name.clone(), span: self.peek().span.clone() };
            self.advance();
            if matches!(self.peek_kind(), TokenKind::ColonColon) {
                // Enum::Variant pattern
                let enum_name = ident;
                self.advance(); // skip ::
                let variant = self.expect_ident()?;
                let mut fields = Vec::new();
                if matches!(self.peek_kind(), TokenKind::LParen) {
                    self.advance();
                    while !matches!(self.peek_kind(), TokenKind::RParen) {
                        let fname = self.expect_ident()?;
                        self.expect(TokenKind::Colon)?;
                        let pat = self.parse_pattern()?;
                        fields.push((fname, pat));
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::RParen)?;
                }
                Ok(Pattern::Enum { enum_name, variant, fields })
            } else if matches!(self.peek_kind(), TokenKind::LBrace) {
                // Struct pattern
                self.advance(); // skip {
                let mut fields = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    let fname = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let pat = self.parse_pattern()?;
                    fields.push((fname, pat));
                    if !matches!(self.peek_kind(), TokenKind::Comma) {
                        break;
                    }
                    self.advance();
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Pattern::Struct { name: ident, fields })
            } else {
                Ok(Pattern::Ident(ident))
            }
        }
        _ => Err(TenthError::ParseError {
            line: self.span().line,
            col: self.span().col,
            message: format!("unexpected token in pattern: {:?}", self.peek_kind()),
        }),
    }
}
```

- [ ] **Step 3: 检查 token.rs 和 lexer**

确认 `Underscore`, `FatArrow`, `ColonColon` token 存在。如不存在则添加。

查看 `token.rs` 和 `lexer.rs` 确认。

- [ ] **Step 4: 扩展 HIR**

在 `HirExprKind` 中新增：
```rust
Match {
    target: Box<HirExpr>,
    arms: Vec<HirMatchArm>,
    ty: Type,
},
```

新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum HirPattern {
    Wildcard,
    Var(String),
    Literal(Literal),
    Struct { name: String, fields: Vec<(String, HirPattern)> },
    Enum { enum_name: String, variant: String, fields: Vec<(String, HirPattern)> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<Box<HirExpr>>,
    pub body: HirExpr,
}
```

- [ ] **Step 5: Lowerer 处理 match**

在 `lower_expr` 中添加：

```rust
ExprKind::Match { target, arms } => {
    let t = self.lower_expr(target)?;
    let lowered_arms: Vec<HirMatchArm> = arms.iter()
        .map(|arm| {
            Ok(HirMatchArm {
                pattern: self.lower_pattern(&arm.pattern),
                guard: arm.guard.as_ref().map(|g| self.lower_expr(g)).transpose()?.map(Box::new),
                body: self.lower_expr(&arm.body)?,
            })
        })
        .collect::<TenthResult<_>>()?;
    let ty = lowered_arms.first().map(|a| a.body.ty.clone()).unwrap_or(Type::unit());
    (HirExprKind::Match { target: Box::new(t), arms: lowered_arms, ty: ty.clone() }, ty)
}
```

添加 `lower_pattern` 方法：
```rust
fn lower_pattern(&self, pat: &Pattern) -> HirPattern {
    match pat {
        Pattern::Wildcard => HirPattern::Wildcard,
        Pattern::Ident(id) => HirPattern::Var(id.name.clone()),
        Pattern::Literal(lit) => HirPattern::Literal(lower_literal(lit)),
        Pattern::Struct { name, fields } => HirPattern::Struct {
            name: name.name.clone(),
            fields: fields.iter().map(|(n, p)| (n.name.clone(), self.lower_pattern(p))).collect(),
        },
        Pattern::Enum { enum_name, variant, fields } => HirPattern::Enum {
            enum_name: enum_name.name.clone(),
            variant: variant.name.clone(),
            fields: fields.iter().map(|(n, p)| (n.name.clone(), self.lower_pattern(p))).collect(),
        },
    }
}
```

- [ ] **Step 6: 解释器执行 match**

在 `eval_expr` 中添加：
```rust
HirExprKind::Match { target, arms, .. } => {
    let val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
        message: "match target is void".into(),
    })?;

    for arm in arms {
        if self.pattern_match(&arm.pattern, &val) {
            let guard_ok = match &arm.guard {
                Some(g) => {
                    let gv = self.eval_expr(g)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "guard is void".into(),
                    })?;
                    gv.is_truthy()
                }
                None => true,
            };
            if guard_ok {
                self.bind_pattern(&arm.pattern, &val);
                let result = self.eval_expr(&arm.body);
                self.unbind_pattern(&arm.pattern);
                return result;
            }
        }
    }

    Err(TenthError::RuntimeError {
        message: "non-exhaustive match".into(),
    })
}
```

添加模式匹配辅助方法：
```rust
fn pattern_match(&self, pat: &HirPattern, val: &Value) -> bool {
    match (pat, val) {
        (HirPattern::Wildcard, _) => true,
        (HirPattern::Var(_), _) => true,
        (HirPattern::Literal(Literal::Int(n)), Value::Int(v)) => n == v,
        (HirPattern::Literal(Literal::Float(n)), Value::Float(v)) => (n - v).abs() < 1e-10,
        (HirPattern::Literal(Literal::Bool(b)), Value::Bool(v)) => b == v,
        (HirPattern::Enum { enum_name, variant, fields }, Value::Enum { enum_name: ev, variant: vv, .. }) => {
            enum_name == ev && variant == vv && fields.iter().all(|(fn, fp)| {
                val_as_enum_field(val, fn).map(|fv| self.pattern_match(fp, fv)).unwrap_or(false)
            })
        }
        (HirPattern::Struct { name, fields }, Value::Struct { name: sn, .. }) => {
            name == sn && fields.iter().all(|(fn, fp)| {
                val_as_struct_field(val, fn).map(|fv| self.pattern_match(fp, fv)).unwrap_or(false)
            })
        }
        _ => false,
    }
}
```

- [ ] **Step 7: 编写测试**

在 `enum_test.rs` 中添加：
```rust
#[test]
fn test_match_enum() {
    let src = r#"
    enum Option { Some(value: i32), None }
    let x = Option::Some(value: 42);
    match x {
        Option::Some(value: v) => v * 2,
        Option::None => 0,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(84)) => {}
        v => panic!("expected Int(84), got {:?}", v),
    }
}

#[test]
fn test_match_wildcard() {
    let src = r#"
    enum Option { Some(value: i32), None }
    let x = Option::None;
    match x {
        Option::Some(value: v) => v,
        _ => -1,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(-1)) => {}
        v => panic!("expected Int(-1), got {:?}", v),
    }
}
```

- [ ] **Step 8: 编译与测试验证**

```bash
cd /workspace/tenth && cargo test enum_test -- --nocapture
```

- [ ] **Step 9: Commit**

```bash
git add tenth/src/parser/ast.rs tenth/src/parser/parser.rs tenth/src/hir/hir.rs tenth/src/hir/lower.rs tenth/src/runtime/interpreter.rs tenth/tests/enum_test.rs
git commit -m "feat(interpreter): add match expression and pattern matching"
```

---

### Task 5: 模块系统 (mod/use)

**Files:**
- Modify: `tenth/src/parser/parser.rs` (~50行)
- Modify: `tenth/src/hir/hir.rs`
- Modify: `tenth/src/hir/lower.rs` (~40行)
- Modify: `tenth/src/runtime/interpreter.rs` (~20行)
- Modify: `tenth/src/repl.rs` (~15行)
- Create: `tenth/tests/module_test.rs`

**语法目标:**
```tenth
// File: main.th
mod math {
    fn add(a: i32, b: i32) -> i32 {
        a + b
    }
}

use math::add;
add(1, 2)  // => 3
```

- [ ] **Step 1: 实现 Parser 解析 mod 和 use**

AST 中已有 `ItemKind::Mod` 和 `ItemKind::Use`。实现 parser 中的对应分支：

```rust
TokenKind::Mod => {
    self.advance();
    let name = self.expect_ident()?;
    self.expect(TokenKind::LBrace)?;
    let mut items = Vec::new();
    while !matches!(self.peek_kind(), TokenKind::RBrace) {
        items.push(self.parse_item()?);
    }
    self.expect(TokenKind::RBrace)?;
    Ok(Item { kind: ItemKind::Mod { name, items }, span })
}

TokenKind::Use => {
    self.advance();
    let mut path = vec![self.expect_ident()?];
    while matches!(self.peek_kind(), TokenKind::ColonColon) {
        self.advance();
        path.push(self.expect_ident()?);
    }
    self.expect(TokenKind::Semicolon)?;
    Ok(Item { kind: ItemKind::Use { path }, span })
}
```

- [ ] **Step 2: 扩展 HIR**

在 `hir.rs` 中新增：
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirModule {
    pub name: String,
    pub items: Vec<HirItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirItem {
    Function(HirFnDef),
    Struct(HirStructDef),
    Enum(HirEnumDef),
    Impl(HirImpl),
    Module(HirModule),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirUse {
    pub path: Vec<String>,
}
```

`HirProgram` 更新为：
```rust
pub struct HirProgram {
    pub modules: Vec<HirModule>,
    pub uses: Vec<HirUse>,
    pub functions: Vec<HirFnDef>,
    pub structs: Vec<HirStructDef>,
    pub enums: Vec<HirEnumDef>,
    pub impls: Vec<HirImpl>,
    pub main_expr: Option<HirExpr>,
}
```

- [ ] **Step 3: Lowerer 处理模块**

在 `lower_program` 中遍历 items，按类型归类并递归处理嵌套 mod：

```rust
pub fn lower_program(&mut self, program: &Program) -> TenthResult<HirProgram> {
    let mut modules = Vec::new();
    let mut uses = Vec::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Mod { name, items } => {
                let hir_items = self.lower_items(items)?;
                modules.push(HirModule { name: name.name.clone(), items: hir_items });
            }
            ItemKind::Use { path } => {
                uses.push(HirUse { path: path.iter().map(|i| i.name.clone()).collect() });
            }
            _ => { /* handled in lower_items */ }
        }
    }

    let items = self.lower_items(&program.items)?;
    let (functions, structs, enums, impls) = self.partition_items(items);

    Ok(HirProgram { modules, uses, functions, structs, enums, impls, main_expr: self.main_expr.take() })
}
```

- [ ] **Step 4: 解释器支持 use**

在 `Interpreter` 中处理 use 语句：将模块内的函数注册到全局作用域：

```rust
fn resolve_use(&mut self, use_: &HirUse, modules: &[HirModule]) -> TenthResult<()> {
    for module in modules {
        if module.name == use_.path[0] {
            if use_.path.len() == 2 {
                // use module::name
                for item in &module.items {
                    match item {
                        HirItem::Function(f) => {
                            self.register_function(f);
                        }
                        _ => {}
                    }
                }
            }
            // full module import
            if use_.path.len() == 1 {
                for item in &module.items {
                    if let HirItem::Function(f) = item {
                        let name = format!("{}::{}", module.name, f.name);
                        self.variables.insert(name, Value::FnRef { ... });
                    }
                }
            }
        }
    }
    Ok(())
}
```

- [ ] **Step 5: REPL 支持多行模块输入**

修改 `repl.rs`，在 REPL 中添加简单的文件加载支持，或有状态的多行模块输入。

- [ ] **Step 6: 编写测试**

创建 `tenth/tests/module_test.rs`：

```rust
#[test]
fn test_module_function() {
    let src = r#"
    mod math {
        fn add(a: i32, b: i32) -> i32 { a + b }
    }
    math::add(1, 2)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Int(3), got {:?}", v),
    }
}
```

- [ ] **Step 7: 编译与测试验证**

```bash
cd /workspace/tenth && cargo test module_test -- --nocapture
```

- [ ] **Step 8: Commit**

```bash
git add tenth/src/parser/parser.rs tenth/src/hir/hir.rs tenth/src/hir/lower.rs tenth/src/runtime/interpreter.rs tenth/src/repl.rs tenth/tests/module_test.rs
git commit -m "feat(interpreter): add module system with mod and use"
```

---

### Task 6: 标准库扩展 — 张量操作增强

**Files:**
- Modify: `tenth/src/runtime/tensor.rs`
- Modify: `tenth/src/runtime/interpreter.rs`
- Modify: `tenth/src/runtime/value.rs`
- Modify: `tenth/tests/integration_test.rs`

**目标:** 新增 `rand`, `randn`, `matmul`, `softmax`, `broadcast_to` 等张量操作。

- [ ] **Step 1: 扩展 Tensor 实现**

在 `tenth/src/runtime/tensor.rs` 中新增方法：

```rust
impl Tensor {
    pub fn rand(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.r#gen()).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    pub fn randn(shape: &[usize]) -> Self {
        use rand_distr::{Normal, Distribution};
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    pub fn matmul(&self, other: &Tensor) -> Option<Tensor> {
        let a_shape = self.shape();
        let b_shape = other.shape();
        if a_shape.len() < 2 || b_shape.len() < 2 {
            return None;
        }
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        if k != k2 {
            return None;
        }
        // Simple 2D matmul using ndarray
        let a = ndarray::Array2::from_shape_vec((m, k), self.data.clone()).ok()?;
        let b = ndarray::Array2::from_shape_vec((k, n), other.data.clone()).ok()?;
        let c = a.dot(&b);
        Tensor::from_vec(c.into_raw_vec(), vec![m, n]).into()
    }

    pub fn softmax(&self, axis: isize) -> Option<Tensor> {
        // Subtract max for numerical stability, then exp and normalize
        let max_val = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = self.data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let result: Vec<f64> = exps.iter().map(|x| x / sum).collect();
        Some(Tensor::from_vec(result, self.shape.clone()))
    }

    pub fn broadcast_to(&self, target_shape: &[usize]) -> Option<Tensor> {
        // Simple broadcast implementation for common cases
        if self.shape == target_shape {
            return Some(self.clone());
        }
        let self_size: usize = self.shape.iter().product();
        let target_size: usize = target_shape.iter().product();
        if self_size == 1 {
            let repeated: Vec<f64> = vec![self.data[0]; target_size];
            return Some(Tensor::from_vec(repeated, target_shape.to_vec()));
        }
        None
    }
}
```

- [ ] **Step 2: 解释器注册新的内置函数**

在 `call_named_fn` 中添加：
```rust
"rand" => {
    let shape: Vec<usize> = args.iter()
        .map(|a| a.as_int().unwrap_or(1) as usize)
        .collect();
    let t = Tensor::rand(&shape);
    return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
}
"randn" => {
    let shape: Vec<usize> = args.iter()
        .map(|a| a.as_int().unwrap_or(1) as usize)
        .collect();
    let t = Tensor::randn(&shape);
    return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
}
```

在 `eval_method` 的 `Value::Tensor` 分支中新增方法：
```rust
"matmul" => {
    if let Some(arg) = args.first() {
        if let Value::Tensor(other) = arg {
            let result = tensor.matmul(&other.borrow())
                .ok_or_else(|| TenthError::RuntimeError {
                    message: "matmul: incompatible shapes".into(),
                })?;
            return Ok(Value::Tensor(Rc::new(RefCell::new(result))));
        }
    }
    return Err(TenthError::RuntimeError {
        message: "matmul: expected tensor argument".into(),
    });
}
"softmax" => {
    let axis = args.first().map(|a| a.as_int().unwrap_or(-1) as isize).unwrap_or(-1);
    let result = tensor.softmax(axis)
        .ok_or_else(|| TenthError::RuntimeError {
            message: "softmax failed".into(),
        })?;
    return Ok(Value::Tensor(Rc::new(RefCell::new(result))));
}
```

- [ ] **Step 3: 编写集成测试**

在 `tests/integration_test.rs` 中添加：

```rust
#[test]
fn test_tensor_rand_and_sum() {
    let src = "let x = rand(3, 224, 224); x.sum()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!(v >= 0.0),
        v => panic!("expected Float >= 0, got {:?}", v),
    }
}

#[test]
fn test_tensor_softmax() {
    let src = "tensor[[1.0, 2.0, 3.0]].softmax().sum()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 1.0).abs() < 1e-6),
        v => panic!("expected Float(1.0), got {:?}", v),
    }
}
```

- [ ] **Step 4: 运行全部测试验证**

```bash
cd /workspace/tenth && cargo test
```

- [ ] **Step 5: Commit**

```bash
git add tenth/src/runtime/tensor.rs tenth/src/runtime/interpreter.rs tenth/tests/integration_test.rs
git commit -m "feat(stdlib): add rand, randn, matmul, softmax tensor ops"
```

---

### Task 7: 改进错误诊断

**Files:**
- Modify: `tenth/src/error.rs`
- Modify: `tenth/src/lexer/lexer.rs`
- Modify: `tenth/src/parser/parser.rs`
- Modify: `tenth/src/repl.rs`

**目标:** 错误信息中包含源码片段和位置指示器。

- [ ] **Step 1: 扩展 TenthError**

在 `error.rs` 中为 `ParseError` 和 `TypeError` 添加源码上下文：

```rust
#[derive(Debug)]
pub enum TenthError {
    LexerError { line: u32, col: u32, message: String },
    ParseError { line: u32, col: u32, message: String, source_line: Option<String> },
    TypeError { line: u32, col: u32, message: String },
    RuntimeError { message: String },
    UnexpectedEof,
}
```

- [ ] **Step 2: 改进错误 Display**

实现带源码片段的错误显示：
```rust
impl fmt::Display for TenthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TenthError::ParseError { line, col, message, source_line } => {
                write!(f, "Parse error at line {}, col {}: {}", line, col, message)?;
                if let Some(src) = source_line {
                    write!(f, "\n  {} | {}", line, src)?;
                    write!(f, "\n  {} | {}{}", " ".repeat(line.to_string().len()), " ".repeat(*col as usize - 1), "^")?;
                }
                Ok(())
            }
            // ... other variants
        }
    }
}
```

- [ ] **Step 3: Parser 传递源码行**

修改 `Parser` 结构体保存源码行：
```rust
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    source_lines: Vec<String>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, source_lines: Vec::new() }
    }

    pub fn with_source(tokens: Vec<Token>, source: &str) -> Self {
        let source_lines: Vec<String> = source.lines().map(String::from).collect();
        Self { tokens, pos: 0, source_lines }
    }
}
```

- [ ] **Step 4: REPL 使用 with_source**

在 `repl.rs` 的 `execute_line` 中：
```rust
let mut parser = Parser::with_source(tokens, line);
```

- [ ] **Step 5: 运行全量测试验证**

```bash
cd /workspace/tenth && cargo test && cargo run
```

- [ ] **Step 6: Commit**

```bash
git add tenth/src/error.rs tenth/src/parser/parser.rs tenth/src/repl.rs
git commit -m "feat(error): improve error diagnostics with source context"
```

---

### Task 8: 全量测试与验收

- [ ] **Step 1: 运行全部测试**

```bash
cd /workspace/tenth && cargo test
```

验证所有测试通过（预计 ~40 个测试）。

- [ ] **Step 2: REPL 验收测试**

```bash
cd /workspace/tenth && echo '
struct Point { x: f64, y: f64 }
impl Point {
    fn dist(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
let p = Point { x: 3.0, y: 4.0 };
p.dist()
' | cargo run --quiet
```

预期输出: `= 5`

- [ ] **Step 3: 张量验收**

```bash
cd /workspace/tenth && echo '
let x = rand(3, 224, 224);
x.sum()
' | cargo run --quiet
```

预期输出: `= <一个大数>` 而非错误。

- [ ] **Step 4: Match 枚举验收**

```bash
cd /workspace/tenth && echo '
enum Option { Some(value: i32), None }
let x = Option::Some(value: 42);
match x {
    Option::Some(value: v) => v * 2,
    Option::None => 0,
}
' | cargo run --quiet
```

预期输出: `= 84`

- [ ] **Step 5: Commit 最终状态**

```bash
git add -A && git commit -m "feat: Phase 2 interpreter hardening complete"
```

---

## Phase 2 完成标准

- [ ] `struct` 定义和字面量：能定义、实例化、访问字段
- [ ] `impl` 块：能为 struct 定义方法并通过 `.method()` 调用
- [ ] `enum` 类型：能定义带数据变体的枚举并构造值
- [ ] `match` 表达式：能对枚举进行模式匹配（含通配符 `_` 和解构绑定）
- [ ] `mod`/`use`：模块化代码组织，支持 `mod::name::func()` 调用
- [ ] 张量 `rand(shape)` / `randn(shape)`：随机张量创建
- [ ] 张量 `softmax()` / `matmul()`：更多 AI 相关运算
- [ ] 所有测试通过（预计 35-40 个测试）
- [ ] 错误信息包含源码片段和 `^` 指示器