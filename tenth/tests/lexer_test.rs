use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::TokenKind;
use tenth::hir::types::BaseType;

fn tokenize(src: &str) -> Vec<TokenKind> {
    let mut lexer = Lexer::new(src);
    lexer.tokenize().unwrap().into_iter().map(|t| t.kind).collect()
}

#[test]
fn test_integers() {
    let tokens = tokenize("42 0 100");
    assert_eq!(tokens[0], TokenKind::IntLiteral(42, BaseType::I32));
    assert_eq!(tokens[1], TokenKind::IntLiteral(0, BaseType::I32));
    assert_eq!(tokens[2], TokenKind::IntLiteral(100, BaseType::I32));
}

#[test]
fn test_keywords() {
    let tokens = tokenize("fn let mut if else match for while return");
    assert_eq!(tokens[0], TokenKind::Fn);
    assert_eq!(tokens[1], TokenKind::Let);
    assert_eq!(tokens[2], TokenKind::Mut);
    assert_eq!(tokens[3], TokenKind::If);
    assert_eq!(tokens[4], TokenKind::Else);
    assert_eq!(tokens[5], TokenKind::Match);
    assert_eq!(tokens[6], TokenKind::For);
    assert_eq!(tokens[7], TokenKind::While);
    assert_eq!(tokens[8], TokenKind::Return);
}

#[test]
fn test_operators() {
    let tokens = tokenize("+ - * / == != < > <= >= && || !");
    assert_eq!(tokens[0], TokenKind::Plus);
    assert_eq!(tokens[1], TokenKind::Minus);
    assert_eq!(tokens[2], TokenKind::Star);
    assert_eq!(tokens[3], TokenKind::Slash);
    assert_eq!(tokens[4], TokenKind::EqEq);
    assert_eq!(tokens[5], TokenKind::NotEq);
    assert_eq!(tokens[6], TokenKind::Lt);
    assert_eq!(tokens[7], TokenKind::Gt);
    assert_eq!(tokens[8], TokenKind::LtEq);
    assert_eq!(tokens[9], TokenKind::GtEq);
    assert_eq!(tokens[10], TokenKind::AndAnd);
    assert_eq!(tokens[11], TokenKind::OrOr);
    assert_eq!(tokens[12], TokenKind::Not);
}

#[test]
fn test_string_literal() {
    let tokens = tokenize("\"hello world\"");
    assert_eq!(tokens[0], TokenKind::StringLiteral("hello world".into()));
}

#[test]
fn test_comment_skip() {
    let tokens = tokenize("// this is a comment\n42");
    assert_eq!(tokens[0], TokenKind::IntLiteral(42, BaseType::I32));
}

#[test]
fn test_bom_skip() {
    // UTF-8 BOM (U+FEFF) should be skipped — PowerShell Out-File -Encoding utf8 默认添加
    let tokens = tokenize("\u{FEFF}fn main");
    assert_eq!(tokens[0], TokenKind::Fn);
    assert_eq!(tokens[1], TokenKind::Identifier("main".into()));
}

#[test]
fn test_identifier() {
    let tokens = tokenize("my_var tensor randn");
    assert_eq!(tokens[0], TokenKind::Identifier("my_var".into()));
    assert_eq!(tokens[1], TokenKind::Identifier("tensor".into()));
    assert_eq!(tokens[2], TokenKind::Identifier("randn".into()));
}

// --- 问题4：char 字面量 ---

#[test]
fn test_char_literal_simple() {
    let tokens = tokenize("'a'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('a'));
}

#[test]
fn test_char_literal_digit() {
    let tokens = tokenize("'5'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('5'));
}

#[test]
fn test_char_literal_escape_n() {
    let tokens = tokenize("'\\n'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('\n'));
}

#[test]
fn test_char_literal_escape_t() {
    let tokens = tokenize("'\\t'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('\t'));
}

#[test]
fn test_char_literal_escape_r() {
    let tokens = tokenize("'\\r'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('\r'));
}

#[test]
fn test_char_literal_escape_backslash() {
    let tokens = tokenize("'\\\\'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('\\'));
}

#[test]
fn test_char_literal_escape_single_quote() {
    let tokens = tokenize("'\\''");
    assert_eq!(tokens[0], TokenKind::CharLiteral('\''));
}

#[test]
fn test_char_literal_escape_double_quote() {
    let tokens = tokenize("'\\\"'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('"'));
}

#[test]
fn test_char_literal_escape_null() {
    let tokens = tokenize("'\\0'");
    assert_eq!(tokens[0], TokenKind::CharLiteral('\0'));
}

#[test]
fn test_char_literal_in_expression() {
    // char literal followed by other tokens
    let tokens = tokenize("'x' 42");
    assert_eq!(tokens[0], TokenKind::CharLiteral('x'));
    assert_eq!(tokens[1], TokenKind::IntLiteral(42, BaseType::I32));
}

// --- 问题6：进制字面量 ---

#[test]
fn test_hex_literal() {
    let tokens = tokenize("0xFF");
    assert_eq!(tokens[0], TokenKind::IntLiteral(255, BaseType::I32));
}

#[test]
fn test_hex_literal_uppercase_prefix() {
    let tokens = tokenize("0XFF");
    assert_eq!(tokens[0], TokenKind::IntLiteral(255, BaseType::I32));
}

#[test]
fn test_hex_literal_lowercase() {
    let tokens = tokenize("0xff");
    assert_eq!(tokens[0], TokenKind::IntLiteral(255, BaseType::I32));
}

#[test]
fn test_hex_literal_mixed_case() {
    let tokens = tokenize("0xAbCdEf");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0xABCDEF, BaseType::I32));
}

#[test]
fn test_hex_literal_with_underscore() {
    let tokens = tokenize("0xFF_FF");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0xFFFF, BaseType::I32));
}

#[test]
fn test_hex_literal_zero() {
    let tokens = tokenize("0x0");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0, BaseType::I32));
}

#[test]
fn test_hex_literal_max_i64() {
    let tokens = tokenize("0x7FFFFFFFFFFFFFFF");
    assert_eq!(tokens[0], TokenKind::IntLiteral(i64::MAX, BaseType::I32));
}

#[test]
fn test_binary_literal() {
    let tokens = tokenize("0b1010");
    assert_eq!(tokens[0], TokenKind::IntLiteral(10, BaseType::I32));
}

#[test]
fn test_binary_literal_uppercase_prefix() {
    let tokens = tokenize("0B1010");
    assert_eq!(tokens[0], TokenKind::IntLiteral(10, BaseType::I32));
}

#[test]
fn test_binary_literal_with_underscore() {
    let tokens = tokenize("0b1010_1010");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0b10101010, BaseType::I32));
}

#[test]
fn test_binary_literal_zero() {
    let tokens = tokenize("0b0");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0, BaseType::I32));
}

#[test]
fn test_octal_literal() {
    let tokens = tokenize("0o777");
    assert_eq!(tokens[0], TokenKind::IntLiteral(511, BaseType::I32));
}

#[test]
fn test_octal_literal_uppercase_prefix() {
    let tokens = tokenize("0O777");
    assert_eq!(tokens[0], TokenKind::IntLiteral(511, BaseType::I32));
}

#[test]
fn test_octal_literal_with_underscore() {
    let tokens = tokenize("0o777_777");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0o777777, BaseType::I32));
}

#[test]
fn test_octal_literal_zero() {
    let tokens = tokenize("0o0");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0, BaseType::I32));
}

#[test]
fn test_decimal_still_works_after_radix_support() {
    // 0 不应被误认为进制前缀
    let tokens = tokenize("0 42 100");
    assert_eq!(tokens[0], TokenKind::IntLiteral(0, BaseType::I32));
    assert_eq!(tokens[1], TokenKind::IntLiteral(42, BaseType::I32));
    assert_eq!(tokens[2], TokenKind::IntLiteral(100, BaseType::I32));
}

#[test]
fn test_float_starting_with_zero_still_works() {
    // 0.5 不应被进制解析截断
    let tokens = tokenize("0.5 0.0");
    assert_eq!(tokens[0], TokenKind::FloatLiteral(0.5, BaseType::F64));
    assert_eq!(tokens[1], TokenKind::FloatLiteral(0.0, BaseType::F64));
}