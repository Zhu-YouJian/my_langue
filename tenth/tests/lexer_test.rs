use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::TokenKind;

fn tokenize(src: &str) -> Vec<TokenKind> {
    let mut lexer = Lexer::new(src);
    lexer.tokenize().unwrap().into_iter().map(|t| t.kind).collect()
}

#[test]
fn test_integers() {
    let tokens = tokenize("42 0 100");
    assert_eq!(tokens[0], TokenKind::IntLiteral(42));
    assert_eq!(tokens[1], TokenKind::IntLiteral(0));
    assert_eq!(tokens[2], TokenKind::IntLiteral(100));
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
    assert_eq!(tokens[0], TokenKind::IntLiteral(42));
}

#[test]
fn test_identifier() {
    let tokens = tokenize("my_var tensor randn");
    assert_eq!(tokens[0], TokenKind::Identifier("my_var".into()));
    assert_eq!(tokens[1], TokenKind::Identifier("tensor".into()));
    assert_eq!(tokens[2], TokenKind::Identifier("randn".into()));
}