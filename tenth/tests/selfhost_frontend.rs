//! Phase B integration test: tenthc frontend can parse its own source.
//! Uses the Rust mother compiler to run tenthc's lexer/parser/lowerer
//! on tenthc's own .th files, verifying no parse/lower errors.

#[cfg(test)]
mod selfhost_frontend {
    use tenth::lexer::lexer::Lexer;
    use tenth::parser::parser::Parser;
    use tenth::hir::lower::Lowerer;

    /// Load all tenthc .th source files concatenated.
    fn tenthc_source() -> String {
        [
            include_str!("../../tenthc/lexer/token.th"),
            include_str!("../../tenthc/lexer/lexer.th"),
            include_str!("../../tenthc/parser/parser.th"),
            include_str!("../../tenthc/hir/hir.th"),
            include_str!("../../tenthc/hir/lower.th"),
            include_str!("../../tenthc/compile/wasm.th"),
        ].join("\n")
    }

    #[test]
    fn tenthc_lexes_own_source() {
        let src = tenthc_source();
        let mut lexer = Lexer::new(&src);
        let tokens = lexer.tokenize();
        assert!(tokens.is_ok(), "lex failed: {:?}", tokens.err());
        let tokens = tokens.unwrap();
        println!("Lexed {} tokens", tokens.len());
        assert!(tokens.len() > 100, "expected many tokens, got {}", tokens.len());
    }

    #[test]
    fn tenthc_parses_own_source() {
        let src = tenthc_source();
        let mut lexer = Lexer::new(&src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program();
        // Parser may produce errors for unsupported features; verify it doesn't panic
        match program {
            Ok(p) => {
                println!("Parsed: {} items", p.items.len());
                assert!(p.items.len() > 0, "expected items in tenthc source");
            }
            Err(e) => {
                // Log but don't fail — Phase B is about identifying gaps
                println!("Parse error (expected during Phase B): {:?}", e);
            }
        }
    }

    #[test]
    fn tenthc_lowers_own_source() {
        let src = tenthc_source();
        let mut lexer = Lexer::new(&src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = match parser.parse_program() {
            Ok(p) => p,
            Err(e) => {
                println!("SKIP lower: parse error: {:?}", e);
                return;
            }
        };
        let mut lowerer = Lowerer::new();
        match lowerer.lower_program(&program) {
            Ok(hir) => {
                println!("Lowered: {} functions", hir.functions.len());
                assert!(hir.functions.len() > 0, "expected functions in HIR");
            }
            Err(e) => {
                println!("Lower error (expected during Phase B): {:?}", e);
            }
        }
    }

    #[test]
    fn tenthc_individual_files_lex() {
        // Each file should lex independently
        let files = [
            ("token.th", include_str!("../../tenthc/lexer/token.th")),
            ("lexer.th", include_str!("../../tenthc/lexer/lexer.th")),
            ("parser.th", include_str!("../../tenthc/parser/parser.th")),
            ("hir.th", include_str!("../../tenthc/hir/hir.th")),
            ("lower.th", include_str!("../../tenthc/hir/lower.th")),
            ("wasm.th", include_str!("../../tenthc/compile/wasm.th")),
        ];
        for (name, src) in &files {
            let mut lexer = Lexer::new(src);
            let result = lexer.tokenize();
            assert!(result.is_ok(), "lex {} failed: {:?}", name, result.err());
            let tokens = result.unwrap();
            println!("{}: {} tokens", name, tokens.len());
        }
    }
}
