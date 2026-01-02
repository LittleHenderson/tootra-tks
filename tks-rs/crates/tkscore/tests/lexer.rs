use tkscore::lexer::{Lexer, TokenKind};

#[test]
fn lex_string_literal() {
    let tokens = Lexer::new(r#""hello world""#).tokenize().unwrap();
    assert!(matches!(&tokens[0].kind, TokenKind::StringLit(s) if s == "hello world"));
    assert_eq!(tokens[1].kind, TokenKind::Eof);
}

#[test]
fn lex_string_with_escapes() {
    let input = r#""line1\nline2\ttab\slash\"quote""#;
    let tokens = Lexer::new(input).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            // After processing escapes, we should have actual newline, tab, backslash, quote
            assert_eq!(s, "line1\nline2\ttabslash\"quote");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_empty_string() {
    let tokens = Lexer::new(r#""""#).tokenize().unwrap();
    assert!(matches!(&tokens[0].kind, TokenKind::StringLit(s) if s.is_empty()));
}

#[test]
fn lex_string_with_carriage_return() {
    let tokens = Lexer::new(r#""hello\rworld""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "hello\rworld");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_unterminated_string_error() {
    let result = Lexer::new(r#""hello"#).tokenize();
    assert!(result.is_err());
    assert!(result.unwrap_err().message.contains("unterminated"));
}

#[test]
fn lex_unknown_escape_passes_through() {
    // Unknown escape sequences pass through the character after backslash
    let tokens = Lexer::new(r#""hello\xworld""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "helloxworld");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_array_brackets() {
    let tokens = Lexer::new("[1, 2, 3]").tokenize().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LBracket);
    assert_eq!(tokens[1].kind, TokenKind::Int(1));
    assert_eq!(tokens[2].kind, TokenKind::Comma);
    assert_eq!(tokens[3].kind, TokenKind::Int(2));
    assert_eq!(tokens[4].kind, TokenKind::Comma);
    assert_eq!(tokens[5].kind, TokenKind::Int(3));
    assert_eq!(tokens[6].kind, TokenKind::RBracket);
    assert_eq!(tokens[7].kind, TokenKind::Eof);
}

#[test]
fn lex_array_of_strings() {
    let tokens = Lexer::new(r#"["a", "b"]"#).tokenize().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LBracket);
    assert!(matches!(&tokens[1].kind, TokenKind::StringLit(s) if s == "a"));
    assert_eq!(tokens[2].kind, TokenKind::Comma);
    assert!(matches!(&tokens[3].kind, TokenKind::StringLit(s) if s == "b"));
    assert_eq!(tokens[4].kind, TokenKind::RBracket);
}

#[test]
fn lex_nested_brackets() {
    let tokens = Lexer::new("[[1], [2]]").tokenize().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LBracket);
    assert_eq!(tokens[1].kind, TokenKind::LBracket);
    assert_eq!(tokens[2].kind, TokenKind::Int(1));
    assert_eq!(tokens[3].kind, TokenKind::RBracket);
    assert_eq!(tokens[4].kind, TokenKind::Comma);
    assert_eq!(tokens[5].kind, TokenKind::LBracket);
    assert_eq!(tokens[6].kind, TokenKind::Int(2));
    assert_eq!(tokens[7].kind, TokenKind::RBracket);
    assert_eq!(tokens[8].kind, TokenKind::RBracket);
}

// Additional escape sequence tests

#[test]
fn lex_string_newline_escape() {
    let tokens = Lexer::new(r#""hello\nworld""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "hello\nworld");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_tab_escape() {
    let tokens = Lexer::new(r#""col1\tcol2""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "col1\tcol2");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_backslash_escape() {
    // Test escaped backslashes: \\ -> \
    let tokens = Lexer::new(r#""path\\to\\file""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, r"path\to\file");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_quote_escape() {
    let tokens = Lexer::new(r#""say \"hello\"""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "say \"hello\"");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_multiple_escapes() {
    // Tests: \n \t \\ \"
    let tokens = Lexer::new(r#""line1\nline2\ttab\\slash\"quote""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "line1\nline2\ttab\\slash\"quote");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_consecutive_escapes() {
    let tokens = Lexer::new(r#""\n\n\t\t\\\\""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "\n\n\t\t\\\\");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_escape_at_end() {
    let tokens = Lexer::new(r#""end with newline\n""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "end with newline\n");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_escape_at_start() {
    let tokens = Lexer::new(r#""\tindented""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "\tindented");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_only_escapes() {
    let tokens = Lexer::new(r#""\n\t\r\\\"""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "\n\t\r\\\"");
        }
        _ => panic!("expected StringLit"),
    }
}

#[test]
fn lex_string_with_spaces_and_escapes() {
    let tokens = Lexer::new(r#""hello world\nand\ttabs""#).tokenize().unwrap();
    match &tokens[0].kind {
        TokenKind::StringLit(s) => {
            assert_eq!(s, "hello world\nand\ttabs");
        }
        _ => panic!("expected StringLit"),
    }
}
