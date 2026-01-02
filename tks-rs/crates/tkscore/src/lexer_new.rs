use crate::ast::{Aspect, Ident, World};
use crate::span::Span;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Static keyword lookup table for O(1) keyword resolution.
/// Using LazyLock ensures the HashMap is initialized exactly once on first access.
static KEYWORDS: LazyLock<HashMap<&'static str, TokenKind>> = LazyLock::new(|| {
    HashMap::from([
        ("let", TokenKind::Let),
        ("in", TokenKind::In),
        ("if", TokenKind::If),
        ("then", TokenKind::Then),
        ("else", TokenKind::Else),
        ("return", TokenKind::Return),
        ("check", TokenKind::Check),
        ("acquire", TokenKind::Acquire),
        ("acbe", TokenKind::Acbe),
        ("effect", TokenKind::Effect),
        ("handle", TokenKind::Handle),
        ("with", TokenKind::With),
        ("resume", TokenKind::Resume),
        ("perform", TokenKind::Perform),
        ("handler", TokenKind::Handler),
        ("op", TokenKind::Op),
        ("blueprint", TokenKind::Blueprint),
        ("class", TokenKind::Class),
        ("plan", TokenKind::Plan),
        ("specifics", TokenKind::Specifics),
        ("field", TokenKind::Field),
        ("details", TokenKind::Details),
        ("description", TokenKind::Description),
        ("actions", TokenKind::Actions),
        ("method", TokenKind::Method),
        ("identity", TokenKind::Identity),
        ("repeat", TokenKind::Repeat),
        ("new", TokenKind::New),
        ("self", TokenKind::SelfKw),
        ("mut", TokenKind::Mut),
        ("module", TokenKind::Module),
        ("import", TokenKind::Import),
        ("export", TokenKind::Export),
        ("from", TokenKind::From),
        ("step", TokenKind::Step),
        ("as", TokenKind::As),
        ("type", TokenKind::TypeKw),
        ("extern", TokenKind::External),
        ("external", TokenKind::External),
        ("fn", TokenKind::FnKw),
        ("safe", TokenKind::Safe),
        ("unsafe", TokenKind::Unsafe),
        ("omega", TokenKind::Omega),
        ("epsilon", TokenKind::Epsilon),
        ("aleph", TokenKind::Aleph),
        ("sup", TokenKind::Sup),
        ("ord", TokenKind::Ord),
        ("limit", TokenKind::Limit),
        ("succ", TokenKind::Succ),
        ("transfinite", TokenKind::Transfinite),
        ("loop", TokenKind::Loop),
        ("superpose", TokenKind::Superpose),
        ("measure", TokenKind::Measure),
        ("entangle", TokenKind::Entangle),
        ("qstate", TokenKind::QState),
        ("amplitude", TokenKind::Amplitude),
        ("basis", TokenKind::Basis),
        ("true", TokenKind::Bool(true)),
        ("false", TokenKind::Bool(false)),
        ("plus", TokenKind::Plus),
    ])
});

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Element { world: World, index: u8 },
    Noetic { index: u8 },
    Foundation { level: u8, aspect: Aspect },
    Int(i64),
    Bool(bool),
    Float(f64),
    Complex { re: f64, im: f64 },
    StringLit(String),
    Ident(Ident),
    Nu,
    Lambda,
    Let,
    In,
    If,
    Then,
    Else,
    Return,
    Bind,
    Check,
    Acquire,
    Acbe,
    Effect,
    Handle,
    With,
    Resume,
    Perform,
    Handler,
    Op,
    Blueprint,
    Class,
    Plan,
    Specifics,
    Field,
    Details,
    Description,
    Actions,
    Method,
    Identity,
    Repeat,
    New,
    SelfKw,
    Mut,
    Module,
    Import,
    Export,
    From,
    Step,
    As,
    TypeKw,
    External,
    FnKw,
    Safe,
    Unsafe,
    Omega,
    Epsilon,
    Aleph,
    Sup,
    Ord,
    Limit,
    Succ,
    Transfinite,
    Loop,
    Measure,
    Superpose,
    Entangle,
    QState,
    Amplitude,
    Basis,
    FracOpen,
    FracClose,
    Colon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Arrow,
    DoubleArrow,
    Equals,
    Plus,
    Minus,
    Times,
    Divide,
    Caret,
    LBrace,
    RBrace,
    Pipe,
    Bang,
    Dot,
    Comma,
    Semicolon,
    LAngle,
    RAngle,
    Underscore,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub struct Lexer<'a> {
    input: Vec<char>,
    pos: usize,
    line: u32,
    column: u32,
    _source: &'a str,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            _source: input,
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, LexerError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        self.skip_ws_and_comments();

        let start_pos = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        let c = match self.peek() {
            Some(ch) => ch,
            None => {
                return Ok(Token {
                    kind: TokenKind::Eof,
                    span: Span::new(start_pos, start_pos, start_line, start_col),
                });
            }
        };

        if self.match_str(">>=") {
            return Ok(self.make_token(TokenKind::Bind, start_pos, start_line, start_col));
        }
        if self.match_str("->") {
            return Ok(self.make_token(TokenKind::Arrow, start_pos, start_line, start_col));
        }
        if self.match_str("=>") {
            return Ok(self.make_token(TokenKind::DoubleArrow, start_pos, start_line, start_col));
        }
        if self.match_str("<<") {
            return Ok(self.make_token(TokenKind::FracOpen, start_pos, start_line, start_col));
        }
        if self.match_str(">>") {
            return Ok(self.make_token(TokenKind::FracClose, start_pos, start_line, start_col));
        }
        if self.match_unicode_fractal_open() {
            return Ok(self.make_token(TokenKind::FracOpen, start_pos, start_line, start_col));
        }
        if self.match_unicode_fractal_close() {
            return Ok(self.make_token(TokenKind::FracClose, start_pos, start_line, start_col));
        }

        // String literal
        if c == '"' {
            return self.lex_string_literal(start_pos, start_line, start_col);
        }

        if c.is_ascii_digit() {
            return self.lex_number_or_foundation(start_pos, start_line, start_col);
        }

        if matches!(c, 'A' | 'B' | 'C' | 'D') {
            if let Some(tok) = self.lex_element(start_pos, start_line, start_col) {
                return Ok(tok);
            }
        }

        if c == 'F' {
            if let Some(tok) = self.lex_foundation_prefixed(start_pos, start_line, start_col) {
                return Ok(tok);
            }
        }

        if c == 'n' && self.peek_n(1) == Some('u') {
            if let Some(tok) = self.lex_noetic(start_pos, start_line, start_col) {
                return Ok(tok);
            }
        }

        if is_ident_start(c) {
            return Ok(self.lex_ident_or_keyword(start_pos, start_line, start_col));
        }

        self.advance();
        let kind = match c {
            ':' => TokenKind::Colon,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '\' => TokenKind::Lambda,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '<' => TokenKind::LAngle,
            '>' => TokenKind::RAngle,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '=' => TokenKind::Equals,
            '+' => TokenKind::Plus,
            
