use crate::ast::{Aspect, Ident, World};
use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Element { world: World, index: u8 },
    Noetic { index: u8 },
    Foundation { level: u8, aspect: Aspect },
    Int(i64),
    Bool(bool),
    Float(f64),
    Complex { re: f64, im: f64 },
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
            '\\' => TokenKind::Lambda,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '<' => TokenKind::LAngle,
            '>' => TokenKind::RAngle,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '=' => TokenKind::Equals,
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Times,
            '/' => TokenKind::Divide,
            '^' => TokenKind::Caret,
            '!' => TokenKind::Bang,
            '|' => TokenKind::Pipe,
            '.' => TokenKind::Dot,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            '_' => TokenKind::Underscore,
            _ => {
                return Err(LexerError::new(format!("unexpected character '{c}'")));
            }
        };

        Ok(self.make_token(kind, start_pos, start_line, start_col))
    }

    fn lex_number_or_foundation(
        &mut self,
        start_pos: usize,
        start_line: u32,
        start_col: u32,
    ) -> Result<Token, LexerError> {
        let number = self.read_while(|ch| ch.is_ascii_digit());

        if number.len() == 1 {
            if let Some(next) = self.peek() {
                if matches!(next, 'a' | 'b' | 'c' | 'd') {
                    let aspect = self.advance().unwrap();
                    if let Ok(level) = number.parse::<u8>() {
                        if (1..=7).contains(&level) {
                            return Ok(Token {
                                kind: TokenKind::Foundation {
                                    level,
                                    aspect: aspect_from_char(aspect),
                                },
                                span: Span::new(start_pos, self.pos, start_line, start_col),
                            });
                        }
                    }
                }
            }
        }

        if self.peek() == Some('.') && self.peek_n(1).map(|c| c.is_ascii_digit()).unwrap_or(false) {
            self.advance();
            let frac = self.read_while(|ch| ch.is_ascii_digit());
            let text = format!("{number}.{frac}");
            let value = text.parse::<f64>().map_err(|_| {
                LexerError::new(format!("invalid float literal '{text}'"))
            })?;
            if self.peek() == Some('i') {
                self.advance();
                return Ok(Token {
                    kind: TokenKind::Complex { re: 0.0, im: value },
                    span: Span::new(start_pos, self.pos, start_line, start_col),
                });
            }
            return Ok(Token {
                kind: TokenKind::Float(value),
                span: Span::new(start_pos, self.pos, start_line, start_col),
            });
        }

        let value = number.parse::<i64>().map_err(|_| {
            LexerError::new(format!("invalid integer literal '{number}'"))
        })?;
        Ok(Token {
            kind: TokenKind::Int(value),
            span: Span::new(start_pos, self.pos, start_line, start_col),
        })
    }

    fn lex_element(&mut self, start_pos: usize, start_line: u32, start_col: u32) -> Option<Token> {
        let snapshot = (self.pos, self.line, self.column);
        let world = self.advance()?;
        let digits = self.read_while(|ch| ch.is_ascii_digit());
        if digits.is_empty() {
            self.pos = snapshot.0;
            self.line = snapshot.1;
            self.column = snapshot.2;
            return None;
        }
        let value = match digits.parse::<u8>() {
            Ok(value) => value,
            Err(_) => {
                self.pos = snapshot.0;
                self.line = snapshot.1;
                self.column = snapshot.2;
                return None;
            }
        };
        if !(1..=10).contains(&value) {
            self.pos = snapshot.0;
            self.line = snapshot.1;
            self.column = snapshot.2;
            return None;
        }
        Some(Token {
            kind: TokenKind::Element {
                world: world_from_char(world),
                index: value,
            },
            span: Span::new(start_pos, self.pos, start_line, start_col),
        })
    }

    fn lex_foundation_prefixed(
        &mut self,
        start_pos: usize,
        start_line: u32,
        start_col: u32,
    ) -> Option<Token> {
        let snapshot = (self.pos, self.line, self.column);
        let _prefix = self.advance()?;
        let digit = self.advance()?;
        let aspect = self.advance()?;
        if !digit.is_ascii_digit() || !matches!(aspect, 'a' | 'b' | 'c' | 'd') {
            self.pos = snapshot.0;
            self.line = snapshot.1;
            self.column = snapshot.2;
            return None;
        }
        let level = digit.to_digit(10)? as u8;
        if !(1..=7).contains(&level) {
            self.pos = snapshot.0;
            self.line = snapshot.1;
            self.column = snapshot.2;
            return None;
        }
        Some(Token {
            kind: TokenKind::Foundation {
                level,
                aspect: aspect_from_char(aspect),
            },
            span: Span::new(start_pos, self.pos, start_line, start_col),
        })
    }

    fn lex_noetic(&mut self, start_pos: usize, start_line: u32, start_col: u32) -> Option<Token> {
        let snapshot = (self.pos, self.line, self.column);
        let _n = self.advance()?;
        let _u = self.advance()?;
        let digits = self.read_while(|ch| ch.is_ascii_digit());
        if digits.is_empty() {
            return Some(Token {
                kind: TokenKind::Nu,
                span: Span::new(start_pos, self.pos, start_line, start_col),
            });
        }
        let value = match digits.parse::<u8>() {
            Ok(value) => value,
            Err(_) => {
                self.pos = snapshot.0;
                self.line = snapshot.1;
                self.column = snapshot.2;
                return None;
            }
        };
        if value > 9 {
            self.pos = snapshot.0;
            self.line = snapshot.1;
            self.column = snapshot.2;
            return None;
        }
        Some(Token {
            kind: TokenKind::Noetic { index: value },
            span: Span::new(start_pos, self.pos, start_line, start_col),
        })
    }

    fn lex_ident_or_keyword(
        &mut self,
        start_pos: usize,
        start_line: u32,
        start_col: u32,
    ) -> Token {
        let ident = self.read_while(is_ident_continue);
        let kind = match ident.as_str() {
            "let" => TokenKind::Let,
            "in" => TokenKind::In,
            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            "return" => TokenKind::Return,
            "check" => TokenKind::Check,
            "acquire" => TokenKind::Acquire,
            "acbe" => TokenKind::Acbe,
            "effect" => TokenKind::Effect,
            "handle" => TokenKind::Handle,
            "with" => TokenKind::With,
            "resume" => TokenKind::Resume,
            "perform" => TokenKind::Perform,
            "handler" => TokenKind::Handler,
            "op" => TokenKind::Op,
            "module" => TokenKind::Module,
            "import" => TokenKind::Import,
            "export" => TokenKind::Export,
            "from" => TokenKind::From,
            "step" => TokenKind::Step,
            "as" => TokenKind::As,
            "type" => TokenKind::TypeKw,
            "extern" => TokenKind::External,
            "external" => TokenKind::External,
            "fn" => TokenKind::FnKw,
            "safe" => TokenKind::Safe,
            "unsafe" => TokenKind::Unsafe,
            "omega" => TokenKind::Omega,
            "epsilon" => TokenKind::Epsilon,
            "aleph" => TokenKind::Aleph,
            "sup" => TokenKind::Sup,
            "ord" => TokenKind::Ord,
            "limit" => TokenKind::Limit,
            "succ" => TokenKind::Succ,
            "transfinite" => TokenKind::Transfinite,
            "loop" => TokenKind::Loop,
            "superpose" => TokenKind::Superpose,
            "measure" => TokenKind::Measure,
            "entangle" => TokenKind::Entangle,
            "qstate" => TokenKind::QState,
            "amplitude" => TokenKind::Amplitude,
            "basis" => TokenKind::Basis,
            "true" => TokenKind::Bool(true),
            "false" => TokenKind::Bool(false),
            _ => TokenKind::Ident(ident),
        };
        Token {
            kind,
            span: Span::new(start_pos, self.pos, start_line, start_col),
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            self.skip_whitespace();
            if self.peek() == Some('-') && self.peek_n(1) == Some('-') {
                while let Some(ch) = self.advance() {
                    if ch == '\n' {
                        break;
                    }
                }
                continue;
            }
            break;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_while<F>(&mut self, mut predicate: F) -> String
    where
        F: FnMut(char) -> bool,
    {
        let mut out = String::new();
        while let Some(ch) = self.peek() {
            if !predicate(ch) {
                break;
            }
            out.push(ch);
            self.advance();
        }
        out
    }

    fn make_token(&self, kind: TokenKind, start_pos: usize, line: u32, col: u32) -> Token {
        Token {
            kind,
            span: Span::new(start_pos, self.pos, line, col),
        }
    }

    fn match_str(&mut self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        if self.pos + chars.len() > self.input.len() {
            return false;
        }
        for (i, ch) in chars.iter().enumerate() {
            if self.input[self.pos + i] != *ch {
                return false;
            }
        }
        for _ in 0..chars.len() {
            self.advance();
        }
        true
    }

    fn match_unicode_fractal_open(&mut self) -> bool {
        let open = ['\u{0192}', 'Y', '"'];
        if self.pos + open.len() > self.input.len() {
            return false;
        }
        for (i, ch) in open.iter().enumerate() {
            if self.input[self.pos + i] != *ch {
                return false;
            }
        }
        for _ in 0..open.len() {
            self.advance();
        }
        true
    }

    fn match_unicode_fractal_close(&mut self) -> bool {
        let close = ['\u{0192}', 'Y', 'c'];
        if self.pos + close.len() > self.input.len() {
            return false;
        }
        for (i, ch) in close.iter().enumerate() {
            if self.input[self.pos + i] != *ch {
                return false;
            }
        }
        for _ in 0..close.len() {
            self.advance();
        }
        true
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_n(&self, n: usize) -> Option<char> {
        self.input.get(self.pos + n).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied();
        if let Some(c) = ch {
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        ch
    }
}

fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn world_from_char(ch: char) -> World {
    match ch {
        'A' => World::A,
        'B' => World::B,
        'C' => World::C,
        _ => World::D,
    }
}

fn aspect_from_char(ch: char) -> Aspect {
    match ch {
        'a' => Aspect::A,
        'b' => Aspect::B,
        'c' => Aspect::C,
        _ => Aspect::D,
    }
}

#[derive(Debug, Clone)]
pub struct LexerError {
    pub message: String,
}

impl LexerError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}


