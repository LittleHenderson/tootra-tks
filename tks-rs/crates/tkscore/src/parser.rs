use crate::ast::Program;
use crate::lexer::Token;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParserError> {
        let _ = self;
        Err(ParserError::new("parser not implemented"))
    }
}

#[derive(Debug, Clone)]
pub struct ParserError {
    pub message: String,
}

impl ParserError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
