use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

use tkscore::parser::parse_program;
use tkscore::lexer::Lexer;
use tkstypes::infer::infer_program;
use tksir::lower::lower_program;
use tksbytecode::emit::emit;
use tksvm::vm::VmState;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Result of executing TKS code
#[derive(Serialize, Deserialize)]
pub struct ExecuteResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub execution_time_ms: f64,
}

/// Token information for syntax highlighting
#[derive(Serialize, Deserialize)]
pub struct TokenInfo {
    pub start: usize,
    pub end: usize,
    pub kind: String,
    pub class: String,
}

/// Highlighting result
#[derive(Serialize, Deserialize)]
pub struct HighlightResult {
    pub tokens: Vec<TokenInfo>,
}

/// Example code snippet
#[derive(Serialize, Deserialize)]
pub struct Example {
    pub name: String,
    pub description: String,
    pub code: String,
}

/// Execute TKS source code and return the result
#[wasm_bindgen]
pub fn execute_tks(source: &str) -> JsValue {
    let start = js_sys::Date::now();

    let result = match run_pipeline(source) {
        Ok(value) => ExecuteResult {
            success: true,
            output: format_value(&value),
            error: None,
            execution_time_ms: js_sys::Date::now() - start,
        },
        Err(e) => ExecuteResult {
            success: false,
            output: String::new(),
            error: Some(e),
            execution_time_ms: js_sys::Date::now() - start,
        },
    };

    serde_wasm_bindgen::to_value(&result).unwrap()
}

/// Tokenize TKS source for syntax highlighting
#[wasm_bindgen]
pub fn highlight_tks(source: &str) -> JsValue {
    let lexer = Lexer::new(source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(_) => return serde_wasm_bindgen::to_value(&HighlightResult { tokens: vec![] }).unwrap(),
    };

    let token_infos: Vec<TokenInfo> = tokens
        .iter()
        .map(|t| TokenInfo {
            start: t.span.start,
            end: t.span.end,
            kind: format!("{:?}", t.kind),
            class: token_class(&t.kind),
        })
        .collect();

    serde_wasm_bindgen::to_value(&HighlightResult { tokens: token_infos }).unwrap()
}

/// Get built-in code examples
#[wasm_bindgen]
pub fn get_examples() -> JsValue {
    let examples = vec![
        Example {
            name: "Hello TKS".to_string(),
            description: "Simple arithmetic".to_string(),
            code: "let x = 42;\nlet y = x + 8;\ny".to_string(),
        },
        Example {
            name: "Lambda".to_string(),
            description: "Anonymous function".to_string(),
            code: "let double = \\x -> x + x;\ndouble 21".to_string(),
        },
        Example {
            name: "Conditional".to_string(),
            description: "If-then-else expression".to_string(),
            code: "if true then 1 else 0".to_string(),
        },
        Example {
            name: "Blueprint".to_string(),
            description: "OOP class example".to_string(),
            code: "blueprint Counter {\n  specifics { value: Int; }\n  details { doubled: Int = identity.value; }\n  actions { }\n}\n\nrepeat Counter { value: 42 }".to_string(),
        },
        Example {
            name: "Quantum".to_string(),
            description: "Superposition state".to_string(),
            code: "let state = superpose {\n  1: |3>,\n  2: |5>\n};\nmeasure(state)".to_string(),
        },
        Example {
            name: "Ordinal".to_string(),
            description: "Transfinite arithmetic".to_string(),
            code: "|omega>".to_string(),
        },
        Example {
            name: "Foundation".to_string(),
            description: "Kabbalistic levels".to_string(),
            code: "3a".to_string(),
        },
        Example {
            name: "Element".to_string(),
            description: "World elements".to_string(),
            code: "A5".to_string(),
        },
    ];

    serde_wasm_bindgen::to_value(&examples).unwrap()
}

/// Get list of TKS keywords for autocomplete
#[wasm_bindgen]
pub fn get_keywords() -> JsValue {
    let keywords = vec![
        "let", "in", "if", "then", "else", "return",
        "blueprint", "class", "plan", "specifics", "details", "actions",
        "method", "new", "repeat", "self", "identity",
        "effect", "handle", "with", "perform", "resume",
        "omega", "epsilon", "aleph", "superpose", "measure", "entangle",
        "true", "false",
    ];

    serde_wasm_bindgen::to_value(&keywords).unwrap()
}

fn run_pipeline(source: &str) -> Result<tksvm::vm::Value, String> {
    let program = parse_program(source)
        .map_err(|e| {
            let loc = e.span.map(|s| format!("{}:{}: ", s.line, s.column)).unwrap_or_default();
            format!("{}Parse error: {}", loc, e.message)
        })?;

    infer_program(&program)
        .map_err(|e| format!("Type error: {}", e))?;

    let ir = lower_program(&program)
        .map_err(|e| format!("Lower error: {}", e))?;

    let bytecode = emit(&ir)
        .map_err(|e| format!("Emit error: {}", e))?;

    let mut vm = VmState::new(bytecode);
    vm.run()
        .map_err(|e| format!("Runtime error: {:?}", e))
}

fn format_value(value: &tksvm::vm::Value) -> String {
    use tksvm::vm::Value;
    match value {
        Value::Int(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Unit => "()".to_string(),
        Value::Record { .. } => format!("{:?}", value),
        Value::Ket(inner) => format!("|{}>", format_value(inner)),
        Value::Ordinal(ord) => format!("{:?}", ord),
        Value::Foundation { level, aspect } => format!("{}a{}", level, aspect),
        Value::Element { world, index } => {
            let w = (b'A' + *world) as char;
            format!("{}{}", w, index)
        }
        Value::Noetic(n) => format!("nu{}", n),
        _ => format!("{:?}", value),
    }
}

fn token_class(kind: &tkscore::lexer::TokenKind) -> String {
    use tkscore::lexer::TokenKind;

    match kind {
        // Keywords
        TokenKind::Let | TokenKind::In | TokenKind::If | TokenKind::Then |
        TokenKind::Else | TokenKind::Return | TokenKind::Handle | TokenKind::With |
        TokenKind::Effect | TokenKind::Perform | TokenKind::Resume |
        TokenKind::Blueprint | TokenKind::Class | TokenKind::Plan |
        TokenKind::Specifics | TokenKind::Details | TokenKind::Actions |
        TokenKind::Method | TokenKind::New | TokenKind::Repeat |
        TokenKind::SelfKw | TokenKind::Identity => "keyword".to_string(),

        // Literals
        TokenKind::Int(_) | TokenKind::Float(_) => "literal".to_string(),
        TokenKind::Bool(_) => "keyword".to_string(),

        // Domain-specific
        TokenKind::Element { .. } => "element".to_string(),
        TokenKind::Noetic { .. } | TokenKind::Nu => "noetic".to_string(),
        TokenKind::Foundation { .. } => "foundation".to_string(),
        TokenKind::Omega | TokenKind::Epsilon | TokenKind::Aleph |
        TokenKind::Superpose | TokenKind::Measure | TokenKind::Entangle => "quantum".to_string(),

        // Operators
        TokenKind::Lambda | TokenKind::Arrow | TokenKind::DoubleArrow |
        TokenKind::Bind | TokenKind::Plus | TokenKind::Minus |
        TokenKind::Times | TokenKind::Divide | TokenKind::Equals => "operator".to_string(),

        // Default
        _ => "default".to_string(),
    }
}
