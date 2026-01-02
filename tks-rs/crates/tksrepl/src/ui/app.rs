use std::collections::VecDeque;

use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;
use tksir::lower::lower_program;
use tksbytecode::emit::emit;
use tksvm::vm::VmState;

/// Application state for the TKS Console
pub struct App {
    pub input: String,
    pub cursor_pos: usize,
    pub history: VecDeque<String>,
    pub history_index: Option<usize>,
    pub output: Vec<OutputLine>,
    #[allow(dead_code)]
    pub scroll_offset: usize,
    pub should_quit: bool,
}

#[derive(Clone)]
pub enum OutputLine {
    Input(String),
    Result(String),
    Error(String),
    Info(String),
}

impl App {
    pub fn new() -> Self {
        Self {
            input: String::new(),
            cursor_pos: 0,
            history: VecDeque::with_capacity(1000),
            history_index: None,
            output: vec![
                OutputLine::Info("TKS Console v0.1.0".to_string()),
                OutputLine::Info("The Tootra Kabbalistic System".to_string()),
                OutputLine::Info(String::new()),
                OutputLine::Info("Type :help for commands, :quit to exit".to_string()),
                OutputLine::Info(String::new()),
            ],
            scroll_offset: 0,
            should_quit: false,
        }
    }

    pub fn submit(&mut self) {
        let input = self.input.trim().to_string();
        if input.is_empty() {
            return;
        }

        // Add to history
        self.history.push_back(input.clone());
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
        self.history_index = None;

        // Handle commands
        if input.starts_with(':') {
            self.handle_command(&input);
        } else {
            self.execute_tks(&input);
        }

        // Clear input
        self.input.clear();
        self.cursor_pos = 0;
    }

    fn handle_command(&mut self, cmd: &str) {
        match cmd {
            ":quit" | ":q" => {
                self.should_quit = true;
            }
            ":clear" | ":c" => {
                self.output.clear();
            }
            ":help" | ":h" => {
                self.output.push(OutputLine::Info(String::new()));
                self.output.push(OutputLine::Info("Commands:".to_string()));
                self.output.push(OutputLine::Info("  :help, :h    Show this help".to_string()));
                self.output.push(OutputLine::Info("  :clear, :c   Clear output".to_string()));
                self.output.push(OutputLine::Info("  :quit, :q    Exit console".to_string()));
                self.output.push(OutputLine::Info(String::new()));
                self.output.push(OutputLine::Info("TKS Language Features:".to_string()));
                self.output.push(OutputLine::Info("  let x = 42           Variable binding".to_string()));
                self.output.push(OutputLine::Info("  \\x -> x + 1          Lambda function".to_string()));
                self.output.push(OutputLine::Info("  if cond then a else b Conditional".to_string()));
                self.output.push(OutputLine::Info("  blueprint Name {...}  OOP class".to_string()));
                self.output.push(OutputLine::Info("  repeat Name {...}     Constructor".to_string()));
                self.output.push(OutputLine::Info("  omega, epsilon        Ordinals".to_string()));
                self.output.push(OutputLine::Info("  superpose {...}       Quantum state".to_string()));
                self.output.push(OutputLine::Info(String::new()));
            }
            _ => {
                self.output.push(OutputLine::Error(format!("Unknown command: {}", cmd)));
            }
        }
    }

    fn execute_tks(&mut self, source: &str) {
        self.output.push(OutputLine::Input(source.to_string()));

        match self.run_pipeline(source) {
            Ok(value) => {
                self.output.push(OutputLine::Result(format!("{:?}", value)));
            }
            Err(e) => {
                self.output.push(OutputLine::Error(e));
            }
        }
    }

    fn run_pipeline(&self, source: &str) -> Result<tksvm::vm::Value, String> {
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

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }

        match self.history_index {
            None => {
                self.history_index = Some(self.history.len() - 1);
            }
            Some(0) => {}
            Some(i) => {
                self.history_index = Some(i - 1);
            }
        }

        if let Some(i) = self.history_index {
            if let Some(cmd) = self.history.get(i) {
                self.input = cmd.clone();
                self.cursor_pos = self.input.len();
            }
        }
    }

    pub fn history_down(&mut self) {
        match self.history_index {
            None => {}
            Some(i) if i + 1 >= self.history.len() => {
                self.history_index = None;
                self.input.clear();
                self.cursor_pos = 0;
            }
            Some(i) => {
                self.history_index = Some(i + 1);
                if let Some(cmd) = self.history.get(i + 1) {
                    self.input = cmd.clone();
                    self.cursor_pos = self.input.len();
                }
            }
        }
    }

    pub fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor_pos, c);
        self.cursor_pos += 1;
    }

    pub fn delete_char(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            self.input.remove(self.cursor_pos);
        }
    }

    pub fn move_cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    pub fn move_cursor_right(&mut self) {
        if self.cursor_pos < self.input.len() {
            self.cursor_pos += 1;
        }
    }

    pub fn move_cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    pub fn move_cursor_end(&mut self) {
        self.cursor_pos = self.input.len();
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
