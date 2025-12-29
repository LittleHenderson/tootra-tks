use tksbytecode::bytecode::Instruction;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone)]
pub struct VmState {
    pub code: Vec<Instruction>,
    pub pc: usize,
    pub stack: Vec<Value>,
}

impl VmState {
    pub fn new(code: Vec<Instruction>) -> Self {
        Self {
            code,
            pc: 0,
            stack: Vec::new(),
        }
    }
}
