use tksbytecode::bytecode::{Instruction, Opcode};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VmError {
    StackUnderflow,
    TypeMismatch { expected: &'static str, found: Value },
    MissingOperand { opcode: Opcode, operand: &'static str },
    InvalidOperand { opcode: Opcode, operand: &'static str, value: u64 },
    InvalidLocal { index: usize },
    DivisionByZero,
    UnsupportedOpcode(Opcode),
    NoReturn,
}

#[derive(Debug, Clone)]
pub struct VmState {
    pub code: Vec<Instruction>,
    pub pc: usize,
    pub stack: Vec<Value>,
    pub locals: Vec<Value>,
}

impl VmState {
    pub fn new(code: Vec<Instruction>) -> Self {
        Self {
            code,
            pc: 0,
            stack: Vec::new(),
            locals: Vec::new(),
        }
    }

    pub fn run(&mut self) -> Result<Value, VmError> {
        loop {
            if self.pc >= self.code.len() {
                return Err(VmError::NoReturn);
            }
            let instr = self.code[self.pc].clone();
            self.pc += 1;

            match instr.opcode {
                Opcode::PushInt => {
                    let raw = Self::expect_operand1(&instr)?;
                    let value = i64::try_from(raw).map_err(|_| VmError::InvalidOperand {
                        opcode: instr.opcode,
                        operand: "operand1",
                        value: raw,
                    })?;
                    self.stack.push(Value::Int(value));
                }
                Opcode::PushBool => {
                    let raw = Self::expect_operand1(&instr)?;
                    self.stack.push(Value::Bool(raw != 0));
                }
                Opcode::PushUnit => {
                    self.stack.push(Value::Unit);
                }
                Opcode::Load => {
                    let index = Self::expect_operand1_usize(&instr)?;
                    let value = self
                        .locals
                        .get(index)
                        .cloned()
                        .ok_or(VmError::InvalidLocal { index })?;
                    self.stack.push(value);
                }
                Opcode::Store => {
                    let index = Self::expect_operand1_usize(&instr)?;
                    let value = self.pop()?;
                    if index >= self.locals.len() {
                        self.locals.resize(index + 1, Value::Unit);
                    }
                    self.locals[index] = value;
                }
                Opcode::Add => {
                    let rhs = self.pop_int()?;
                    let lhs = self.pop_int()?;
                    self.stack.push(Value::Int(lhs + rhs));
                }
                Opcode::Sub => {
                    let rhs = self.pop_int()?;
                    let lhs = self.pop_int()?;
                    self.stack.push(Value::Int(lhs - rhs));
                }
                Opcode::Mul => {
                    let rhs = self.pop_int()?;
                    let lhs = self.pop_int()?;
                    self.stack.push(Value::Int(lhs * rhs));
                }
                Opcode::Div => {
                    let rhs = self.pop_int()?;
                    if rhs == 0 {
                        return Err(VmError::DivisionByZero);
                    }
                    let lhs = self.pop_int()?;
                    self.stack.push(Value::Int(lhs / rhs));
                }
                Opcode::Ret => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    return Ok(result);
                }
                _ => {
                    return Err(VmError::UnsupportedOpcode(instr.opcode));
                }
            }
        }
    }

    fn pop(&mut self) -> Result<Value, VmError> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    fn pop_int(&mut self) -> Result<i64, VmError> {
        match self.pop()? {
            Value::Int(value) => Ok(value),
            other => Err(VmError::TypeMismatch {
                expected: "int",
                found: other,
            }),
        }
    }

    fn expect_operand1(instr: &Instruction) -> Result<u64, VmError> {
        instr.operand1.ok_or(VmError::MissingOperand {
            opcode: instr.opcode,
            operand: "operand1",
        })
    }

    fn expect_operand1_usize(instr: &Instruction) -> Result<usize, VmError> {
        let raw = Self::expect_operand1(instr)?;
        usize::try_from(raw).map_err(|_| VmError::InvalidOperand {
            opcode: instr.opcode,
            operand: "operand1",
            value: raw,
        })
    }
}
