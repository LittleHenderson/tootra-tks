use tksbytecode::bytecode::{Instruction, Opcode};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Unit,
    Closure { entry: usize },
    Element { world: u8, index: u8 },
    Noetic(u8),
    NoeticApplied { index: u8, value: Box<Value> },
    Ket(Box<Value>),
    Superpose(Vec<(Value, Value)>),
    Entangle(Box<Value>, Box<Value>),
    Ordinal(OrdinalValue),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrdinalValue {
    Finite(u64),
    Omega,
    Epsilon(u64),
    Succ(Box<OrdinalValue>),
    Add(Box<OrdinalValue>, Box<OrdinalValue>),
    Mul(Box<OrdinalValue>, Box<OrdinalValue>),
    Exp(Box<OrdinalValue>, Box<OrdinalValue>),
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
pub struct CallFrame {
    pub return_pc: usize,
    pub locals: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct VmState {
    pub code: Vec<Instruction>,
    pub pc: usize,
    pub stack: Vec<Value>,
    pub locals: Vec<Value>,
    pub frames: Vec<CallFrame>,
}

impl VmState {
    pub fn new(code: Vec<Instruction>) -> Self {
        Self {
            code,
            pc: 0,
            stack: Vec::new(),
            locals: Vec::new(),
            frames: Vec::new(),
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
                Opcode::PushClosure => {
                    let entry = Self::expect_operand1_usize(&instr)?;
                    self.stack.push(Value::Closure { entry });
                }
                Opcode::PushElement => {
                    let world_raw = Self::expect_operand1(&instr)?;
                    let world = match world_raw {
                        0..=3 => world_raw as u8,
                        _ => {
                            return Err(VmError::InvalidOperand {
                                opcode: instr.opcode,
                                operand: "operand1",
                                value: world_raw,
                            })
                        }
                    };
                    let index = Self::expect_operand2_u8(&instr)?;
                    self.stack.push(Value::Element { world, index });
                }
                Opcode::PushNoetic => {
                    let index = Self::expect_operand1_u8(&instr)?;
                    self.stack.push(Value::Noetic(index));
                }
                Opcode::PushOrd => {
                    let value = Self::expect_operand1(&instr)?;
                    self.stack.push(Value::Ordinal(OrdinalValue::Finite(value)));
                }
                Opcode::PushOmega => {
                    self.stack.push(Value::Ordinal(OrdinalValue::Omega));
                }
                Opcode::PushEpsilon => {
                    let value = Self::expect_operand1(&instr)?;
                    self.stack.push(Value::Ordinal(OrdinalValue::Epsilon(value)));
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
                Opcode::Jmp => {
                    let target = Self::expect_operand1_usize(&instr)?;
                    self.pc = target;
                }
                Opcode::JmpIf => {
                    let target = Self::expect_operand1_usize(&instr)?;
                    let cond = self.pop_bool()?;
                    if cond {
                        self.pc = target;
                    }
                }
                Opcode::JmpUnless => {
                    let target = Self::expect_operand1_usize(&instr)?;
                    let cond = self.pop_bool()?;
                    if !cond {
                        self.pc = target;
                    }
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
                Opcode::OrdSucc => {
                    let inner = self.pop_ordinal()?;
                    self.stack
                        .push(Value::Ordinal(OrdinalValue::Succ(Box::new(inner))));
                }
                Opcode::OrdAdd => {
                    let rhs = self.pop_ordinal()?;
                    let lhs = self.pop_ordinal()?;
                    self.stack
                        .push(Value::Ordinal(OrdinalValue::Add(Box::new(lhs), Box::new(rhs))));
                }
                Opcode::OrdMul => {
                    let rhs = self.pop_ordinal()?;
                    let lhs = self.pop_ordinal()?;
                    self.stack
                        .push(Value::Ordinal(OrdinalValue::Mul(Box::new(lhs), Box::new(rhs))));
                }
                Opcode::OrdExp => {
                    let rhs = self.pop_ordinal()?;
                    let lhs = self.pop_ordinal()?;
                    self.stack
                        .push(Value::Ordinal(OrdinalValue::Exp(Box::new(lhs), Box::new(rhs))));
                }
                Opcode::Call => {
                    let arg = self.pop()?;
                    let callee = self.pop()?;
                    let entry = match callee {
                        Value::Closure { entry } => entry,
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "closure",
                                found: other,
                            })
                        }
                    };
                    let locals = std::mem::take(&mut self.locals);
                    self.frames.push(CallFrame {
                        return_pc: self.pc,
                        locals,
                    });
                    self.locals = vec![arg];
                    self.pc = entry;
                }
                Opcode::ApplyNoetic => {
                    let arg = self.pop()?;
                    let callee = self.pop()?;
                    let index = match callee {
                        Value::Noetic(index) => index,
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "noetic",
                                found: other,
                            })
                        }
                    };
                    self.stack.push(Value::NoeticApplied {
                        index,
                        value: Box::new(arg),
                    });
                }
                Opcode::MakeKet => {
                    let inner = self.pop()?;
                    self.stack.push(Value::Ket(Box::new(inner)));
                }
                Opcode::Superpose => {
                    let count = Self::expect_operand1_usize(&instr)?;
                    let mut states = Vec::with_capacity(count);
                    for _ in 0..count {
                        let ket = self.pop()?;
                        let amp = self.pop()?;
                        states.push((amp, ket));
                    }
                    states.reverse();
                    self.stack.push(Value::Superpose(states));
                }
                Opcode::Measure => {
                    let value = self.pop()?;
                    match value {
                        Value::Ket(inner) => {
                            self.stack.push(*inner);
                        }
                        Value::Superpose(states) => {
                            if states.is_empty() {
                                return Err(VmError::TypeMismatch {
                                    expected: "non-empty superpose",
                                    found: Value::Superpose(states),
                                });
                            }
                            let ket = states.into_iter().next().map(|(_, ket)| ket).unwrap();
                            self.stack.push(ket);
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "ket or superpose",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::Entangle => {
                    let right = self.pop()?;
                    let left = self.pop()?;
                    self.stack
                        .push(Value::Entangle(Box::new(left), Box::new(right)));
                }
                Opcode::Ret => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    if let Some(frame) = self.frames.pop() {
                        self.locals = frame.locals;
                        self.pc = frame.return_pc;
                        self.stack.push(result);
                    } else {
                        return Ok(result);
                    }
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

    fn pop_bool(&mut self) -> Result<bool, VmError> {
        match self.pop()? {
            Value::Bool(value) => Ok(value),
            other => Err(VmError::TypeMismatch {
                expected: "bool",
                found: other,
            }),
        }
    }

    fn pop_ordinal(&mut self) -> Result<OrdinalValue, VmError> {
        match self.pop()? {
            Value::Ordinal(value) => Ok(value),
            other => Err(VmError::TypeMismatch {
                expected: "ordinal",
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

    fn expect_operand1_u8(instr: &Instruction) -> Result<u8, VmError> {
        let raw = Self::expect_operand1(instr)?;
        u8::try_from(raw).map_err(|_| VmError::InvalidOperand {
            opcode: instr.opcode,
            operand: "operand1",
            value: raw,
        })
    }

    fn expect_operand2(instr: &Instruction) -> Result<u64, VmError> {
        instr.operand2.ok_or(VmError::MissingOperand {
            opcode: instr.opcode,
            operand: "operand2",
        })
    }

    fn expect_operand2_u8(instr: &Instruction) -> Result<u8, VmError> {
        let raw = Self::expect_operand2(instr)?;
        u8::try_from(raw).map_err(|_| VmError::InvalidOperand {
            opcode: instr.opcode,
            operand: "operand2",
            value: raw,
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
