use std::collections::HashMap;
use std::sync::Arc;

use tksbytecode::bytecode::{Instruction, Opcode};
use tksbytecode::extern_id;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Unit,
    Closure { entry: usize },
    Extern { id: u64, arity: usize, args: Vec<Value> },
    Handler {
        return_entry: usize,
        op_clauses: Vec<(u64, usize)>,
    },
    Continuation {
        pc: usize,
        stack: Vec<Value>,
        locals: Vec<Value>,
        frames: Vec<CallFrame>,
        handlers: Vec<HandlerFrame>,
    },
    Element { world: u8, index: u8 },
    Foundation { level: u8, aspect: u8 },
    Noetic(u8),
    NoeticApplied { index: u8, value: Box<Value> },
    Fractal {
        seq: Vec<u8>,
        ellipsis: Option<u8>,
        subscript: Option<OrdinalValue>,
        value: Box<Value>,
    },
    AcbeState {
        goal: Box<Value>,
        current: Box<Value>,
        complete: bool,
    },
    RpmState { ok: bool, value: Box<Value> },
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
    UnknownExtern(u64),
    AcbeIncomplete,
    HandlerNotImplemented,
    UnhandledEffect,
    RpmUnwrapFailed,
    UnsupportedOpcode(Opcode),
    NoReturn,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallFrameKind {
    Normal,
    RpmBind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallFrame {
    pub return_pc: usize,
    pub locals: Vec<Value>,
    pub kind: CallFrameKind,
    pub handler: Option<HandlerFrame>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandlerFrame {
    pub return_entry: usize,
    pub op_clauses: Vec<(u64, usize)>,
}

pub type ExternRegistry = HashMap<u64, Arc<dyn Fn(Vec<Value>) -> Result<Value, VmError>>>;

#[derive(Clone)]
pub struct VmState {
    pub code: Vec<Instruction>,
    pub pc: usize,
    pub stack: Vec<Value>,
    pub locals: Vec<Value>,
    pub frames: Vec<CallFrame>,
    pub handlers: Vec<HandlerFrame>,
    pub externs: ExternRegistry,
}

impl std::fmt::Debug for VmState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VmState")
            .field("code", &self.code)
            .field("pc", &self.pc)
            .field("stack", &self.stack)
            .field("locals", &self.locals)
            .field("frames", &self.frames)
            .field("handlers", &self.handlers)
            .field("externs", &self.externs.len())
            .finish()
    }
}

impl VmState {
    pub fn new(code: Vec<Instruction>) -> Self {
        Self {
            code,
            pc: 0,
            stack: Vec::new(),
            locals: Vec::new(),
            frames: Vec::new(),
            handlers: Vec::new(),
            externs: HashMap::new(),
        }
    }

    pub fn with_externs(code: Vec<Instruction>, externs: ExternRegistry) -> Self {
        Self {
            code,
            pc: 0,
            stack: Vec::new(),
            locals: Vec::new(),
            frames: Vec::new(),
            handlers: Vec::new(),
            externs,
        }
    }

    pub fn register_extern<F>(&mut self, name: &str, func: F)
    where
        F: Fn(Vec<Value>) -> Result<Value, VmError> + 'static,
    {
        let id = extern_id(name);
        self.externs.insert(id, Arc::new(func));
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
                Opcode::PushFoundation => {
                    let level_raw = Self::expect_operand1(&instr)?;
                    let level = match level_raw {
                        1..=7 => level_raw as u8,
                        _ => {
                            return Err(VmError::InvalidOperand {
                                opcode: instr.opcode,
                                operand: "operand1",
                                value: level_raw,
                            })
                        }
                    };
                    let aspect_raw = Self::expect_operand2(&instr)?;
                    let aspect = match aspect_raw {
                        0..=3 => aspect_raw as u8,
                        _ => {
                            return Err(VmError::InvalidOperand {
                                opcode: instr.opcode,
                                operand: "operand2",
                                value: aspect_raw,
                            })
                        }
                    };
                    self.stack.push(Value::Foundation { level, aspect });
                }
                Opcode::PushNoetic => {
                    let index = Self::expect_operand1_u8(&instr)?;
                    self.stack.push(Value::Noetic(index));
                }
                Opcode::PushExtern => {
                    let id = Self::expect_operand1(&instr)?;
                    let arity_raw = Self::expect_operand2(&instr)?;
                    let arity = usize::try_from(arity_raw).map_err(|_| VmError::InvalidOperand {
                        opcode: instr.opcode,
                        operand: "operand2",
                        value: arity_raw,
                    })?;
                    self.stack.push(Value::Extern {
                        id,
                        arity,
                        args: Vec::new(),
                    });
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
                Opcode::LoadGlobal => {
                    let id = Self::expect_operand1(&instr)?;
                    self.stack.push(Value::Extern {
                        id,
                        arity: 1,
                        args: Vec::new(),
                    });
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
                    match callee {
                        Value::Closure { entry } => {
                            self.enter_call(entry, arg, CallFrameKind::Normal);
                        }
                        cont @ Value::Continuation { .. } => {
                            self.resume_continuation(cont, arg)?;
                        }
                        Value::Extern { id, arity, mut args } => {
                            args.push(arg);
                            if args.len() < arity {
                                self.stack.push(Value::Extern { id, arity, args });
                            } else {
                                let value = self.call_extern(id, args)?;
                                self.stack.push(value);
                            }
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "closure, continuation, or extern",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::FoundationLevel => {
                    let value = self.pop()?;
                    match value {
                        Value::Foundation { level, .. } => {
                            self.stack.push(Value::Int(level as i64));
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "foundation",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::FoundationAspect => {
                    let value = self.pop()?;
                    match value {
                        Value::Foundation { aspect, .. } => {
                            self.stack.push(Value::Int(aspect as i64));
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "foundation",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::NoeticCompose => {
                    let right = self.pop()?;
                    let left = self.pop()?;
                    match (left, right) {
                        (Value::Noetic(left), Value::Noetic(right)) => {
                            let index = ((u16::from(left) + u16::from(right)) % 10) as u8;
                            self.stack.push(Value::Noetic(index));
                        }
                        (Value::Noetic(_), other) | (other, _) => {
                            return Err(VmError::TypeMismatch {
                                expected: "noetic",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::NoeticLevel => {
                    let value = self.pop()?;
                    match value {
                        Value::Noetic(index) => {
                            self.stack.push(Value::Int(index as i64));
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "noetic",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::MakeFractal => {
                    let count = Self::expect_operand1_usize(&instr)?;
                    let flags = instr.operand2.unwrap_or(0);
                    let has_subscript = flags & 1 == 1;
                    let has_ellipsis = flags & 2 == 2;
                    let value = self.pop()?;
                    let subscript = if has_subscript {
                        Some(self.pop_ordinal()?)
                    } else {
                        None
                    };
                    let mut seq = Vec::with_capacity(count);
                    for _ in 0..count {
                        let value = self.pop()?;
                        match value {
                            Value::Noetic(index) => seq.push(index),
                            other => {
                                return Err(VmError::TypeMismatch {
                                    expected: "noetic",
                                    found: other,
                                })
                            }
                        }
                    }
                    seq.reverse();
                    if has_ellipsis && seq.is_empty() {
                        return Err(VmError::TypeMismatch {
                            expected: "non-empty fractal sequence",
                            found: Value::Unit,
                        });
                    }
                    let ellipsis = if has_ellipsis {
                        seq.last().copied()
                    } else {
                        None
                    };
                    self.stack.push(Value::Fractal {
                        seq,
                        ellipsis,
                        subscript,
                        value: Box::new(value),
                    });
                }
                Opcode::FractalLevel => {
                    let value = self.pop()?;
                    match value {
                        Value::Fractal { seq, subscript, .. } => {
                            let level = subscript.unwrap_or_else(|| {
                                OrdinalValue::Finite(seq.len() as u64)
                            });
                            self.stack.push(Value::Ordinal(level));
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "fractal",
                                found: other,
                            })
                        }
                    }
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
                Opcode::AcbeInit => {
                    let expr = self.pop()?;
                    let goal = self.pop()?;
                    self.stack.push(Value::AcbeState {
                        goal: Box::new(goal),
                        current: Box::new(expr),
                        complete: false,
                    });
                }
                Opcode::AcbeDescend => {
                    let current = self.pop()?;
                    let state = self.pop()?;
                    match state {
                        Value::AcbeState { goal, .. } => {
                            self.stack.push(Value::AcbeState {
                                goal,
                                current: Box::new(current),
                                complete: false,
                            });
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "acbe state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::AcbeComplete => {
                    let state = self.pop()?;
                    match state {
                        Value::AcbeState {
                            goal,
                            current,
                            ..
                        } => {
                            self.stack.push(Value::AcbeState {
                                goal,
                                current,
                                complete: true,
                            });
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "acbe state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::AcbeResult => {
                    let state = self.pop()?;
                    match state {
                        Value::AcbeState { current, complete, .. } => {
                            if complete {
                                self.stack.push(*current);
                            } else {
                                return Err(VmError::AcbeIncomplete);
                            }
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "acbe state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::RpmAcquire => {
                    let value = self.pop()?;
                    self.stack.push(Value::RpmState {
                        ok: true,
                        value: Box::new(value),
                    });
                }
                Opcode::RpmReturn => {
                    let value = self.pop()?;
                    self.stack.push(Value::RpmState {
                        ok: true,
                        value: Box::new(value),
                    });
                }
                Opcode::RpmFail => {
                    self.stack.push(Value::RpmState {
                        ok: false,
                        value: Box::new(Value::Unit),
                    });
                }
                Opcode::RpmCheck => {
                    let state = self.pop()?;
                    match state {
                        Value::RpmState { ok, .. } => {
                            self.stack.push(Value::Bool(ok));
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "rpm state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::RpmIsSuccess => {
                    let state = self.pop()?;
                    match state {
                        Value::RpmState { ok, .. } => {
                            self.stack.push(Value::Bool(ok));
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "rpm state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::RpmUnwrap => {
                    let state = self.pop()?;
                    match state {
                        Value::RpmState { ok, value } => {
                            if ok {
                                self.stack.push(*value);
                            } else {
                                return Err(VmError::RpmUnwrapFailed);
                            }
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "rpm state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::RpmState => {
                    self.stack.push(Value::Unit);
                }
                Opcode::RpmBind => {
                    let func = self.pop()?;
                    let state = self.pop()?;
                    match state {
                        Value::RpmState { ok, value } => {
                            if ok {
                                let entry = match func {
                                    Value::Closure { entry } => entry,
                                    other => {
                                        return Err(VmError::TypeMismatch {
                                            expected: "closure",
                                            found: other,
                                        })
                                    }
                                };
                                self.enter_call(entry, *value, CallFrameKind::RpmBind);
                            } else {
                                self.stack.push(Value::RpmState { ok, value });
                            }
                        }
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "rpm state",
                                found: other,
                            })
                        }
                    }
                }
                Opcode::InstallHandler => {
                    let count = self.pop_int()? as usize;
                    let mut op_clauses = Vec::with_capacity(count);
                    for _ in 0..count {
                        let clause = self.pop()?;
                        let op_id = self.pop_int()? as u64;
                        let entry = match clause {
                            Value::Closure { entry } => entry,
                            other => {
                                return Err(VmError::TypeMismatch {
                                    expected: "closure",
                                    found: other,
                                })
                            }
                        };
                        op_clauses.push((op_id, entry));
                    }
                    let return_value = self.pop()?;
                    let return_entry = match return_value {
                        Value::Closure { entry } => entry,
                        other => {
                            return Err(VmError::TypeMismatch {
                                expected: "closure",
                                found: other,
                            })
                        }
                    };
                    self.handlers.push(HandlerFrame {
                        return_entry,
                        op_clauses,
                    });
                }
                Opcode::RemoveHandler => {
                    if self.handlers.pop().is_none() {
                        return Err(VmError::HandlerNotImplemented);
                    }
                }
                Opcode::PerformEffect => {
                    let op_id = Self::expect_operand1(&instr)?;
                    let arg = self.pop()?;
                    let mut handler_index = None;
                    let mut entry = None;
                    for (idx, handler) in self.handlers.iter().enumerate().rev() {
                        if let Some((_, clause_entry)) =
                            handler.op_clauses.iter().find(|(id, _)| *id == op_id)
                        {
                            handler_index = Some(idx);
                            entry = Some(*clause_entry);
                            break;
                        }
                    }
                    let Some(handler_index) = handler_index else {
                        return Err(VmError::UnhandledEffect);
                    };
                    let Some(entry) = entry else {
                        return Err(VmError::UnhandledEffect);
                    };
                    let handler = self.handlers.remove(handler_index);
                    let continuation = Value::Continuation {
                        pc: self.pc,
                        stack: self.stack.clone(),
                        locals: self.locals.clone(),
                        frames: self.frames.clone(),
                        handlers: {
                            let mut handlers = self.handlers.clone();
                            handlers.insert(handler_index, handler.clone());
                            handlers
                        },
                    };
                    let locals = std::mem::take(&mut self.locals);
                    self.frames.push(CallFrame {
                        return_pc: self.pc,
                        locals,
                        kind: CallFrameKind::Normal,
                        handler: Some(handler),
                    });
                    self.locals = vec![arg, continuation];
                    self.pc = entry;
                    self.stack.clear();
                }
                Opcode::Resume => {
                    let cont = self.pop()?;
                    self.resume_continuation(cont, Value::Unit)?;
                }
                Opcode::ResumeWith => {
                    let value = self.pop()?;
                    let cont = self.pop()?;
                    self.resume_continuation(cont, value)?;
                }
                Opcode::HandlerReturn => {
                    let value = self.pop()?;
                    let handler = self.handlers.pop().ok_or(VmError::UnhandledEffect)?;
                    let locals = std::mem::take(&mut self.locals);
                    self.frames.push(CallFrame {
                        return_pc: self.pc,
                        locals,
                        kind: CallFrameKind::Normal,
                        handler: None,
                    });
                    self.locals = vec![value];
                    self.pc = handler.return_entry;
                    self.stack.clear();
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
                        if let Some(handler) = frame.handler {
                            self.handlers.push(handler);
                        }
                        let result = match frame.kind {
                            CallFrameKind::Normal => result,
                            CallFrameKind::RpmBind => Value::RpmState {
                                ok: true,
                                value: Box::new(result),
                            },
                        };
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

    fn call_extern(&self, id: u64, args: Vec<Value>) -> Result<Value, VmError> {
        let Some(func) = self.externs.get(&id) else {
            return Err(VmError::UnknownExtern(id));
        };
        (func)(args)
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

    fn enter_call(&mut self, entry: usize, arg: Value, kind: CallFrameKind) {
        let locals = std::mem::take(&mut self.locals);
        self.frames.push(CallFrame {
            return_pc: self.pc,
            locals,
            kind,
            handler: None,
        });
        self.locals = vec![arg];
        self.pc = entry;
    }

    fn resume_continuation(&mut self, cont: Value, value: Value) -> Result<(), VmError> {
        match cont {
            Value::Continuation {
                pc,
                mut stack,
                locals,
                frames,
                handlers,
            } => {
                stack.push(value);
                self.pc = pc;
                self.stack = stack;
                self.locals = locals;
                self.frames = frames;
                self.handlers = handlers;
                Ok(())
            }
            other => Err(VmError::TypeMismatch {
                expected: "continuation",
                found: other,
            }),
        }
    }
}
