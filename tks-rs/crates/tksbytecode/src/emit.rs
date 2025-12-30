use tkscore::ast::{Ident, Literal, World};
use tksir::ir::{IRComp, IRTerm, IRVal};

use crate::bytecode::{Instruction, Opcode};

#[derive(Debug, Clone)]
pub enum EmitError {
    Unimplemented(&'static str),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
        }
    }
}

struct EmitState {
    locals: Vec<(Ident, u64)>,
    next_local: u64,
    code: Vec<Instruction>,
}

impl EmitState {
    fn new() -> Self {
        Self {
            locals: Vec::new(),
            next_local: 0,
            code: Vec::new(),
        }
    }

    fn emit_term(&mut self, term: &IRTerm) -> Result<(), EmitError> {
        match term {
            IRTerm::Return(val) => {
                self.emit_val(val)?;
                self.code.push(inst(Opcode::Ret));
                Ok(())
            }
            IRTerm::Let(name, comp, body) => match comp.as_ref() {
                IRComp::Pure(val) => {
                    self.emit_val(val)?;
                    let slot = self.alloc_local(name.clone());
                    self.code.push(inst1(Opcode::Store, slot));
                    self.emit_term(body)?;
                    self.locals.pop();
                    Ok(())
                }
                _ => Err(EmitError::Unimplemented("let emission")),
            },
            _ => Err(EmitError::Unimplemented("term emission")),
        }
    }

    fn emit_val(&mut self, val: &IRVal) -> Result<(), EmitError> {
        match val {
            IRVal::Lit(literal) => self.emit_literal(literal),
            IRVal::Var(name) => {
                let slot = self
                    .lookup_local(name)
                    .ok_or(EmitError::Unimplemented("unknown local"))?;
                self.code.push(inst1(Opcode::Load, slot));
                Ok(())
            }
            IRVal::Element(world, index) => {
                let world_id = world_to_u64(*world);
                self.code
                    .push(inst2(Opcode::PushElement, world_id, u64::from(*index)));
                Ok(())
            }
            _ => Err(EmitError::Unimplemented("value emission")),
        }
    }

    fn emit_literal(&mut self, literal: &Literal) -> Result<(), EmitError> {
        match literal {
            Literal::Int(value) => {
                self.code.push(inst1(Opcode::PushInt, *value as u64));
                Ok(())
            }
            Literal::Bool(value) => {
                let encoded = if *value { 1 } else { 0 };
                self.code.push(inst1(Opcode::PushBool, encoded));
                Ok(())
            }
            Literal::Unit => {
                self.code.push(inst(Opcode::PushUnit));
                Ok(())
            }
            _ => Err(EmitError::Unimplemented("literal emission")),
        }
    }

    fn alloc_local(&mut self, name: Ident) -> u64 {
        let slot = self.next_local;
        self.next_local += 1;
        self.locals.push((name, slot));
        slot
    }

    fn lookup_local(&self, name: &str) -> Option<u64> {
        self.locals.iter().rev().find_map(|(local, slot)| {
            if local == name {
                Some(*slot)
            } else {
                None
            }
        })
    }
}

fn inst(opcode: Opcode) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: None,
        operand2: None,
    }
}

fn inst1(opcode: Opcode, operand1: u64) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: Some(operand1),
        operand2: None,
    }
}

fn inst2(opcode: Opcode, operand1: u64, operand2: u64) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: Some(operand1),
        operand2: Some(operand2),
    }
}

fn world_to_u64(world: World) -> u64 {
    match world {
        World::A => 0,
        World::B => 1,
        World::C => 2,
        World::D => 3,
    }
}

pub fn emit(term: &IRTerm) -> Result<Vec<Instruction>, EmitError> {
    let mut state = EmitState::new();
    state.emit_term(term)?;
    Ok(state.code)
}
