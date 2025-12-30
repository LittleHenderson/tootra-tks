use tkscore::ast::{Expr, Ident, Literal, OrdinalLiteral, World};
use tksir::ir::{IRComp, IRTerm, IRVal, OrdOp};

use crate::bytecode::{Instruction, Opcode};

#[derive(Debug, Clone)]
pub enum EmitError {
    Unimplemented(&'static str),
    InvalidJumpTarget(usize),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
            EmitError::InvalidJumpTarget(index) => {
                write!(f, "invalid jump target at instruction {index}")
            }
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
                Ok(())
            }
            IRTerm::App(func, arg) => {
                self.emit_val(func)?;
                self.emit_val(arg)?;
                self.code.push(inst(Opcode::Call));
                Ok(())
            }
            IRTerm::Let(name, comp, body) => {
                self.emit_comp(comp)?;
                let slot = self.alloc_local(name.clone());
                self.code.push(inst1(Opcode::Store, slot));
                self.emit_term(body)?;
                self.locals.pop();
                Ok(())
            }
            IRTerm::If(cond, then_term, else_term) => {
                self.emit_val(cond)?;
                let jmp_to_else = self.emit_jump_placeholder(Opcode::JmpUnless);
                self.emit_term(then_term)?;
                let jmp_to_end = self.emit_jump_placeholder(Opcode::Jmp);
                let else_target = self.code.len();
                self.patch_jump(jmp_to_else, else_target)?;
                self.emit_term(else_term)?;
                let end_target = self.code.len();
                self.patch_jump(jmp_to_end, end_target)?;
                Ok(())
            }
            IRTerm::OrdSucc(val) => {
                self.emit_val(val)?;
                self.code.push(inst(Opcode::OrdSucc));
                Ok(())
            }
            IRTerm::OrdOp(op, left, right) => {
                self.emit_val(left)?;
                self.emit_val(right)?;
                let opcode = match op {
                    OrdOp::Add => Opcode::OrdAdd,
                    OrdOp::Mul => Opcode::OrdMul,
                    OrdOp::Exp => Opcode::OrdExp,
                };
                self.code.push(inst(opcode));
                Ok(())
            }
            _ => Err(EmitError::Unimplemented("term emission")),
        }
    }

    fn emit_comp(&mut self, comp: &IRComp) -> Result<(), EmitError> {
        match comp {
            IRComp::Pure(val) => self.emit_val(val),
            IRComp::Effect(term) => self.emit_term(term),
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
            IRVal::Lam(param, body) => {
                let body_code = self.emit_lambda_body(param, body)?;
                let entry = self.code.len() + 2;
                let target = entry + body_code.len();
                self.code
                    .push(inst1(Opcode::PushClosure, entry as u64));
                self.code.push(inst1(Opcode::Jmp, target as u64));
                self.code.extend(body_code);
                Ok(())
            }
            IRVal::Noetic(index) => {
                self.code
                    .push(inst1(Opcode::PushNoetic, u64::from(*index)));
                Ok(())
            }
            IRVal::Ket(inner) => {
                self.emit_val(inner)?;
                self.code.push(inst(Opcode::MakeKet));
                Ok(())
            }
            IRVal::Ordinal(expr) => self.emit_ordinal_expr(expr),
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

    fn emit_ordinal_expr(&mut self, expr: &Expr) -> Result<(), EmitError> {
        match expr {
            Expr::OrdLit { value, .. } => self.emit_ordinal_literal(value),
            Expr::OrdOmega { .. } => {
                self.code.push(inst(Opcode::PushOmega));
                Ok(())
            }
            Expr::OrdEpsilon { index, .. } => {
                self.code.push(inst1(Opcode::PushEpsilon, *index));
                Ok(())
            }
            Expr::OrdAleph { .. } => Err(EmitError::Unimplemented("ordinal aleph emission")),
            _ => Err(EmitError::Unimplemented("ordinal literal emission")),
        }
    }

    fn emit_ordinal_literal(&mut self, literal: &OrdinalLiteral) -> Result<(), EmitError> {
        match literal {
            OrdinalLiteral::Finite(value) => {
                self.code.push(inst1(Opcode::PushOrd, *value));
                Ok(())
            }
            OrdinalLiteral::Omega => {
                self.code.push(inst(Opcode::PushOmega));
                Ok(())
            }
            OrdinalLiteral::Epsilon(index) => {
                self.code.push(inst1(Opcode::PushEpsilon, *index));
                Ok(())
            }
            OrdinalLiteral::OmegaPlus(value) => {
                self.code.push(inst(Opcode::PushOmega));
                self.code.push(inst1(Opcode::PushOrd, *value));
                self.code.push(inst(Opcode::OrdAdd));
                Ok(())
            }
            OrdinalLiteral::OmegaTimes(value) => {
                self.code.push(inst(Opcode::PushOmega));
                self.code.push(inst1(Opcode::PushOrd, *value));
                self.code.push(inst(Opcode::OrdMul));
                Ok(())
            }
            OrdinalLiteral::OmegaPow(value) => {
                self.code.push(inst(Opcode::PushOmega));
                self.code.push(inst1(Opcode::PushOrd, *value));
                self.code.push(inst(Opcode::OrdExp));
                Ok(())
            }
            OrdinalLiteral::Aleph(_) => Err(EmitError::Unimplemented("ordinal aleph emission")),
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

    fn emit_jump_placeholder(&mut self, opcode: Opcode) -> usize {
        let index = self.code.len();
        self.code.push(inst1(opcode, 0));
        index
    }

    fn patch_jump(&mut self, index: usize, target: usize) -> Result<(), EmitError> {
        let Some(instr) = self.code.get_mut(index) else {
            return Err(EmitError::InvalidJumpTarget(index));
        };
        instr.operand1 = Some(target as u64);
        Ok(())
    }

    fn emit_lambda_body(&self, param: &Ident, body: &IRTerm) -> Result<Vec<Instruction>, EmitError> {
        let mut lambda_state = EmitState::new();
        lambda_state.alloc_local(param.clone());
        lambda_state.emit_term(body)?;
        lambda_state.code.push(inst(Opcode::Ret));
        Ok(lambda_state.code)
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
    state.code.push(inst(Opcode::Ret));
    Ok(state.code)
}
