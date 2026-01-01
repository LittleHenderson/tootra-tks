use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use tkscore::ast::{Expr, Ident, Literal, OrdinalLiteral, World};
use tksir::ir::{IntOp, IRComp, IRTerm, IRVal, OrdOp};

use crate::bytecode::{extern_id, field_id, string_id, Instruction, Opcode};

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
    op_table: Rc<RefCell<OpTable>>,
}

impl EmitState {
    fn new() -> Self {
        Self {
            locals: Vec::new(),
            next_local: 0,
            code: Vec::new(),
            op_table: Rc::new(RefCell::new(OpTable::new())),
        }
    }

    fn with_shared_ops(op_table: Rc<RefCell<OpTable>>) -> Self {
        Self {
            locals: Vec::new(),
            next_local: 0,
            code: Vec::new(),
            op_table,
        }
    }

    fn emit_term(&mut self, term: &IRTerm) -> Result<(), EmitError> {
        match term {
            IRTerm::Return(val) => {
                self.emit_val(val)?;
                Ok(())
            }
            IRTerm::App(func, arg) => {
                if let IRVal::Noetic(index) = func {
                    self.code.push(inst1(Opcode::PushNoetic, u64::from(*index)));
                    self.emit_val(arg)?;
                    self.code.push(inst(Opcode::ApplyNoetic));
                } else {
                    self.emit_val(func)?;
                    self.emit_val(arg)?;
                    self.code.push(inst(Opcode::Call));
                }
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
            IRTerm::RecordGet(record, field) => {
                self.emit_val(record)?;
                let id = field_id(field);
                self.code.push(inst1(Opcode::RecordGet, id));
                Ok(())
            }
            IRTerm::RecordSet(record, field, value) => {
                self.emit_val(record)?;
                self.emit_val(value)?;
                let id = field_id(field);
                self.code.push(inst1(Opcode::RecordSet, id));
                Ok(())
            }
            IRTerm::Superpose(states) => {
                for (amp, ket) in states {
                    self.emit_val(amp)?;
                    self.emit_val(ket)?;
                }
                self.code
                    .push(inst1(Opcode::Superpose, states.len() as u64));
                Ok(())
            }
            IRTerm::Measure(val) => {
                self.emit_val(val)?;
                self.code.push(inst(Opcode::Measure));
                Ok(())
            }
            IRTerm::Entangle(left, right) => {
                self.emit_val(left)?;
                self.emit_val(right)?;
                self.code.push(inst(Opcode::Entangle));
                Ok(())
            }
            IRTerm::RPMCheck(val) => {
                self.emit_val(val)?;
                self.code.push(inst(Opcode::RpmCheck));
                Ok(())
            }
            IRTerm::RPMAcquire(val) => {
                self.emit_val(val)?;
                self.code.push(inst(Opcode::RpmAcquire));
                Ok(())
            }
            IRTerm::RPMFail => {
                self.code.push(inst(Opcode::RpmFail));
                Ok(())
            }
            IRTerm::Perform(op, arg) => {
                self.emit_val(arg)?;
                let op_id = self.op_id(op);
                self.code.push(inst1(Opcode::PerformEffect, op_id));
                Ok(())
            }
            IRTerm::Handle(expr, handler) => {
                let (ret_name, ret_term) = &handler.return_clause;
                self.emit_closure(&[ret_name.clone()], ret_term)?;
                for clause in &handler.op_clauses {
                    let op_id = self.op_id(&clause.op);
                    self.code.push(inst1(Opcode::PushInt, op_id));
                    self.emit_closure(
                        &[clause.arg.clone(), clause.k.clone()],
                        &clause.body,
                    )?;
                }
                self.code
                    .push(inst1(Opcode::PushInt, handler.op_clauses.len() as u64));
                let install_index = self.code.len();
                self.code.push(inst1(Opcode::InstallHandler, 0));
                self.emit_term(expr)?;
                self.code.push(inst(Opcode::HandlerReturn));
                let resume_pc = self.code.len();
                self.patch_jump(install_index, resume_pc)?;
                Ok(())
            }
            IRTerm::RpmReturn(val) => {
                self.emit_val(val)?;
                self.code.push(inst(Opcode::RpmReturn));
                Ok(())
            }
            IRTerm::RpmBind(left, right) => {
                self.emit_val(left)?;
                self.emit_val(right)?;
                self.code.push(inst(Opcode::RpmBind));
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
            IRTerm::IntOp(op, left, right) => {
                self.emit_val(left)?;
                self.emit_val(right)?;
                let opcode = match op {
                    IntOp::Add => Opcode::Add,
                    IntOp::Sub => Opcode::Sub,
                    IntOp::Mul => Opcode::Mul,
                    IntOp::Div => Opcode::Div,
                };
                self.code.push(inst(opcode));
                Ok(())
            }
            IRTerm::ArrayGet(array, index) => {
                self.emit_val(array)?;
                self.emit_val(index)?;
                self.code.push(inst(Opcode::ArrayGet));
                Ok(())
            }
            IRTerm::ArraySet(array, index, value) => {
                self.emit_val(array)?;
                self.emit_val(index)?;
                self.emit_val(value)?;
                self.code.push(inst(Opcode::ArraySet));
                Ok(())
            }
            IRTerm::ForIn { binder, collection, body } => {
                // Get array length
                self.emit_val(collection)?;
                self.code.push(inst(Opcode::ArrayLen));
                let len_slot = self.alloc_local("_len".to_string());
                self.code.push(inst1(Opcode::Store, len_slot));
                // Initialize index
                self.code.push(inst1(Opcode::PushInt, 0));
                let idx_slot = self.alloc_local("_idx".to_string());
                self.code.push(inst1(Opcode::Store, idx_slot));
                // Loop start
                let loop_start = self.code.len();
                // Check if index < len
                self.code.push(inst1(Opcode::Load, idx_slot));
                self.code.push(inst1(Opcode::Load, len_slot));
                self.code.push(inst(Opcode::Lt));
                let exit_jmp = self.emit_jump_placeholder(Opcode::JmpUnless);
                // Load element into binder
                self.emit_val(collection)?;
                self.code.push(inst1(Opcode::Load, idx_slot));
                self.code.push(inst(Opcode::ArrayGet));
                let binder_slot = self.alloc_local(binder.clone());
                self.code.push(inst1(Opcode::Store, binder_slot));
                // Execute body
                self.emit_term(body)?;
                self.locals.pop(); // Remove binder
                // Increment index
                self.code.push(inst1(Opcode::Load, idx_slot));
                self.code.push(inst1(Opcode::PushInt, 1));
                self.code.push(inst(Opcode::Add));
                self.code.push(inst1(Opcode::Store, idx_slot));
                // Jump back to loop start
                self.code.push(inst1(Opcode::Jmp, loop_start as u64));
                // Exit point
                let exit_target = self.code.len();
                self.patch_jump(exit_jmp, exit_target)?;
                self.locals.pop(); // Remove _idx
                self.locals.pop(); // Remove _len
                self.code.push(inst(Opcode::PushUnit)); // for-in returns unit
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
                if let Some(slot) = self.lookup_local(name) {
                    self.code.push(inst1(Opcode::Load, slot));
                } else {
                    let id = extern_id(name);
                    self.code.push(inst1(Opcode::LoadGlobal, id));
                }
                Ok(())
            }
            IRVal::Element(world, index) => {
                let world_id = world_to_u64(*world);
                self.code
                    .push(inst2(Opcode::PushElement, world_id, u64::from(*index)));
                Ok(())
            }
            IRVal::Lam(param, body) => {
                self.emit_closure(&[param.clone()], body)
            }
            IRVal::Extern(name, arity) => {
                let id = extern_id(name);
                self.code
                    .push(inst2(Opcode::PushExtern, id, *arity as u64));
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
            IRVal::Record(fields) => {
                self.code.push(inst(Opcode::MakeRecord));
                for (name, value) in fields {
                    self.emit_val(value)?;
                    let id = field_id(name);
                    self.code.push(inst1(Opcode::RecordSet, id));
                }
                Ok(())
            }
            IRVal::Str(s) => {
                let id = string_id(s);
                self.code.push(inst1(Opcode::PushStr, id));
                Ok(())
            }
            IRVal::Array(elements) => {
                for elem in elements {
                    self.emit_val(elem)?;
                }
                self.code.push(inst1(Opcode::MakeArray, elements.len() as u64));
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
            Literal::Str(s) => {
                let id = string_id(s);
                self.code.push(inst1(Opcode::PushStr, id));
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

    fn emit_closure(&mut self, params: &[Ident], body: &IRTerm) -> Result<(), EmitError> {
        let body_code = self.emit_closure_body(params, body)?;
        let entry = self.code.len() + 2;
        let target = entry + body_code.len();
        self.code
            .push(inst1(Opcode::PushClosure, entry as u64));
        self.code.push(inst1(Opcode::Jmp, target as u64));
        self.code.extend(body_code);
        Ok(())
    }

    fn emit_closure_body(&self, params: &[Ident], body: &IRTerm) -> Result<Vec<Instruction>, EmitError> {
        let mut state = EmitState::with_shared_ops(Rc::clone(&self.op_table));
        for param in params {
            state.alloc_local(param.clone());
        }
        state.emit_term(body)?;
        state.code.push(inst(Opcode::Ret));
        Ok(state.code)
    }

    fn op_id(&self, op: &str) -> u64 {
        let mut table = self.op_table.borrow_mut();
        table.id_for(op)
    }
}

#[derive(Debug, Default)]
struct OpTable {
    next_id: u64,
    ids: HashMap<String, u64>,
}

impl OpTable {
    fn new() -> Self {
        Self::default()
    }

    fn id_for(&mut self, name: &str) -> u64 {
        if let Some(id) = self.ids.get(name) {
            return *id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.ids.insert(name.to_string(), id);
        id
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
