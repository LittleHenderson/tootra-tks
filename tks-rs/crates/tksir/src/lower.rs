use tkscore::ast::{Expr, HandlerDef, Literal, Program, TopDecl};

use crate::ir::{IRComp, IRHandler, IRHandlerClause, IRTerm, IRVal};

#[derive(Debug, Clone)]
pub enum LowerError {
    Unimplemented(&'static str),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
        }
    }
}

#[derive(Default)]
struct LowerState {
    next_tmp: usize,
}

impl LowerState {
    fn fresh_temp(&mut self) -> String {
        let name = format!("_t{}", self.next_tmp);
        self.next_tmp += 1;
        name
    }
}

pub fn lower_program(program: &Program) -> Result<IRTerm, LowerError> {
    let mut state = LowerState::default();
    let mut term = match &program.entry {
        Some(expr) => lower_term(&mut state, expr)?,
        None => IRTerm::Return(IRVal::Lit(Literal::Unit)),
    };

    for decl in program.decls.iter().rev() {
        if let TopDecl::LetDecl { name, value, .. } = decl {
            let comp = lower_comp(&mut state, value)?;
            term = IRTerm::Let(name.clone(), Box::new(comp), Box::new(term));
        }
    }

    Ok(term)
}

fn lower_comp(state: &mut LowerState, expr: &Expr) -> Result<IRComp, LowerError> {
    if is_value_expr(expr) {
        Ok(IRComp::Pure(lower_val(state, expr)?))
    } else {
        Ok(IRComp::Effect(Box::new(lower_term(state, expr)?)))
    }
}

fn lower_to_val(
    state: &mut LowerState,
    expr: &Expr,
) -> Result<(Vec<(String, IRComp)>, IRVal), LowerError> {
    match lower_comp(state, expr)? {
        IRComp::Pure(val) => Ok((Vec::new(), val)),
        comp => {
            let tmp = state.fresh_temp();
            Ok((vec![(tmp.clone(), comp)], IRVal::Var(tmp)))
        }
    }
}

fn wrap_lets(mut bindings: Vec<(String, IRComp)>, mut body: IRTerm) -> IRTerm {
    while let Some((name, comp)) = bindings.pop() {
        body = IRTerm::Let(name, Box::new(comp), Box::new(body));
    }
    body
}

fn is_value_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Var { .. }
            | Expr::Lit { .. }
            | Expr::Lam { .. }
            | Expr::Element { .. }
            | Expr::OrdLit { .. }
            | Expr::OrdOmega { .. }
            | Expr::OrdEpsilon { .. }
            | Expr::OrdAleph { .. }
    )
}

fn lower_val(state: &mut LowerState, expr: &Expr) -> Result<IRVal, LowerError> {
    match expr {
        Expr::Var { name, .. } => Ok(IRVal::Var(name.clone())),
        Expr::Lit { literal, .. } => Ok(IRVal::Lit(literal.clone())),
        Expr::Lam { param, body, .. } => {
            let body_term = lower_term(state, body)?;
            Ok(IRVal::Lam(param.clone(), Box::new(body_term)))
        }
        Expr::Element { world, index, .. } => Ok(IRVal::Element(*world, *index)),
        Expr::OrdLit { .. }
        | Expr::OrdOmega { .. }
        | Expr::OrdEpsilon { .. }
        | Expr::OrdAleph { .. } => Ok(IRVal::Ordinal(expr.clone())),
        _ => Err(LowerError::Unimplemented("value lowering")),
    }
}

fn lower_term(state: &mut LowerState, expr: &Expr) -> Result<IRTerm, LowerError> {
    match expr {
        Expr::Var { .. }
        | Expr::Lit { .. }
        | Expr::Lam { .. }
        | Expr::Element { .. }
        | Expr::OrdLit { .. }
        | Expr::OrdOmega { .. }
        | Expr::OrdEpsilon { .. }
        | Expr::OrdAleph { .. } => Ok(IRTerm::Return(lower_val(state, expr)?)),
        Expr::App { func, arg, .. } => {
            let (mut bindings, func_val) = lower_to_val(state, func)?;
            let (arg_bindings, arg_val) = lower_to_val(state, arg)?;
            bindings.extend(arg_bindings);
            let term = IRTerm::App(func_val, arg_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::Let { name, value, body, .. } => {
            let comp = lower_comp(state, value)?;
            let body_term = lower_term(state, body)?;
            Ok(IRTerm::Let(
                name.clone(),
                Box::new(comp),
                Box::new(body_term),
            ))
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let (bindings, cond_val) = lower_to_val(state, cond)?;
            let then_term = lower_term(state, then_branch)?;
            let else_term = lower_term(state, else_branch)?;
            let term = IRTerm::If(cond_val, Box::new(then_term), Box::new(else_term));
            Ok(wrap_lets(bindings, term))
        }
        Expr::Noetic { index, expr, .. } => {
            let (bindings, arg_val) = lower_to_val(state, expr)?;
            let term = IRTerm::App(IRVal::Noetic(*index), arg_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::Perform { op, arg, .. } => {
            let (bindings, arg_val) = lower_to_val(state, arg)?;
            let term = IRTerm::Perform(op.clone(), arg_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::Handle { expr, handler, .. } => {
            let expr_term = lower_term(state, expr)?;
            let handler = match handler {
                tkscore::ast::HandlerRef::Inline { def, .. } => {
                    lower_handler_def(state, def)?
                }
                tkscore::ast::HandlerRef::Named { .. } => {
                    return Err(LowerError::Unimplemented("named handler lowering"));
                }
            };
            Ok(IRTerm::Handle(Box::new(expr_term), Box::new(handler)))
        }
        Expr::RPMCheck { expr, .. } => {
            let (bindings, arg_val) = lower_to_val(state, expr)?;
            let term = IRTerm::RPMCheck(arg_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::RPMAcquire { expr, .. } => {
            let (bindings, arg_val) = lower_to_val(state, expr)?;
            let term = IRTerm::RPMAcquire(arg_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::OrdSucc { expr, .. } => {
            let (bindings, val) = lower_to_val(state, expr)?;
            let term = IRTerm::OrdSucc(val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::OrdAdd { left, right, .. } => {
            let (mut bindings, left_val) = lower_to_val(state, left)?;
            let (right_bindings, right_val) = lower_to_val(state, right)?;
            bindings.extend(right_bindings);
            let term = IRTerm::OrdOp(crate::ir::OrdOp::Add, left_val, right_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::OrdMul { left, right, .. } => {
            let (mut bindings, left_val) = lower_to_val(state, left)?;
            let (right_bindings, right_val) = lower_to_val(state, right)?;
            bindings.extend(right_bindings);
            let term = IRTerm::OrdOp(crate::ir::OrdOp::Mul, left_val, right_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::OrdExp { left, right, .. } => {
            let (mut bindings, left_val) = lower_to_val(state, left)?;
            let (right_bindings, right_val) = lower_to_val(state, right)?;
            bindings.extend(right_bindings);
            let term = IRTerm::OrdOp(crate::ir::OrdOp::Exp, left_val, right_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::OrdLimit {
            binder, bound, body, ..
        } => {
            let (bindings, bound_val) = lower_to_val(state, bound)?;
            let body_term = lower_term(state, body)?;
            let term = IRTerm::OrdLimit(binder.clone(), bound_val, Box::new(body_term));
            Ok(wrap_lets(bindings, term))
        }
        _ => Err(LowerError::Unimplemented("expression lowering")),
    }
}

fn lower_handler_def(state: &mut LowerState, def: &HandlerDef) -> Result<IRHandler, LowerError> {
    let (ret_name, ret_expr) = &def.return_clause;
    let ret_term = lower_term(state, ret_expr)?;
    let mut clauses = Vec::new();
    for clause in &def.op_clauses {
        let body_term = lower_term(state, &clause.body)?;
        clauses.push(IRHandlerClause {
            op: clause.op.clone(),
            arg: clause.arg.clone(),
            k: clause.k.clone(),
            body: Box::new(body_term),
        });
    }
    Ok(IRHandler {
        return_clause: (ret_name.clone(), Box::new(ret_term)),
        op_clauses: clauses,
    })
}
