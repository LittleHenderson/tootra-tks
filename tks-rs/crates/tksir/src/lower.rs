use std::collections::HashMap;
use tkscore::ast::{ClassDecl, Expr, HandlerDef, Literal, Program, TopDecl};
use tkscore::span::Span;

use crate::ir::{IntOp, IRComp, IRHandler, IRHandlerClause, IRTerm, IRVal};

#[derive(Debug, Clone)]
pub enum LowerError {
    Unimplemented(&'static str),
    UnknownClass(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
            LowerError::UnknownClass(name) => write!(f, "unknown class: {name}"),
        }
    }
}

#[derive(Default)]
struct LowerState {
    next_tmp: usize,
    current_class: Option<String>,
    class_map: HashMap<String, ClassDecl>,
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

    // Pre-scan to populate class_map
    for decl in &program.decls {
        if let TopDecl::ClassDecl(class_decl) = decl {
            state
                .class_map
                .insert(class_decl.name.clone(), class_decl.clone());
        }
    }

    let mut term = match &program.entry {
        Some(expr) => lower_term(&mut state, expr)?,
        None => IRTerm::Return(IRVal::Lit(Literal::Unit)),
    };

    for decl in program.decls.iter().rev() {
        match decl {
            TopDecl::LetDecl { name, value, .. } => {
                let comp = lower_comp(&mut state, value)?;
                term = IRTerm::Let(name.clone(), Box::new(comp), Box::new(term));
            }
            TopDecl::ExternDecl(decl) => {
                let arity = decl.params.len();
                let comp = IRComp::Pure(IRVal::Extern(decl.name.clone(), arity));
                term = IRTerm::Let(decl.name.clone(), Box::new(comp), Box::new(term));
            }
            TopDecl::ClassDecl(decl) => {
                term = lower_class_decl(&mut state, decl, term)?;
            }
            _ => {}
        }
    }

    Ok(term)
}

fn lower_class_decl(
    state: &mut LowerState,
    decl: &ClassDecl,
    mut term: IRTerm,
) -> Result<IRTerm, LowerError> {
    let previous_class = state.current_class.clone();
    state.current_class = Some(decl.name.clone());
    for method in decl.actions.iter().rev() {
        let name = format!("{}::{}", decl.name, method.name);
        let expr = build_member_lambda(
            method.body.clone(),
            &method.self_param,
            method_param_names(method),
        );
        let val = lower_val(state, &expr)?;
        term = IRTerm::Let(name, Box::new(IRComp::Pure(val)), Box::new(term));
    }
    for prop in decl.details.iter().rev() {
        let name = format!("{}::{}", decl.name, prop.name);
        let expr = build_member_lambda(prop.value.clone(), "self", vec!["self".to_string()]);
        let val = lower_val(state, &expr)?;
        term = IRTerm::Let(name, Box::new(IRComp::Pure(val)), Box::new(term));
    }
    let ctor_expr = build_constructor_lambda(constructor_param_names(decl));
    let ctor_val = lower_val(state, &ctor_expr)?;
    term = IRTerm::Let(
        decl.name.clone(),
        Box::new(IRComp::Pure(ctor_val)),
        Box::new(term),
    );
    state.current_class = previous_class;
    Ok(term)
}

fn build_member_lambda(body: Expr, self_param: &str, params: Vec<String>) -> Expr {
    let mut expr = wrap_self_alias(body, self_param);
    for param in params.into_iter().rev() {
        expr = Expr::Lam {
            span: dummy_span(),
            param,
            param_type: None,
            body: Box::new(expr),
        };
    }
    expr
}

fn wrap_self_alias(body: Expr, self_param: &str) -> Expr {
    let (alias, target) = if self_param == "identity" {
        ("self", "identity")
    } else {
        ("identity", "self")
    };
    Expr::Let {
        span: dummy_span(),
        name: alias.to_string(),
        scheme: None,
        value: Box::new(Expr::Var {
            span: dummy_span(),
            name: target.to_string(),
        }),
        body: Box::new(body),
    }
}

fn method_param_names(method: &tkscore::ast::MethodDecl) -> Vec<String> {
    let mut params = Vec::with_capacity(method.params.len() + 1);
    params.push(method.self_param.clone());
    for param in &method.params {
        params.push(param.name.clone());
    }
    params
}

fn constructor_param_names(decl: &ClassDecl) -> Vec<String> {
    decl.specifics
        .iter()
        .map(|field| field.name.clone())
        .collect()
}

fn build_constructor_lambda(params: Vec<String>) -> Expr {
    let mut expr = Expr::Lit {
        span: dummy_span(),
        literal: Literal::Unit,
    };
    for param in params.into_iter().rev() {
        expr = Expr::Lam {
            span: dummy_span(),
            param,
            param_type: None,
            body: Box::new(expr),
        };
    }
    expr
}

fn dummy_span() -> Span {
    Span::new(0, 0, 0, 0)
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
    if let Expr::Ket { expr: inner, .. } = expr {
        let (bindings, inner_val) = lower_to_val(state, inner.as_ref())?;
        return Ok((bindings, IRVal::Ket(Box::new(inner_val))));
    }
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
            | Expr::ArrayLit { .. }
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
        Expr::ArrayLit { elements, .. } => {
            let mut ir_elements = Vec::new();
            for elem in elements {
                ir_elements.push(lower_val(state, elem)?);
            }
            Ok(IRVal::Array(ir_elements))
        }
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
        Expr::Member { object, field, .. } => lower_member_term(state, object, field),
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
        Expr::RPMReturn { expr, .. } => {
            let (bindings, val) = lower_to_val(state, expr)?;
            let term = IRTerm::RpmReturn(val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::RPMBind { left, right, .. } => {
            let (mut bindings, left_val) = lower_to_val(state, left)?;
            let (right_bindings, right_val) = lower_to_val(state, right)?;
            bindings.extend(right_bindings);
            let term = IRTerm::RpmBind(left_val, right_val);
            Ok(wrap_lets(bindings, term))
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
        Expr::Constructor { name, fields, .. } => lower_constructor_term(state, name, fields),
        Expr::Superpose { states, .. } => {
            let mut bindings = Vec::new();
            let mut lowered_states = Vec::new();
            for (amp, ket) in states {
                let (amp_bindings, amp_val) = lower_to_val(state, amp)?;
                bindings.extend(amp_bindings);
                let (ket_bindings, ket_val) = lower_to_val(state, ket)?;
                bindings.extend(ket_bindings);
                lowered_states.push((amp_val, ket_val));
            }
            let term = IRTerm::Superpose(lowered_states);
            Ok(wrap_lets(bindings, term))
        }
        Expr::Measure { expr, .. } => {
            let (bindings, arg_val) = lower_to_val(state, expr)?;
            let term = IRTerm::Measure(arg_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::Entangle { left, right, .. } => {
            let (mut bindings, left_val) = lower_to_val(state, left)?;
            let (right_bindings, right_val) = lower_to_val(state, right)?;
            bindings.extend(right_bindings);
            let term = IRTerm::Entangle(left_val, right_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::Ket { expr, .. } => {
            let (bindings, inner_val) = lower_to_val(state, expr)?;
            let term = IRTerm::Return(IRVal::Ket(Box::new(inner_val)));
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
        Expr::BinOp { op, left, right, .. } => {
            let (mut bindings, left_val) = lower_to_val(state, left)?;
            let (right_bindings, right_val) = lower_to_val(state, right)?;
            bindings.extend(right_bindings);
            let ir_op = match op {
                tkscore::ast::BinOp::Add => IntOp::Add,
                tkscore::ast::BinOp::Sub => IntOp::Sub,
                tkscore::ast::BinOp::Mul => IntOp::Mul,
                tkscore::ast::BinOp::Div => IntOp::Div,
            };
            let term = IRTerm::IntOp(ir_op, left_val, right_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::ArrayLit { elements, .. } => {
            let mut bindings = Vec::new();
            let mut ir_elements = Vec::new();
            for elem in elements {
                let (elem_bindings, elem_val) = lower_to_val(state, elem)?;
                bindings.extend(elem_bindings);
                ir_elements.push(elem_val);
            }
            let term = IRTerm::Return(IRVal::Array(ir_elements));
            Ok(wrap_lets(bindings, term))
        }
        Expr::Index { array, index, .. } => {
            let (mut bindings, array_val) = lower_to_val(state, array)?;
            let (index_bindings, index_val) = lower_to_val(state, index)?;
            bindings.extend(index_bindings);
            let term = IRTerm::ArrayGet(array_val, index_val);
            Ok(wrap_lets(bindings, term))
        }
        Expr::ForIn { binder, collection, body, .. } => {
            let (bindings, coll_val) = lower_to_val(state, collection)?;
            let body_term = lower_term(state, body)?;
            let term = IRTerm::ForIn {
                binder: binder.clone(),
                collection: coll_val,
                body: Box::new(body_term),
            };
            Ok(wrap_lets(bindings, term))
        }
        _ => Err(LowerError::Unimplemented("expression lowering")),
    }
}

/// Lower member access `object.field` to RecordGet.
/// For any object expression, we lower the object to a value and emit RecordGet.
fn lower_member_term(
    state: &mut LowerState,
    object: &Expr,
    field: &str,
) -> Result<IRTerm, LowerError> {
    // First, check if we are accessing self/identity inside a class context
    // and the field refers to a class member (property or method)
    if let Some(class_name) = &state.current_class {
        let is_self = matches!(
            object,
            Expr::Var { name, .. } if name == "self" || name == "identity"
        );
        if is_self {
            // Check if this is a known class member that needs lookup via Class::field
            let member_name = format!("{}::{}", class_name, field);
            let expr = Expr::App {
                span: dummy_span(),
                func: Box::new(Expr::Var {
                    span: dummy_span(),
                    name: member_name,
                }),
                arg: Box::new(object.clone()),
            };
            return lower_term(state, &expr);
        }
    }

    // For general member access, lower to RecordGet
    let (bindings, obj_val) = lower_to_val(state, object)?;
    let term = IRTerm::RecordGet(obj_val, field.to_string());
    Ok(wrap_lets(bindings, term))
}

/// Lower a constructor call `repeat ClassName { field: value, ... }` to create a record.
/// The resulting record contains:
/// 1. All field values from the constructor call
/// 2. Computed properties (by calling Class::property on self)
/// 3. Method closures (partial application of Class::method with self)
fn lower_constructor_term(
    state: &mut LowerState,
    name: &str,
    fields: &[(String, Expr)],
) -> Result<IRTerm, LowerError> {
    // Look up the class declaration
    let class_decl = match state.class_map.get(name) {
        Some(decl) => decl.clone(),
        None => {
            // Fall back to old behavior for unknown classes
            let mut expr = Expr::Var {
                span: dummy_span(),
                name: name.to_string(),
            };
            for (_, field_value) in fields {
                expr = Expr::App {
                    span: dummy_span(),
                    func: Box::new(expr),
                    arg: Box::new(field_value.clone()),
                };
            }
            return lower_term(state, &expr);
        }
    };

    // Step 1: Lower all field values and collect bindings
    let mut bindings = Vec::new();
    let mut field_vals = Vec::new();
    for (field_name, field_expr) in fields {
        let (field_bindings, field_val) = lower_to_val(state, field_expr)?;
        bindings.extend(field_bindings);
        field_vals.push((field_name.clone(), field_val));
    }

    // Step 2: Create initial record with field values
    let record_val = IRVal::Record(field_vals);
    let self_tmp = state.fresh_temp();
    bindings.push((self_tmp.clone(), IRComp::Pure(record_val)));

    // Step 3: Add computed properties
    // For each property, call Class::property(self) and add to record
    let mut current_record = self_tmp.clone();
    for prop in &class_decl.details {
        let prop_func_name = format!("{}::{}", name, prop.name);
        // Call the property function with current record
        let prop_call = IRTerm::App(
            IRVal::Var(prop_func_name),
            IRVal::Var(current_record.clone()),
        );
        let prop_tmp = state.fresh_temp();
        bindings.push((prop_tmp.clone(), IRComp::Effect(Box::new(prop_call))));

        // RecordSet to add property to record
        let new_record = IRTerm::RecordSet(
            IRVal::Var(current_record.clone()),
            prop.name.clone(),
            IRVal::Var(prop_tmp),
        );
        let new_record_tmp = state.fresh_temp();
        bindings.push((new_record_tmp.clone(), IRComp::Effect(Box::new(new_record))));
        current_record = new_record_tmp;
    }

    // Step 4: Add method closures
    // For each method, create a closure that captures self and add to record
    for method in &class_decl.actions {
        let method_func_name = format!("{}::{}", name, method.name);
        // Create a closure that partially applies self to the method
        let method_closure = build_method_closure(
            &method_func_name,
            &current_record,
            method.params.len(),
            state,
        );
        let method_tmp = state.fresh_temp();
        bindings.push((method_tmp.clone(), IRComp::Pure(method_closure)));

        // RecordSet to add method closure to record
        let new_record = IRTerm::RecordSet(
            IRVal::Var(current_record.clone()),
            method.name.clone(),
            IRVal::Var(method_tmp),
        );
        let new_record_tmp = state.fresh_temp();
        bindings.push((new_record_tmp.clone(), IRComp::Effect(Box::new(new_record))));
        current_record = new_record_tmp;
    }

    // Return the final record
    let term = IRTerm::Return(IRVal::Var(current_record));
    Ok(wrap_lets(bindings, term))
}

/// Build a method closure that captures self and forwards remaining arguments.
/// For a method `Class::method(self, arg1, arg2, ...)`, create:
/// `\arg1 -> \arg2 -> ... -> Class::method(self, arg1, arg2, ...)`
fn build_method_closure(
    method_name: &str,
    self_var: &str,
    param_count: usize,
    state: &mut LowerState,
) -> IRVal {
    // Generate parameter names
    let param_names: Vec<String> = (0..param_count)
        .map(|i| format!("_arg{}", state.next_tmp + i))
        .collect();
    state.next_tmp += param_count;

    // Build the innermost application: Class::method(self)(arg1)(arg2)...
    let mut body_term = IRTerm::App(
        IRVal::Var(method_name.to_string()),
        IRVal::Var(self_var.to_string()),
    );
    for param_name in &param_names {
        // Wrap previous app in a let to sequence it, then apply next arg
        let tmp = format!("_app{}", state.next_tmp);
        state.next_tmp += 1;
        body_term = IRTerm::Let(
            tmp.clone(),
            Box::new(IRComp::Effect(Box::new(body_term))),
            Box::new(IRTerm::App(IRVal::Var(tmp), IRVal::Var(param_name.clone()))),
        );
    }

    // Wrap in lambdas from innermost to outermost
    let mut result = body_term;
    for param_name in param_names.into_iter().rev() {
        result = IRTerm::Return(IRVal::Lam(param_name, Box::new(result)));
    }

    // If no parameters, just return the partial application directly
    if param_count == 0 {
        // Method with no extra params: just apply self
        IRVal::Lam(
            "_unused".to_string(),
            Box::new(IRTerm::App(
                IRVal::Var(method_name.to_string()),
                IRVal::Var(self_var.to_string()),
            )),
        )
    } else {
        // Extract the lambda from the Return
        match result {
            IRTerm::Return(val) => val,
            _ => IRVal::Lam("_err".to_string(), Box::new(result)),
        }
    }
}

fn lower_handler_def(state: &mut LowerState, def: &HandlerDef) -> Result<IRHandler, LowerError> {
    let (ret_name, ret_expr) = &def.return_clause;
    let ret_term = lower_term(state, ret_expr)?;
    let mut clauses = Vec::new();
    for clause in &def.op_clauses {
        let body_term = lower_term(state, &clause.body)?;
        let body_term = IRTerm::Let(
            "resume".to_string(),
            Box::new(IRComp::Pure(IRVal::Var(clause.k.clone()))),
            Box::new(body_term),
        );
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
