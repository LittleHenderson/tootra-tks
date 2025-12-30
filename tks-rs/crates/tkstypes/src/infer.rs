use std::collections::{HashMap, HashSet};

use tkscore::ast::{EffectRow, Expr, Literal, Type, TypeScheme};

#[derive(Debug, Clone)]
pub enum TypeError {
    Mismatch(Type, Type),
    Occurs(String, Type),
    RowOccurs(String, EffectRow),
    RowMismatch(EffectRow, EffectRow),
    UnknownVar(String),
    InvalidAnnotation(String),
    Unimplemented(&'static str),
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::Mismatch(left, right) => {
                write!(f, "type mismatch: {left:?} vs {right:?}")
            }
            TypeError::Occurs(var, ty) => write!(f, "occurs check failed: {var} in {ty:?}"),
            TypeError::RowOccurs(var, row) => write!(f, "row occurs check failed: {var} in {row:?}"),
            TypeError::RowMismatch(left, right) => {
                write!(f, "effect row mismatch: {left:?} vs {right:?}")
            }
            TypeError::UnknownVar(name) => write!(f, "unknown variable: {name}"),
            TypeError::InvalidAnnotation(message) => write!(f, "invalid annotation: {message}"),
            TypeError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferOutput {
    pub ty: Type,
    pub effect: EffectRow,
}

#[derive(Debug, Clone, Default)]
pub struct Subst {
    type_subst: HashMap<String, Type>,
    row_subst: HashMap<String, EffectRow>,
}

impl Subst {
    pub fn apply_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(name) => self
                .type_subst
                .get(name)
                .map(|t| self.apply_type(t))
                .unwrap_or_else(|| Type::Var(name.clone())),
            Type::Int => Type::Int,
            Type::Bool => Type::Bool,
            Type::Unit => Type::Unit,
            Type::Void => Type::Void,
            Type::Element(world) => Type::Element(*world),
            Type::Foundation => Type::Foundation,
            Type::Domain => Type::Domain,
            Type::Noetic(inner) => Type::Noetic(Box::new(self.apply_type(inner))),
            Type::Fractal(inner) => Type::Fractal(Box::new(self.apply_type(inner))),
            Type::RPM(inner) => Type::RPM(Box::new(self.apply_type(inner))),
            Type::Fun(left, right) => Type::Fun(
                Box::new(self.apply_type(left)),
                Box::new(self.apply_type(right)),
            ),
            Type::Product(left, right) => Type::Product(
                Box::new(self.apply_type(left)),
                Box::new(self.apply_type(right)),
            ),
            Type::Sum(left, right) => Type::Sum(
                Box::new(self.apply_type(left)),
                Box::new(self.apply_type(right)),
            ),
            Type::Effectful(inner, row) => {
                Type::Effectful(Box::new(self.apply_type(inner)), self.apply_row(row))
            }
            Type::Handler { effect, input, output } => Type::Handler {
                effect: self.apply_row(effect),
                input: Box::new(self.apply_type(input)),
                output: Box::new(self.apply_type(output)),
            },
            Type::Ordinal => Type::Ordinal,
            Type::QState(inner) => Type::QState(Box::new(self.apply_type(inner))),
        }
    }

    pub fn apply_row(&self, row: &EffectRow) -> EffectRow {
        match row {
            EffectRow::Empty => EffectRow::Empty,
            EffectRow::Cons(name, tail) => {
                EffectRow::Cons(name.clone(), Box::new(self.apply_row(tail)))
            }
            EffectRow::Var(name) => self
                .row_subst
                .get(name)
                .map(|r| self.apply_row(r))
                .unwrap_or_else(|| EffectRow::Var(name.clone())),
        }
    }

    pub fn row_binding(&self, name: &str) -> Option<EffectRow> {
        self.row_subst.get(name).cloned()
    }

    fn bind_type_var(&mut self, name: &str, ty: &Type) -> Result<(), TypeError> {
        if let Type::Var(existing) = ty {
            if existing == name {
                return Ok(());
            }
        }
        if type_contains_var(ty, name, self) {
            return Err(TypeError::Occurs(name.to_string(), ty.clone()));
        }
        self.type_subst.insert(name.to_string(), ty.clone());
        Ok(())
    }

    fn bind_row_var(&mut self, name: &str, row: &EffectRow) -> Result<(), TypeError> {
        if let EffectRow::Var(existing) = row {
            if existing == name {
                return Ok(());
            }
        }
        if row_contains_var(row, name, self) {
            return Err(TypeError::RowOccurs(name.to_string(), row.clone()));
        }
        self.row_subst.insert(name.to_string(), row.clone());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Scheme {
    type_vars: Vec<String>,
    row_vars: Vec<String>,
    ty: Type,
}

impl Scheme {
    pub fn mono(ty: Type) -> Self {
        Self {
            type_vars: Vec::new(),
            row_vars: Vec::new(),
            ty,
        }
    }

    fn instantiate(&self, state: &mut InferState) -> Type {
        let mut subst = Subst::default();
        for var in &self.type_vars {
            subst
                .type_subst
                .insert(var.clone(), state.fresh_type_var());
        }
        for var in &self.row_vars {
            subst
                .row_subst
                .insert(var.clone(), state.fresh_row_var());
        }
        subst.apply_type(&self.ty)
    }

    fn apply_subst(&self, subst: &Subst) -> Self {
        let mut filtered = subst.clone();
        for var in &self.type_vars {
            filtered.type_subst.remove(var);
        }
        for var in &self.row_vars {
            filtered.row_subst.remove(var);
        }
        Self {
            type_vars: self.type_vars.clone(),
            row_vars: self.row_vars.clone(),
            ty: filtered.apply_type(&self.ty),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    values: HashMap<String, Scheme>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, scheme: Scheme) {
        self.values.insert(name.into(), scheme);
    }

    pub fn get(&self, name: &str) -> Option<&Scheme> {
        self.values.get(name)
    }

    fn apply_subst(&mut self, subst: &Subst) {
        self.values = self
            .values
            .iter()
            .map(|(name, scheme)| (name.clone(), scheme.apply_subst(subst)))
            .collect();
    }
}

#[derive(Debug, Default)]
struct InferState {
    subst: Subst,
    next_type: usize,
    next_row: usize,
}

impl InferState {
    fn fresh_type_var(&mut self) -> Type {
        let name = format!("t{}", self.next_type);
        self.next_type += 1;
        Type::Var(name)
    }

    fn fresh_row_var(&mut self) -> EffectRow {
        let name = format!("r{}", self.next_row);
        self.next_row += 1;
        EffectRow::Var(name)
    }
}

pub fn unify_types(subst: &mut Subst, left: &Type, right: &Type) -> Result<(), TypeError> {
    let left = subst.apply_type(left);
    let right = subst.apply_type(right);
    match (left, right) {
        (Type::Var(name), ty) | (ty, Type::Var(name)) => subst.bind_type_var(&name, &ty),
        (left, right) => {
            let (left, right) = normalize_effectful(left, right);
            match (left, right) {
                (Type::Int, Type::Int)
                | (Type::Bool, Type::Bool)
                | (Type::Unit, Type::Unit)
                | (Type::Void, Type::Void)
                | (Type::Foundation, Type::Foundation)
                | (Type::Domain, Type::Domain)
                | (Type::Ordinal, Type::Ordinal) => Ok(()),
                (Type::Element(left_world), Type::Element(right_world)) => {
                    match (left_world, right_world) {
                        (Some(l), Some(r)) if l != r => Err(TypeError::Mismatch(
                            Type::Element(Some(l)),
                            Type::Element(Some(r)),
                        )),
                        _ => Ok(()),
                    }
                }
                (Type::Noetic(left), Type::Noetic(right))
                | (Type::Fractal(left), Type::Fractal(right))
                | (Type::RPM(left), Type::RPM(right))
                | (Type::QState(left), Type::QState(right)) => unify_types(subst, &left, &right),
                (Type::Fun(l1, r1), Type::Fun(l2, r2))
                | (Type::Product(l1, r1), Type::Product(l2, r2))
                | (Type::Sum(l1, r1), Type::Sum(l2, r2)) => {
                    unify_types(subst, &l1, &l2)?;
                    unify_types(subst, &r1, &r2)
                }
                (Type::Effectful(ty1, row1), Type::Effectful(ty2, row2)) => {
                    unify_types(subst, &ty1, &ty2)?;
                    unify_rows(subst, &row1, &row2)
                }
                (
                    Type::Handler {
                        effect: eff1,
                        input: in1,
                        output: out1,
                    },
                    Type::Handler {
                        effect: eff2,
                        input: in2,
                        output: out2,
                    },
                ) => {
                    unify_rows(subst, &eff1, &eff2)?;
                    unify_types(subst, &in1, &in2)?;
                    unify_types(subst, &out1, &out2)
                }
                (left, right) => Err(TypeError::Mismatch(left, right)),
            }
        }
    }
}

fn normalize_effectful(left: Type, right: Type) -> (Type, Type) {
    match (&left, &right) {
        (Type::Effectful(_, _), Type::Effectful(_, _)) => (left, right),
        (Type::Effectful(_, _), _) => {
            (left, Type::Effectful(Box::new(right), EffectRow::Empty))
        }
        (_, Type::Effectful(_, _)) => {
            (Type::Effectful(Box::new(left), EffectRow::Empty), right)
        }
        _ => (left, right),
    }
}

pub fn unify_rows(subst: &mut Subst, left: &EffectRow, right: &EffectRow) -> Result<(), TypeError> {
    let left = subst.apply_row(left);
    let right = subst.apply_row(right);
    if left == right {
        return Ok(());
    }
    match (&left, &right) {
        (EffectRow::Var(name), row) | (row, EffectRow::Var(name)) => {
            return subst.bind_row_var(name, row);
        }
        _ => {}
    }

    let (left_labels, left_tail) = flatten_row(&left);
    let (right_labels, right_tail) = flatten_row(&right);

    let mut right_set: HashSet<String> = right_labels.iter().cloned().collect();
    let mut left_only = Vec::new();
    for label in left_labels {
        if !right_set.remove(&label) {
            left_only.push(label);
        }
    }
    let right_only: Vec<String> = right_set.into_iter().collect();

    match (left_tail, right_tail) {
        (None, None) => {
            if left_only.is_empty() && right_only.is_empty() {
                Ok(())
            } else {
                Err(TypeError::RowMismatch(left, right))
            }
        }
        (Some(tail), None) => {
            if !left_only.is_empty() {
                return Err(TypeError::RowMismatch(left, right));
            }
            let row = build_row(right_only, None);
            subst.bind_row_var(&tail, &row)
        }
        (None, Some(tail)) => {
            if !right_only.is_empty() {
                return Err(TypeError::RowMismatch(left, right));
            }
            let row = build_row(left_only, None);
            subst.bind_row_var(&tail, &row)
        }
        (Some(left_tail), Some(right_tail)) => {
            if left_tail == right_tail {
                if left_only.is_empty() && right_only.is_empty() {
                    return Ok(());
                }
                return Err(TypeError::RowMismatch(left, right));
            }
            if !left_only.is_empty() && !right_only.is_empty() {
                return Err(TypeError::RowMismatch(left, right));
            }
            if left_only.is_empty() {
                let row = build_row(right_only, Some(right_tail.clone()));
                subst.bind_row_var(&left_tail, &row)
            } else {
                let row = build_row(left_only, Some(left_tail.clone()));
                subst.bind_row_var(&right_tail, &row)
            }
        }
    }
}

pub fn infer_expr(env: &TypeEnv, expr: &Expr) -> Result<InferOutput, TypeError> {
    let mut state = InferState::default();
    let mut env = env.clone();
    let out = infer_expr_inner(&mut state, &mut env, expr)?;
    Ok(InferOutput {
        ty: state.subst.apply_type(&out.ty),
        effect: state.subst.apply_row(&out.effect),
    })
}

fn infer_expr_inner(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    match expr {
        Expr::Var { name, .. } => {
            let scheme = env
                .get(name)
                .ok_or_else(|| TypeError::UnknownVar(name.clone()))?;
            Ok(InferOutput {
                ty: scheme.instantiate(state),
                effect: EffectRow::Empty,
            })
        }
        Expr::Lit { literal, .. } => Ok(InferOutput {
            ty: infer_literal_type(literal)?,
            effect: EffectRow::Empty,
        }),
        Expr::Element { world, .. } => Ok(InferOutput {
            ty: Type::Element(Some(*world)),
            effect: EffectRow::Empty,
        }),
        Expr::Foundation { .. } => Ok(InferOutput {
            ty: Type::Foundation,
            effect: EffectRow::Empty,
        }),
        Expr::Lam {
            param,
            param_type,
            body,
            ..
        } => {
            let param_ty = match param_type {
                Some(ty) => extract_param_type(ty)?,
                None => state.fresh_type_var(),
            };
            let mut local = env.clone();
            local.insert(param.clone(), Scheme::mono(param_ty.clone()));
            let body_out = infer_expr_inner(state, &mut local, body)?;
            Ok(InferOutput {
                ty: Type::Fun(
                    Box::new(param_ty),
                    Box::new(Type::Effectful(
                        Box::new(body_out.ty),
                        body_out.effect.clone(),
                    )),
                ),
                effect: EffectRow::Empty,
            })
        }
        Expr::App { func, arg, .. } => {
            let func_out = infer_expr_inner(state, env, func)?;
            let arg_out = infer_expr_inner(state, env, arg)?;
            let result_ty = state.fresh_type_var();
            let result_eff = state.fresh_row_var();
            let expected = Type::Fun(
                Box::new(arg_out.ty.clone()),
                Box::new(Type::Effectful(
                    Box::new(result_ty.clone()),
                    result_eff.clone(),
                )),
            );
            unify_types(&mut state.subst, &func_out.ty, &expected)?;
            let effect = combine_effects(state, &func_out.effect, &arg_out.effect)?;
            let effect = combine_effects(state, &effect, &result_eff)?;
            Ok(InferOutput {
                ty: state.subst.apply_type(&result_ty),
                effect: state.subst.apply_row(&effect),
            })
        }
        Expr::Let {
            name,
            scheme,
            value,
            body,
            ..
        } => {
            let value_out = infer_expr_inner(state, env, value)?;
            if let Some(annotation) = scheme {
                check_annotation(state, annotation, &value_out)?;
            }
            env.apply_subst(&state.subst);
            let value_ty = state.subst.apply_type(&value_out.ty);
            let scheme = generalize(env, &value_ty);
            let mut local = env.clone();
            local.insert(name.clone(), scheme);
            let body_out = infer_expr_inner(state, &mut local, body)?;
            let effect = combine_effects(state, &value_out.effect, &body_out.effect)?;
            Ok(InferOutput {
                ty: body_out.ty,
                effect,
            })
        }
        Expr::If {
            cond,
            then_branch,
            else_branch,
            ..
        } => {
            let cond_out = infer_expr_inner(state, env, cond)?;
            unify_types(&mut state.subst, &cond_out.ty, &Type::Bool)?;
            let then_out = infer_expr_inner(state, env, then_branch)?;
            let else_out = infer_expr_inner(state, env, else_branch)?;
            unify_types(&mut state.subst, &then_out.ty, &else_out.ty)?;
            let effect = combine_effects(state, &cond_out.effect, &then_out.effect)?;
            let effect = combine_effects(state, &effect, &else_out.effect)?;
            Ok(InferOutput {
                ty: state.subst.apply_type(&then_out.ty),
                effect,
            })
        }
        Expr::Noetic { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            Ok(inner)
        }
        Expr::Fractal { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            Ok(inner)
        }
        Expr::RPMReturn { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(inner.ty)),
                effect: inner.effect,
            })
        }
        Expr::RPMCheck { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            unify_types(&mut state.subst, &inner.ty, &Type::Element(None))?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(Type::Bool)),
                effect: inner.effect,
            })
        }
        Expr::RPMAcquire { expr, .. } => {
            let inner = infer_expr_inner(state, env, expr)?;
            unify_types(&mut state.subst, &inner.ty, &Type::Element(None))?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(Type::Unit)),
                effect: inner.effect,
            })
        }
        Expr::RPMBind { left, right, .. } => {
            let left_out = infer_expr_inner(state, env, left)?;
            let right_out = infer_expr_inner(state, env, right)?;
            let arg_ty = state.fresh_type_var();
            let res_ty = state.fresh_type_var();
            unify_types(&mut state.subst, &left_out.ty, &Type::RPM(Box::new(arg_ty.clone())))?;
            unify_types(
                &mut state.subst,
                &right_out.ty,
                &Type::Fun(
                    Box::new(arg_ty.clone()),
                    Box::new(Type::RPM(Box::new(res_ty.clone()))),
                ),
            )?;
            let effect = combine_effects(state, &left_out.effect, &right_out.effect)?;
            Ok(InferOutput {
                ty: Type::RPM(Box::new(res_ty)),
                effect,
            })
        }
        Expr::ACBE { .. } => Err(TypeError::Unimplemented("acbe")),
        Expr::Handle { .. } => Err(TypeError::Unimplemented("handle expressions")),
        Expr::Perform { .. } => Err(TypeError::Unimplemented("perform expressions")),
        Expr::OrdLit { .. }
        | Expr::OrdOmega { .. }
        | Expr::OrdEpsilon { .. }
        | Expr::OrdAleph { .. } => Ok(InferOutput {
            ty: Type::Ordinal,
            effect: EffectRow::Empty,
        }),
        Expr::OrdSucc { expr, .. } => infer_ordinal_unary(state, env, expr),
        Expr::OrdAdd { left, right, .. }
        | Expr::OrdMul { left, right, .. }
        | Expr::OrdExp { left, right, .. } => infer_ordinal_binary(state, env, left, right),
        Expr::OrdLimit {
            binder,
            bound,
            body,
            ..
        } => {
            let bound_out = infer_expr_inner(state, env, bound)?;
            unify_types(&mut state.subst, &bound_out.ty, &Type::Ordinal)?;
            let mut local = env.clone();
            local.insert(binder.clone(), Scheme::mono(Type::Ordinal));
            let body_out = infer_expr_inner(state, &mut local, body)?;
            unify_types(&mut state.subst, &body_out.ty, &Type::Ordinal)?;
            let effect = combine_effects(state, &bound_out.effect, &body_out.effect)?;
            Ok(InferOutput {
                ty: Type::Ordinal,
                effect,
            })
        }
        Expr::TransfiniteLoop { .. } => Err(TypeError::Unimplemented("transfinite loop")),
        Expr::Superpose { .. } => Err(TypeError::Unimplemented("quantum superpose")),
        Expr::Measure { .. } => Err(TypeError::Unimplemented("quantum measure")),
        Expr::Entangle { .. } => Err(TypeError::Unimplemented("quantum entangle")),
        Expr::Ket { .. } => Err(TypeError::Unimplemented("ket")),
        Expr::Bra { .. } => Err(TypeError::Unimplemented("bra")),
        Expr::BraKet { .. } => Err(TypeError::Unimplemented("bra-ket")),
    }
}

fn infer_literal_type(literal: &Literal) -> Result<Type, TypeError> {
    match literal {
        Literal::Int(_) => Ok(Type::Int),
        Literal::Bool(_) => Ok(Type::Bool),
        Literal::Float(_) => Err(TypeError::Unimplemented("float literals")),
        Literal::Complex { .. } => Err(TypeError::Unimplemented("complex literals")),
        Literal::Unit => Ok(Type::Unit),
    }
}

fn extract_param_type(ty: &Type) -> Result<Type, TypeError> {
    match ty {
        Type::Effectful(_, _) => Err(TypeError::InvalidAnnotation(
            "parameter types cannot be effectful".to_string(),
        )),
        _ => Ok(ty.clone()),
    }
}

fn infer_ordinal_unary(
    state: &mut InferState,
    env: &mut TypeEnv,
    expr: &Expr,
) -> Result<InferOutput, TypeError> {
    let inner = infer_expr_inner(state, env, expr)?;
    unify_types(&mut state.subst, &inner.ty, &Type::Ordinal)?;
    Ok(InferOutput {
        ty: Type::Ordinal,
        effect: inner.effect,
    })
}

fn infer_ordinal_binary(
    state: &mut InferState,
    env: &mut TypeEnv,
    left: &Expr,
    right: &Expr,
) -> Result<InferOutput, TypeError> {
    let left_out = infer_expr_inner(state, env, left)?;
    let right_out = infer_expr_inner(state, env, right)?;
    unify_types(&mut state.subst, &left_out.ty, &Type::Ordinal)?;
    unify_types(&mut state.subst, &right_out.ty, &Type::Ordinal)?;
    let effect = combine_effects(state, &left_out.effect, &right_out.effect)?;
    Ok(InferOutput {
        ty: Type::Ordinal,
        effect,
    })
}

fn combine_effects(
    state: &mut InferState,
    left: &EffectRow,
    right: &EffectRow,
) -> Result<EffectRow, TypeError> {
    let left = state.subst.apply_row(left);
    let right = state.subst.apply_row(right);
    if left == EffectRow::Empty {
        return Ok(right);
    }
    if right == EffectRow::Empty {
        return Ok(left);
    }
    let (mut labels_left, tail_left) = flatten_row(&left);
    let (labels_right, tail_right) = flatten_row(&right);
    labels_left.extend(labels_right);
    let tail = match (tail_left, tail_right) {
        (None, None) => None,
        (Some(tail), None) | (None, Some(tail)) => Some(tail),
        (Some(left_tail), Some(right_tail)) => {
            if left_tail != right_tail {
                let left_var = EffectRow::Var(left_tail.clone());
                let right_var = EffectRow::Var(right_tail.clone());
                unify_rows(&mut state.subst, &left_var, &right_var)?;
            }
            Some(left_tail)
        }
    };
    Ok(build_row(labels_left, tail))
}

fn check_annotation(
    state: &mut InferState,
    annotation: &TypeScheme,
    out: &InferOutput,
) -> Result<(), TypeError> {
    let scheme = scheme_from_annotation(annotation);
    let inst = scheme.instantiate(state);
    match inst {
        Type::Effectful(inner, row) => {
            unify_types(&mut state.subst, &inner, &out.ty)?;
            unify_rows(&mut state.subst, &row, &out.effect)
        }
        _ => {
            unify_types(&mut state.subst, &inst, &out.ty)?;
            unify_rows(&mut state.subst, &out.effect, &EffectRow::Empty)
        }
    }
}

fn scheme_from_annotation(annotation: &TypeScheme) -> Scheme {
    let row_vars = free_row_vars_type(&annotation.ty);
    Scheme {
        type_vars: annotation.vars.clone(),
        row_vars: row_vars.into_iter().collect(),
        ty: annotation.ty.clone(),
    }
}

fn generalize(env: &TypeEnv, ty: &Type) -> Scheme {
    let mut type_vars = free_type_vars_type(ty);
    let env_type_vars = free_type_vars_env(env);
    type_vars.retain(|var| !env_type_vars.contains(var));

    let mut row_vars = free_row_vars_type(ty);
    let env_row_vars = free_row_vars_env(env);
    row_vars.retain(|var| !env_row_vars.contains(var));

    Scheme {
        type_vars: type_vars.into_iter().collect(),
        row_vars: row_vars.into_iter().collect(),
        ty: ty.clone(),
    }
}

fn free_type_vars_env(env: &TypeEnv) -> HashSet<String> {
    env.values
        .values()
        .flat_map(|scheme| free_type_vars_scheme(scheme))
        .collect()
}

fn free_row_vars_env(env: &TypeEnv) -> HashSet<String> {
    env.values
        .values()
        .flat_map(|scheme| free_row_vars_scheme(scheme))
        .collect()
}

fn free_type_vars_scheme(scheme: &Scheme) -> HashSet<String> {
    let mut vars = free_type_vars_type(&scheme.ty);
    for var in &scheme.type_vars {
        vars.remove(var);
    }
    vars
}

fn free_row_vars_scheme(scheme: &Scheme) -> HashSet<String> {
    let mut vars = free_row_vars_type(&scheme.ty);
    for var in &scheme.row_vars {
        vars.remove(var);
    }
    vars
}

fn free_type_vars_type(ty: &Type) -> HashSet<String> {
    match ty {
        Type::Var(name) => HashSet::from([name.clone()]),
        Type::Noetic(inner)
        | Type::Fractal(inner)
        | Type::RPM(inner)
        | Type::QState(inner) => free_type_vars_type(inner),
        Type::Fun(left, right)
        | Type::Product(left, right)
        | Type::Sum(left, right) => {
            let mut vars = free_type_vars_type(left);
            vars.extend(free_type_vars_type(right));
            vars
        }
        Type::Effectful(inner, _) => free_type_vars_type(inner),
        Type::Handler { input, output, .. } => {
            let mut vars = free_type_vars_type(input);
            vars.extend(free_type_vars_type(output));
            vars
        }
        _ => HashSet::new(),
    }
}

fn free_row_vars_type(ty: &Type) -> HashSet<String> {
    match ty {
        Type::Effectful(inner, row) => {
            let mut vars = free_row_vars_row(row);
            vars.extend(free_row_vars_type(inner));
            vars
        }
        Type::Handler {
            effect,
            input,
            output,
        } => {
            let mut vars = free_row_vars_row(effect);
            vars.extend(free_row_vars_type(input));
            vars.extend(free_row_vars_type(output));
            vars
        }
        Type::Noetic(inner)
        | Type::Fractal(inner)
        | Type::RPM(inner)
        | Type::QState(inner) => free_row_vars_type(inner),
        Type::Fun(left, right)
        | Type::Product(left, right)
        | Type::Sum(left, right) => {
            let mut vars = free_row_vars_type(left);
            vars.extend(free_row_vars_type(right));
            vars
        }
        _ => HashSet::new(),
    }
}

fn free_row_vars_row(row: &EffectRow) -> HashSet<String> {
    match row {
        EffectRow::Empty => HashSet::new(),
        EffectRow::Cons(_, tail) => free_row_vars_row(tail),
        EffectRow::Var(name) => HashSet::from([name.clone()]),
    }
}

fn type_contains_var(ty: &Type, name: &str, subst: &Subst) -> bool {
    match subst.apply_type(ty) {
        Type::Var(var) => var == name,
        Type::Noetic(inner)
        | Type::Fractal(inner)
        | Type::RPM(inner)
        | Type::QState(inner) => type_contains_var(&inner, name, subst),
        Type::Fun(left, right)
        | Type::Product(left, right)
        | Type::Sum(left, right) => {
            type_contains_var(&left, name, subst) || type_contains_var(&right, name, subst)
        }
        Type::Effectful(inner, row) => {
            type_contains_var(&inner, name, subst) || row_contains_var(&row, name, subst)
        }
        Type::Handler {
            effect,
            input,
            output,
        } => {
            row_contains_var(&effect, name, subst)
                || type_contains_var(&input, name, subst)
                || type_contains_var(&output, name, subst)
        }
        _ => false,
    }
}

fn row_contains_var(row: &EffectRow, name: &str, subst: &Subst) -> bool {
    match subst.apply_row(row) {
        EffectRow::Empty => false,
        EffectRow::Cons(_, tail) => row_contains_var(&tail, name, subst),
        EffectRow::Var(var) => var == name,
    }
}

fn flatten_row(row: &EffectRow) -> (Vec<String>, Option<String>) {
    let mut labels = Vec::new();
    let mut tail = None;
    let mut current = row;
    loop {
        match current {
            EffectRow::Empty => break,
            EffectRow::Var(name) => {
                tail = Some(name.clone());
                break;
            }
            EffectRow::Cons(name, next) => {
                labels.push(name.clone());
                current = next;
            }
        }
    }
    labels.sort();
    labels.dedup();
    (labels, tail)
}

fn build_row(mut labels: Vec<String>, tail: Option<String>) -> EffectRow {
    labels.sort();
    labels.dedup();
    let mut row = match tail {
        Some(name) => EffectRow::Var(name),
        None => EffectRow::Empty,
    };
    for label in labels.into_iter().rev() {
        row = EffectRow::Cons(label, Box::new(row));
    }
    row
}
