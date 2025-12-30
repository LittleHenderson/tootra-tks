use std::collections::{HashMap, HashSet};

use tkscore::ast::{EffectRow, Expr, Type, TypeScheme};

#[derive(Debug, Clone)]
pub enum TypeError {
    Mismatch(Type, Type),
    Occurs(String, Type),
    RowOccurs(String, EffectRow),
    RowMismatch(EffectRow, EffectRow),
    UnknownVar(String),
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
            TypeError::UnknownVar(name) => write!(f, "unknown type variable: {name}"),
            TypeError::Unimplemented(feature) => write!(f, "{feature} not implemented"),
        }
    }
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

#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    values: HashMap<String, TypeScheme>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, scheme: TypeScheme) {
        self.values.insert(name.into(), scheme);
    }

    pub fn get(&self, name: &str) -> Option<&TypeScheme> {
        self.values.get(name)
    }
}

pub fn unify_types(subst: &mut Subst, left: &Type, right: &Type) -> Result<(), TypeError> {
    let left = subst.apply_type(left);
    let right = subst.apply_type(right);
    match (left, right) {
        (Type::Var(name), ty) | (ty, Type::Var(name)) => subst.bind_type_var(&name, &ty),
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

pub fn infer_expr(_env: &TypeEnv, _expr: &Expr) -> Result<Type, TypeError> {
    Err(TypeError::Unimplemented("type inference"))
}
