use tkscore::ast::{Expr, Type};

pub struct TypeEnv;

pub fn infer_expr(_env: &TypeEnv, _expr: &Expr) -> Result<Type, String> {
    Err("type inference not implemented".to_string())
}
