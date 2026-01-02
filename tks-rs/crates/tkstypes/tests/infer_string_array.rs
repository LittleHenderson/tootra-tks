//! Type inference tests for String and Array types
use tkscore::ast::{EffectRow, Expr, Literal, Type};
use tkscore::span::Span;
use tkstypes::infer::{infer_expr, InferOutput, TypeEnv};

fn dummy_span() -> Span {
    Span::new(0, 0, 1, 1)
}

fn infer(expr: Expr) -> InferOutput {
    infer_expr(&TypeEnv::new(), &expr).expect("infer")
}

#[test]
fn infer_empty_array() {
    let expr = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![],
    };
    let out = infer(expr);
    match out.ty {
        Type::Array(_) => {}
        other => panic!("expected Array type, got {:?}", other),
    }
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_int_array() {
    let expr = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Int(1) },
            Expr::Lit { span: dummy_span(), literal: Literal::Int(2) },
            Expr::Lit { span: dummy_span(), literal: Literal::Int(3) },
        ],
    };
    let out = infer(expr);
    assert_eq!(out.ty, Type::Array(Box::new(Type::Int)));
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_bool_array() {
    let expr = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Bool(true) },
            Expr::Lit { span: dummy_span(), literal: Literal::Bool(false) },
        ],
    };
    let out = infer(expr);
    assert_eq!(out.ty, Type::Array(Box::new(Type::Bool)));
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_heterogeneous_array_fails() {
    let expr = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Int(1) },
            Expr::Lit { span: dummy_span(), literal: Literal::Bool(true) },
        ],
    };
    let result = infer_expr(&TypeEnv::new(), &expr);
    assert!(result.is_err(), "heterogeneous array should fail type inference");
}

#[test]
fn infer_for_in_returns_unit() {
    let collection = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Int(1) },
            Expr::Lit { span: dummy_span(), literal: Literal::Int(2) },
        ],
    };
    let expr = Expr::ForIn {
        span: dummy_span(),
        binder: "x".to_string(),
        collection: Box::new(collection),
        body: Box::new(Expr::Var { span: dummy_span(), name: "x".to_string() }),
    };
    let out = infer(expr);
    assert_eq!(out.ty, Type::Unit);
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_index_access() {
    let array = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Int(1) },
            Expr::Lit { span: dummy_span(), literal: Literal::Int(2) },
        ],
    };
    let expr = Expr::Index {
        span: dummy_span(),
        array: Box::new(array),
        index: Box::new(Expr::Lit { span: dummy_span(), literal: Literal::Int(0) }),
    };
    let out = infer(expr);
    assert_eq!(out.ty, Type::Int);
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_nested_array() {
    let inner1 = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Int(1) },
            Expr::Lit { span: dummy_span(), literal: Literal::Int(2) },
        ],
    };
    let inner2 = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![
            Expr::Lit { span: dummy_span(), literal: Literal::Int(3) },
            Expr::Lit { span: dummy_span(), literal: Literal::Int(4) },
        ],
    };
    let expr = Expr::ArrayLit {
        span: dummy_span(),
        elements: vec![inner1, inner2],
    };
    let out = infer(expr);
    assert_eq!(out.ty, Type::Array(Box::new(Type::Array(Box::new(Type::Int)))));
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_string_literal() {
    let expr = Expr::Lit {
        span: dummy_span(),
        literal: Literal::Str("hello".to_string()),
    };
    let out = infer(expr);
    assert_eq!(out.ty, Type::Str);
    assert_eq!(out.effect, EffectRow::Empty);
}
