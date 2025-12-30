use tkscore::ast::{EffectRow, Expr, Type, World};
use tkscore::parser::parse_program;
use tkstypes::infer::{infer_expr, InferOutput, TypeEnv};

fn parse_entry(source: &str) -> Expr {
    let program = parse_program(source).expect("parse program");
    program.entry.expect("expected entry")
}

fn infer(source: &str) -> InferOutput {
    let expr = parse_entry(source);
    infer_expr(&TypeEnv::new(), &expr).expect("infer")
}

#[test]
fn infer_int_literal() {
    let out = infer("3");
    assert_eq!(out.ty, Type::Int);
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_lambda_identity() {
    let out = infer("\\x -> x");
    match out.ty {
        Type::Fun(param, ret) => match *ret {
            Type::Effectful(inner, row) => {
                assert_eq!(*param, *inner);
                assert_eq!(row, EffectRow::Empty);
            }
            _ => panic!("expected effectful return type"),
        },
        _ => panic!("expected function type"),
    }
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_application_identity() {
    let out = infer("(\\x -> x) 3");
    assert_eq!(out.ty, Type::Int);
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_if_expression() {
    let out = infer("if true then 1 else 2");
    assert_eq!(out.ty, Type::Int);
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_rpm_return() {
    let out = infer("return 1");
    assert_eq!(out.ty, Type::RPM(Box::new(Type::Int)));
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_noetic_element() {
    let out = infer("nu3(A1)");
    assert_eq!(out.ty, Type::Element(Some(World::A)));
    assert_eq!(out.effect, EffectRow::Empty);
}
