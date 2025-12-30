use tkscore::ast::{EffectRow, Type, World};
use tkscore::parser::parse_program;
use tkstypes::infer::{infer_expr, InferOutput, TypeEnv};

fn infer_entry(source: &str) -> InferOutput {
    let program = parse_program(source).expect("parse program");
    let expr = program.entry.expect("entry");
    infer_expr(&TypeEnv::new(), &expr).expect("infer")
}

#[test]
fn infer_ket_measure() {
    let out = infer_entry("measure(|A1>)");
    assert_eq!(out.ty, Type::Element(Some(World::A)));
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_superpose_entangle() {
    let out = infer_entry("entangle(superpose { 0.5: |A1>, 0.5: |A2> }, |B1>)");
    match out.ty {
        Type::QState(inner) => match *inner {
            Type::Product(left, right) => {
                assert_eq!(*left, Type::Element(Some(World::A)));
                assert_eq!(*right, Type::Element(Some(World::B)));
            }
            _ => panic!("expected product in entangled state"),
        },
        _ => panic!("expected qstate"),
    }
}

#[test]
fn infer_braket_domain() {
    let out = infer_entry("<A1|A2>");
    assert_eq!(out.ty, Type::Domain);
    assert_eq!(out.effect, EffectRow::Empty);
}
