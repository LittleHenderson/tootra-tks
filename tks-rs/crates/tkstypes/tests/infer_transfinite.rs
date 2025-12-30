use tkscore::ast::{EffectRow, Type, World};
use tkscore::parser::parse_program;
use tkstypes::infer::{infer_expr, InferOutput, TypeEnv};

fn infer_entry(source: &str) -> InferOutput {
    let program = parse_program(source).expect("parse program");
    let expr = program.entry.expect("entry");
    infer_expr(&TypeEnv::new(), &expr).expect("infer")
}

#[test]
fn infer_transfinite_loop() {
    let out = infer_entry("transfinite loop beta < omega from A1 step (x -> x) limit (f -> f)");
    assert_eq!(out.ty, Type::Element(Some(World::A)));
    assert_eq!(out.effect, EffectRow::Empty);
}

#[test]
fn infer_acbe_basic() {
    let out = infer_entry("acbe(1, 1)");
    assert_eq!(out.ty, Type::Int);
    assert_eq!(out.effect, EffectRow::Empty);
}
