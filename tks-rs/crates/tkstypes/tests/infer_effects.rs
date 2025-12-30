use tkscore::ast::{EffectRow, Type};
use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;

fn infer_entry(source: &str) -> (Type, EffectRow) {
    let program = parse_program(source).expect("parse program");
    let output = infer_program(&program).expect("infer program");
    let entry = output.entry.expect("entry");
    (entry.ty, entry.effect)
}

#[test]
fn infer_named_handler() {
    let source = r#"
effect Log {
  op log(msg: Int): Unit;
}

handler H: Handler[{Log}, Unit, Unit] {
  return x -> x;
  log(msg) k -> k(());
}

handle perform log(1) with H
"#;
    let (ty, effect) = infer_entry(source);
    assert_eq!(ty, Type::Unit);
    assert_eq!(effect, EffectRow::Empty);
}

#[test]
fn infer_inline_handler() {
    let source = r#"
effect Ping {
  op ping(n: Int): Int;
}

handle perform ping(1) with {
  return x -> x;
  ping(n) k -> k(n);
}
"#;
    let (ty, effect) = infer_entry(source);
    assert_eq!(ty, Type::Int);
    assert_eq!(effect, EffectRow::Empty);
}
