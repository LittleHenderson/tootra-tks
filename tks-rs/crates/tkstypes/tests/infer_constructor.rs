use tkscore::ast::{EffectRow, Type};
use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;

#[test]
fn infer_constructor_expression() {
    let source = r#"
class Point {
  specifics { x: Int; y: Int; }
  details { }
  actions { }
}
repeat Point { x: 1, y: 2 }
"#;
    let program = parse_program(source).expect("parse program");
    let out = infer_program(&program).expect("infer program");
    let entry = out.entry.expect("entry");
    assert_eq!(entry.ty, Type::Class("Point".to_string()));
    assert_eq!(entry.effect, EffectRow::Empty);
}
