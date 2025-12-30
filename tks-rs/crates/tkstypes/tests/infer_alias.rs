use tkscore::ast::{EffectRow, Type};
use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;

#[test]
fn infer_type_alias() {
    let source = r#"
type Count = Int;
let x: Count = 3;
x
"#;
    let program = parse_program(source).expect("parse program");
    let output = infer_program(&program).expect("infer program");
    let entry = output.entry.expect("entry");
    assert_eq!(entry.ty, Type::Int);
    assert_eq!(entry.effect, EffectRow::Empty);
}
