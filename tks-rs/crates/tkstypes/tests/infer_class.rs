use tkscore::ast::{EffectRow, Type};
use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;

#[test]
fn infer_class_member_access_and_method_call() {
    let source = r#"
class Point {
  specifics { x: Int; }
  details { y: Int = identity.x; }
  actions { add(self, delta: Int): Int = self.y; }
}
\p -> let a = p.x in let b = p.y in p.add(a)
"#;
    let program = parse_program(source).expect("parse program");
    let out = infer_program(&program).expect("infer program");
    let entry = out.entry.expect("entry");
    match entry.ty {
        Type::Fun(param, ret) => {
            assert_eq!(*param, Type::Class("Point".to_string()));
            match *ret {
                Type::Effectful(inner, row) => {
                    assert_eq!(*inner, Type::Int);
                    assert_eq!(row, EffectRow::Empty);
                }
                _ => panic!("expected effectful return type"),
            }
        }
        _ => panic!("expected function type"),
    }
    assert_eq!(entry.effect, EffectRow::Empty);
}
