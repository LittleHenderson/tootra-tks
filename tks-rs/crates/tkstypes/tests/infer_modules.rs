use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;

#[test]
fn infer_module_imports() {
    let source = r#"
module A {
  let x = 1;
}

module B {
  import A;
  let y: Int = x;
}
"#;
    let program = parse_program(source).expect("parse program");
    infer_program(&program).expect("infer program");
}

#[test]
fn infer_module_selective_import_alias() {
    let source = r#"
module A {
  let x = 1;
}

module B {
  from A import { x as y };
  let z: Int = y;
}
"#;
    let program = parse_program(source).expect("parse program");
    infer_program(&program).expect("infer program");
}

#[test]
fn infer_module_exports_filter() {
    let source = r#"
module A {
  export { x }
  let x = 1;
  let y = 2;
}

module B {
  import A;
  let z: Int = x;
  let w: Int = y;
}
"#;
    let program = parse_program(source).expect("parse program");
    assert!(infer_program(&program).is_err());
}
