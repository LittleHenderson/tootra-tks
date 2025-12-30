use tkscore::parser::parse_program;
use tkscore::resolve::{resolve_program, ResolveError};

#[test]
fn resolve_imports_and_aliases() {
    let source = r#"
module A {
  export { foo, type T }
  let foo = 1;
  type T = Int;
}
module B {
  import A as M;
  from A import { foo as bar, T };
  let baz = bar;
}
"#;

    let program = parse_program(source).expect("parse program");
    let resolved = resolve_program(&program).expect("resolve program");
    let module_b = resolved.module(&["B"]).expect("module B resolved");

    assert!(module_b.scope.modules.contains_key("M"));
    assert!(module_b.scope.values.contains("bar"));
    assert!(module_b.scope.values.contains("baz"));
    assert!(!module_b.scope.values.contains("foo"));
    assert!(module_b.scope.types.contains("T"));
}

#[test]
fn resolve_unknown_import_item() {
    let source = r#"
module A {
  let foo = 1;
}
module B {
  from A import { missing };
}
"#;

    let program = parse_program(source).expect("parse program");
    let err = resolve_program(&program).expect_err("expected resolve error");
    assert!(matches!(err, ResolveError::UnknownImportItem { .. }));
}
