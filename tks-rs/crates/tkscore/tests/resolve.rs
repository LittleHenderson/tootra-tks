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

#[test]
fn resolve_effect_and_extern_exports() {
    let source = r#"
module A {
  export { effect IO, ping }
  effect IO {
    op log(msg: Int): Unit;
  }
  external c safe fn ping(x: Int): Int !{IO};
}
module B {
  from A import { IO, ping };
  let use = ping;
}
"#;

    let program = parse_program(source).expect("parse program");
    let resolved = resolve_program(&program).expect("resolve program");
    let module_a = resolved.module(&["A"]).expect("module A resolved");
    let module_b = resolved.module(&["B"]).expect("module B resolved");

    assert_eq!(module_a.exports.effects.len(), 1);
    assert_eq!(module_a.exports.effects[0].name, "IO");
    assert_eq!(module_a.exports.externs.len(), 1);
    assert_eq!(module_a.exports.externs[0].name, "ping");
    assert!(module_b.scope.effects.contains_key("IO"));
    assert!(module_b.scope.externs.contains_key("ping"));
}
