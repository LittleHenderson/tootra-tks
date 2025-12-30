use tkscore::parser::parse_program;
use tkscore::resolve::resolve_program;
use tkscore::tksi::emit_tksi;

#[test]
fn emits_effect_and_extern_signatures() {
    let source = r#"
module Net {
  export { effect IO, ping }
  effect IO {
    op log(msg: Int): Unit;
  }
  extern c safe fn ping(x: Int): Int !{IO};
}
"#;

    let program = parse_program(source).expect("parse program");
    let resolved = resolve_program(&program).expect("resolve program");
    let tksi = emit_tksi(&resolved);

    assert!(tksi.contains("module Net"));
    assert!(tksi.contains("export effect IO"));
    assert!(tksi.contains("op log(arg: Int): Unit"));
    assert!(tksi.contains("export extern c safe fn ping(x: Int): Int !{IO}"));
}
