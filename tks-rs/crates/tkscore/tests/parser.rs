use tkscore::ast::{Convention, Expr, Safety, TopDecl};
use tkscore::parser::parse_program;

#[test]
fn parses_basic_program_with_entry() {
    let source = "let x = A1; let y: Int = 3; x";
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 2);
    assert!(program.entry.is_some());
}

#[test]
fn parses_effect_and_handler_decls() {
    let source = r#"
effect QuantumEffect[A] {
  op measure(q: QState[A]): A;
}

handler HQ: Handler[{QuantumEffect}, QState[A], A] {
  return x -> x;
  measure(q) k -> k(q);
}
"#;
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 2);
    match &program.decls[0] {
        TopDecl::EffectDecl { params, .. } => {
            assert_eq!(params, &vec!["A".to_string()]);
        }
        _ => panic!("expected effect declaration"),
    }
}

#[test]
fn parses_transfinite_loop_entry() {
    let source =
        "transfinite loop beta < omega from A1 step (x -> nu3(x)) limit (f -> f)";
    let program = parse_program(source).expect("parse program");
    assert!(matches!(program.entry, Some(Expr::TransfiniteLoop { .. })));
}

#[test]
fn parses_fractals_and_quantum_forms() {
    let source = r#"
let phi = <<1:2:...>>_omega(A1);
let psi = superpose { 0.707: |A1>, 0.707: |A2> };
let chi = superpose[(0.707, |A1>), (0.707, |A2>)];
entangle(psi, chi)
"#;
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 3);
    assert!(matches!(program.entry, Some(Expr::Entangle { .. })));
}

#[test]
fn parses_braket_expression() {
    let source = "<A1|A2>";
    let program = parse_program(source).expect("parse program");
    assert!(matches!(program.entry, Some(Expr::BraKet { .. })));
}

#[test]
fn parses_extern_decl_alias() {
    let source = "extern c unsafe fn ping(x: Int): Int !{IO}";
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 1);
    match &program.decls[0] {
        TopDecl::ExternDecl(decl) => {
            assert_eq!(decl.name, "ping");
            assert_eq!(decl.convention, Convention::C);
            assert_eq!(decl.safety, Safety::Unsafe);
        }
        _ => panic!("expected extern declaration"),
    }
}

#[test]
fn parses_blueprint_class_decl() {
    let source = r#"
blueprint Person {
  specifics {
    mut name: Int;
    age: Int;
  }
  details {
    title: Int = A1;
    motto = A2;
  }
  actions {
    greet(identity, other: Int): Int = identity;
  }
}
"#;
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 1);
    match &program.decls[0] {
        TopDecl::ClassDecl(decl) => {
            assert_eq!(decl.name, "Person");
            assert_eq!(decl.specifics.len(), 2);
            assert!(decl.specifics[0].mutable);
            assert_eq!(decl.details.len(), 2);
            assert_eq!(decl.actions.len(), 1);
            assert_eq!(decl.actions[0].self_param, "identity");
            assert_eq!(decl.actions[0].params.len(), 1);
        }
        _ => panic!("expected class declaration"),
    }
}

#[test]
fn parses_class_alias_sections() {
    let source = r#"
class Box {
  field {
    mut size: Int;
  }
  details {
    tag = A1;
  }
  method {
    resize(self, delta: Int): Int = self;
  }
}
"#;
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 1);
    match &program.decls[0] {
        TopDecl::ClassDecl(decl) => {
            assert_eq!(decl.name, "Box");
            assert_eq!(decl.specifics.len(), 1);
            assert_eq!(decl.details.len(), 1);
            assert_eq!(decl.actions.len(), 1);
            assert_eq!(decl.actions[0].self_param, "self");
        }
        _ => panic!("expected class declaration"),
    }
}

#[test]
fn parses_plan_class_alias() {
    let source = r#"
plan Widget {
  description {
    size: Int;
  }
  details {
    tag = A1;
  }
  actions {
    use(identity): Int = identity;
  }
}
"#;
    let program = parse_program(source).expect("parse program");
    assert_eq!(program.decls.len(), 1);
    match &program.decls[0] {
        TopDecl::ClassDecl(decl) => {
            assert_eq!(decl.name, "Widget");
            assert_eq!(decl.specifics.len(), 1);
            assert_eq!(decl.details.len(), 1);
            assert_eq!(decl.actions.len(), 1);
            assert_eq!(decl.actions[0].self_param, "identity");
        }
        _ => panic!("expected class declaration"),
    }
}

#[test]
fn parses_constructor_and_member_access() {
    let source = "repeat Box { size: 1 }.size";
    let program = parse_program(source).expect("parse program");
    match program.entry {
        Some(Expr::Member { object, field, .. }) => {
            assert_eq!(field, "size");
            assert!(matches!(*object, Expr::Constructor { .. }));
        }
        _ => panic!("expected member access"),
    }
}

#[test]
fn parses_new_constructor() {
    let source = "new Box { size: 1 }";
    let program = parse_program(source).expect("parse program");
    assert!(matches!(program.entry, Some(Expr::Constructor { .. })));
}

#[test]
fn parses_method_call_member_app() {
    let source = "obj.grow(1)";
    let program = parse_program(source).expect("parse program");
    match program.entry {
        Some(Expr::App { func, .. }) => {
            assert!(matches!(*func, Expr::Member { .. }));
        }
        _ => panic!("expected method call"),
    }
}
