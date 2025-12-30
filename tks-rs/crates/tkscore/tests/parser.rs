use tkscore::ast::{Expr, TopDecl};
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
