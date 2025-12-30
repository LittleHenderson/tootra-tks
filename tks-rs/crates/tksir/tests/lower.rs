use tkscore::ast::Literal;
use tkscore::parser::parse_program;
use tksir::ir::{IRTerm, IRVal};
use tksir::lower::lower_program;

#[test]
fn lower_simple_program() {
    let source = r#"
let x = 1;
let f = \y -> y;
f(x)
"#;
    let program = parse_program(source).expect("parse program");
    lower_program(&program).expect("lower program");
}

#[test]
fn lower_ket_with_non_value_inner() {
    let source = r#"
let f = \x -> x;
|f(1)>
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(name, _, body) => {
            assert_eq!(name, "f");
            match *body {
                IRTerm::Let(tmp, _, inner) => {
                    assert_eq!(tmp, "_t0");
                    match *inner {
                        IRTerm::Return(IRVal::Ket(inner_val)) => {
                            assert_eq!(*inner_val, IRVal::Var(tmp));
                        }
                        other => panic!("expected ket return, got {other:?}"),
                    }
                }
                other => panic!("expected temp let, got {other:?}"),
            }
        }
        other => panic!("expected top-level let, got {other:?}"),
    }
}

#[test]
fn lower_superpose_with_ket_bindings() {
    let source = r#"
let f = \x -> x;
superpose { f(0): |f(1)>, 1: |2> }
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(name, _, body) => {
            assert_eq!(name, "f");
            match *body {
                IRTerm::Let(t0, _, body) => {
                    assert_eq!(t0, "_t0");
                    match *body {
                        IRTerm::Let(t1, _, body) => {
                            assert_eq!(t1, "_t1");
                            match *body {
                                IRTerm::Superpose(states) => {
                                    assert_eq!(states.len(), 2);
                                    assert_eq!(states[0].0, IRVal::Var(t0.clone()));
                                    assert_eq!(
                                        states[0].1,
                                        IRVal::Ket(Box::new(IRVal::Var(t1.clone())))
                                    );
                                    assert_eq!(states[1].0, IRVal::Lit(Literal::Int(1)));
                                    assert_eq!(
                                        states[1].1,
                                        IRVal::Ket(Box::new(IRVal::Lit(Literal::Int(2))))
                                    );
                                }
                                other => panic!("expected superpose, got {other:?}"),
                            }
                        }
                        other => panic!("expected temp let, got {other:?}"),
                    }
                }
                other => panic!("expected temp let, got {other:?}"),
            }
        }
        other => panic!("expected top-level let, got {other:?}"),
    }
}

#[test]
fn lower_measure_with_binding() {
    let source = r#"
let f = \x -> x;
measure(f(1))
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(_, _, body) => match *body {
            IRTerm::Let(tmp, _, inner) => match *inner {
                IRTerm::Measure(IRVal::Var(var)) => {
                    assert_eq!(var, tmp);
                }
                other => panic!("expected measure, got {other:?}"),
            },
            other => panic!("expected temp let, got {other:?}"),
        },
        other => panic!("expected top-level let, got {other:?}"),
    }
}

#[test]
fn lower_entangle_with_binding() {
    let source = r#"
let f = \x -> x;
entangle(f(1), f(2))
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(_, _, body) => match *body {
            IRTerm::Let(t0, _, body) => match *body {
                IRTerm::Let(t1, _, body) => match *body {
                    IRTerm::Entangle(IRVal::Var(left), IRVal::Var(right)) => {
                        assert_eq!(left, t0);
                        assert_eq!(right, t1);
                    }
                    other => panic!("expected entangle, got {other:?}"),
                },
                other => panic!("expected temp let, got {other:?}"),
            },
            other => panic!("expected temp let, got {other:?}"),
        },
        other => panic!("expected top-level let, got {other:?}"),
    }
}
