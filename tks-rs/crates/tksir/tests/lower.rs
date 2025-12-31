use tkscore::ast::Literal;
use tkscore::parser::parse_program;
use tksir::ir::{IRComp, IRTerm, IRVal};
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

#[test]
fn lower_rpm_return_with_binding() {
    let source = r#"
let f = \x -> x;
return (f(1))
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
                        IRTerm::RpmReturn(IRVal::Var(var)) => {
                            assert_eq!(var, tmp);
                        }
                        other => panic!("expected rpm return, got {other:?}"),
                    }
                }
                other => panic!("expected temp let, got {other:?}"),
            }
        }
        other => panic!("expected top-level let, got {other:?}"),
    }
}

#[test]
fn lower_rpm_bind_with_bindings() {
    let source = r#"
let f = \x -> x;
let g = \y -> y;
f(1) >>= g(2)
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(name, _, body) => {
            assert_eq!(name, "f");
            match *body {
                IRTerm::Let(name, _, body) => {
                    assert_eq!(name, "g");
                    match *body {
                        IRTerm::Let(t0, _, body) => {
                            assert_eq!(t0, "_t0");
                            match *body {
                                IRTerm::Let(t1, _, body) => {
                                    assert_eq!(t1, "_t1");
                                    match *body {
                                        IRTerm::RpmBind(IRVal::Var(left), IRVal::Var(right)) => {
                                            assert_eq!(left, t0);
                                            assert_eq!(right, t1);
                                        }
                                        other => panic!("expected rpm bind, got {other:?}"),
                                    }
                                }
                                other => panic!("expected temp let, got {other:?}"),
                            }
                        }
                        other => panic!("expected temp let, got {other:?}"),
                    }
                }
                other => panic!("expected let g, got {other:?}"),
            }
        }
        other => panic!("expected top-level let, got {other:?}"),
    }
}

#[test]
fn lower_class_decl_and_constructor() {
    let source = r#"
class Counter {
  field value: Int;
  property current: Int = 0;
  method inc(delta: Int): Int = delta;
}
repeat Counter(value: 1)
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(name, IRComp::Pure(_), body) => {
            assert_eq!(name, "Counter");
            match *body {
                IRTerm::Let(prop_name, IRComp::Pure(_), body) => {
                    assert_eq!(prop_name, "Counter::current");
                    match *body {
                        IRTerm::Let(method_name, IRComp::Pure(_), body) => {
                            assert_eq!(method_name, "Counter::inc");
                            match *body {
                                IRTerm::App(IRVal::Var(func), IRVal::Lit(Literal::Int(value))) => {
                                    assert_eq!(func, "Counter");
                                    assert_eq!(value, 1);
                                }
                                other => panic!("expected constructor app, got {other:?}"),
                            }
                        }
                        other => panic!("expected method binding, got {other:?}"),
                    }
                }
                other => panic!("expected property binding, got {other:?}"),
            }
        }
        other => panic!("expected constructor binding, got {other:?}"),
    }
}

#[test]
fn lower_extern_decl() {
    let source = r#"
external c fn ping(x: Int): Int;
ping(1)
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Let(name, comp, body) => {
            assert_eq!(name, "ping");
            assert_eq!(
                *comp,
                IRComp::Pure(IRVal::Extern("ping".to_string(), 1))
            );
            assert_eq!(
                *body,
                IRTerm::App(
                    IRVal::Var("ping".to_string()),
                    IRVal::Lit(Literal::Int(1))
                )
            );
        }
        other => panic!("expected extern let, got {other:?}"),
    }
}

#[test]
fn lower_resume_in_handler_clause() {
    let source = r#"
handle perform log(1) with {
  return x -> x;
  log(msg) k -> resume(msg);
}
"#;
    let program = parse_program(source).expect("parse program");
    let term = lower_program(&program).expect("lower program");

    match term {
        IRTerm::Handle(_, handler) => {
            assert_eq!(handler.op_clauses.len(), 1);
            let clause = &handler.op_clauses[0];
            assert_eq!(clause.op, "log");
            assert_eq!(clause.k, "k");
            match clause.body.as_ref() {
                IRTerm::Let(name, comp, body) => {
                    assert_eq!(name, "resume");
                    assert_eq!(**comp, IRComp::Pure(IRVal::Var("k".to_string())));
                    assert_eq!(
                        **body,
                        IRTerm::App(
                            IRVal::Var("resume".to_string()),
                            IRVal::Var("msg".to_string())
                        )
                    );
                }
                other => panic!("expected resume let, got {other:?}"),
            }
        }
        other => panic!("expected handle, got {other:?}"),
    }
}
