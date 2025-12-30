use tksbytecode::bytecode::{extern_id, Instruction, Opcode};
use tksbytecode::emit::emit;
use tkscore::ast::{Expr, Literal, OrdinalLiteral};
use tkscore::span::Span;
use tksir::ir::{IRComp, IRHandler, IRTerm, IRVal, OrdOp};

fn inst(opcode: Opcode) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: None,
        operand2: None,
    }
}

fn inst1(opcode: Opcode, operand1: u64) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: Some(operand1),
        operand2: None,
    }
}

fn inst2(opcode: Opcode, operand1: u64, operand2: u64) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: Some(operand1),
        operand2: Some(operand2),
    }
}

fn span() -> Span {
    Span::new(0, 0, 0, 0)
}

#[test]
fn emit_let_int_load() {
    let term = IRTerm::Let(
        "x".to_string(),
        Box::new(IRComp::Pure(IRVal::Lit(Literal::Int(1)))),
        Box::new(IRTerm::Return(IRVal::Var("x".to_string()))),
    );

    let instructions = emit(&term).expect("emit bytecode");
    let opcodes: Vec<Opcode> = instructions.iter().map(|inst| inst.opcode).collect();

    assert_eq!(
        opcodes,
        vec![Opcode::PushInt, Opcode::Store, Opcode::Load, Opcode::Ret]
    );
}

#[test]
fn emit_if_jumps() {
    let term = IRTerm::If(
        IRVal::Lit(Literal::Bool(true)),
        Box::new(IRTerm::Return(IRVal::Lit(Literal::Int(1)))),
        Box::new(IRTerm::Return(IRVal::Lit(Literal::Int(2)))),
    );

    let instructions = emit(&term).expect("emit bytecode");
    let opcodes: Vec<Opcode> = instructions.iter().map(|inst| inst.opcode).collect();

    assert_eq!(
        opcodes,
        vec![
            Opcode::PushBool,
            Opcode::JmpUnless,
            Opcode::PushInt,
            Opcode::Jmp,
            Opcode::PushInt,
            Opcode::Ret
        ]
    );
    assert_eq!(instructions[1].operand1, Some(4));
    assert_eq!(instructions[3].operand1, Some(5));
}

#[test]
fn emit_noetic_apply() {
    let term = IRTerm::App(
        IRVal::Noetic(3),
        IRVal::Lit(Literal::Int(7)),
    );

    let instructions = emit(&term).expect("emit bytecode");
    let opcodes: Vec<Opcode> = instructions.iter().map(|inst| inst.opcode).collect();

    assert_eq!(
        opcodes,
        vec![Opcode::PushNoetic, Opcode::PushInt, Opcode::ApplyNoetic, Opcode::Ret]
    );
    assert_eq!(instructions[0].operand1, Some(3));
}

#[test]
fn emit_app_with_lambda() {
    let term = IRTerm::App(
        IRVal::Lam(
            "x".to_string(),
            Box::new(IRTerm::Return(IRVal::Var("x".to_string()))),
        ),
        IRVal::Lit(Literal::Int(7)),
    );

    let instructions = emit(&term).expect("emit bytecode");
    let opcodes: Vec<Opcode> = instructions.iter().map(|inst| inst.opcode).collect();

    assert_eq!(
        opcodes,
        vec![
            Opcode::PushClosure,
            Opcode::Jmp,
            Opcode::Load,
            Opcode::Ret,
            Opcode::PushInt,
            Opcode::Call,
            Opcode::Ret
        ]
    );
    assert_eq!(instructions[0].operand1, Some(2));
    assert_eq!(instructions[1].operand1, Some(4));
    assert_eq!(instructions[4].operand1, Some(7));
}

#[test]
fn emit_noetic_value() {
    let term = IRTerm::Return(IRVal::Noetic(5));
    let code = emit(&term).expect("emit noetic");
    let expected = vec![inst1(Opcode::PushNoetic, 5), inst(Opcode::Ret)];
    assert_eq!(code, expected);
}

#[test]
fn emit_extern_call() {
    let term = IRTerm::App(
        IRVal::Extern("ping".to_string(), 1),
        IRVal::Lit(Literal::Int(7)),
    );
    let code = emit(&term).expect("emit extern call");
    let expected = vec![
        inst2(Opcode::PushExtern, extern_id("ping"), 1),
        inst1(Opcode::PushInt, 7),
        inst(Opcode::Call),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_ket_value() {
    let term = IRTerm::Return(IRVal::Ket(Box::new(IRVal::Lit(Literal::Int(9)))));
    let code = emit(&term).expect("emit ket");
    let expected = vec![
        inst1(Opcode::PushInt, 9),
        inst(Opcode::MakeKet),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_superpose_term() {
    let term = IRTerm::Superpose(vec![
        (IRVal::Lit(Literal::Int(1)), IRVal::Lit(Literal::Int(10))),
        (IRVal::Lit(Literal::Int(2)), IRVal::Lit(Literal::Int(20))),
    ]);

    let instructions = emit(&term).expect("emit superpose");
    let opcodes: Vec<Opcode> = instructions.iter().map(|inst| inst.opcode).collect();

    assert_eq!(
        opcodes,
        vec![
            Opcode::PushInt,
            Opcode::PushInt,
            Opcode::PushInt,
            Opcode::PushInt,
            Opcode::Superpose,
            Opcode::Ret
        ]
    );
    assert_eq!(instructions[4].operand1, Some(2));
}

#[test]
fn emit_measure_term() {
    let term = IRTerm::Measure(IRVal::Lit(Literal::Int(3)));
    let code = emit(&term).expect("emit measure");
    let expected = vec![inst1(Opcode::PushInt, 3), inst(Opcode::Measure), inst(Opcode::Ret)];
    assert_eq!(code, expected);
}

#[test]
fn emit_entangle_term() {
    let term = IRTerm::Entangle(IRVal::Lit(Literal::Int(4)), IRVal::Lit(Literal::Int(5)));
    let code = emit(&term).expect("emit entangle");
    let expected = vec![
        inst1(Opcode::PushInt, 4),
        inst1(Opcode::PushInt, 5),
        inst(Opcode::Entangle),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_perform_term() {
    let term = IRTerm::Perform("ping".to_string(), IRVal::Lit(Literal::Int(6)));
    let code = emit(&term).expect("emit perform");
    let expected = vec![
        inst1(Opcode::PushInt, 6),
        inst1(Opcode::PerformEffect, 0),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_handle_term() {
    let handler = IRHandler {
        return_clause: (
            "r".to_string(),
            Box::new(IRTerm::Return(IRVal::Lit(Literal::Unit))),
        ),
        op_clauses: Vec::new(),
    };
    let term = IRTerm::Handle(
        Box::new(IRTerm::Return(IRVal::Lit(Literal::Int(2)))),
        Box::new(handler),
    );
    let code = emit(&term).expect("emit handle");
    let expected = vec![
        inst1(Opcode::PushClosure, 2),
        inst1(Opcode::Jmp, 4),
        inst(Opcode::PushUnit),
        inst(Opcode::Ret),
        inst1(Opcode::PushInt, 0),
        inst1(Opcode::InstallHandler, 8),
        inst1(Opcode::PushInt, 2),
        inst(Opcode::HandlerReturn),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_rpm_check_term() {
    let term = IRTerm::RPMCheck(IRVal::Lit(Literal::Int(1)));
    let code = emit(&term).expect("emit rpm check");
    let expected = vec![
        inst1(Opcode::PushInt, 1),
        inst(Opcode::RpmCheck),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_rpm_acquire_term() {
    let term = IRTerm::RPMAcquire(IRVal::Lit(Literal::Int(2)));
    let code = emit(&term).expect("emit rpm acquire");
    let expected = vec![
        inst1(Opcode::PushInt, 2),
        inst(Opcode::RpmAcquire),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_rpm_fail_term() {
    let term = IRTerm::RPMFail;
    let code = emit(&term).expect("emit rpm fail");
    let expected = vec![inst(Opcode::RpmFail), inst(Opcode::Ret)];
    assert_eq!(code, expected);
}

#[test]
fn emit_rpm_return_term() {
    let term = IRTerm::RpmReturn(IRVal::Lit(Literal::Int(3)));
    let code = emit(&term).expect("emit rpm return");
    let expected = vec![
        inst1(Opcode::PushInt, 3),
        inst(Opcode::RpmReturn),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_rpm_bind_term() {
    let term = IRTerm::RpmBind(IRVal::Lit(Literal::Int(1)), IRVal::Lit(Literal::Int(2)));
    let code = emit(&term).expect("emit rpm bind");
    let expected = vec![
        inst1(Opcode::PushInt, 1),
        inst1(Opcode::PushInt, 2),
        inst(Opcode::RpmBind),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_ordinal_finite_literal() {
    let expr = Expr::OrdLit {
        span: span(),
        value: OrdinalLiteral::Finite(3),
    };
    let term = IRTerm::Return(IRVal::Ordinal(expr));
    let code = emit(&term).expect("emit ordinal");
    let expected = vec![inst1(Opcode::PushOrd, 3), inst(Opcode::Ret)];
    assert_eq!(code, expected);
}

#[test]
fn emit_ordinal_omega_plus_literal() {
    let expr = Expr::OrdLit {
        span: span(),
        value: OrdinalLiteral::OmegaPlus(2),
    };
    let term = IRTerm::Return(IRVal::Ordinal(expr));
    let code = emit(&term).expect("emit ordinal");
    let expected = vec![
        inst(Opcode::PushOmega),
        inst1(Opcode::PushOrd, 2),
        inst(Opcode::OrdAdd),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}

#[test]
fn emit_ordinal_succ_term() {
    let expr = Expr::OrdLit {
        span: span(),
        value: OrdinalLiteral::Finite(1),
    };
    let term = IRTerm::OrdSucc(IRVal::Ordinal(expr));
    let code = emit(&term).expect("emit ordinal");
    let expected = vec![inst1(Opcode::PushOrd, 1), inst(Opcode::OrdSucc), inst(Opcode::Ret)];
    assert_eq!(code, expected);
}

#[test]
fn emit_ordinal_op_mul_term() {
    let left = IRVal::Ordinal(Expr::OrdOmega { span: span() });
    let right = IRVal::Ordinal(Expr::OrdLit {
        span: span(),
        value: OrdinalLiteral::Finite(2),
    });
    let term = IRTerm::OrdOp(OrdOp::Mul, left, right);
    let code = emit(&term).expect("emit ordinal");
    let expected = vec![
        inst(Opcode::PushOmega),
        inst1(Opcode::PushOrd, 2),
        inst(Opcode::OrdMul),
        inst(Opcode::Ret),
    ];
    assert_eq!(code, expected);
}
