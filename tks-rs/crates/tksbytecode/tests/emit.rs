use tksbytecode::bytecode::Opcode;
use tksbytecode::emit::emit;
use tkscore::ast::Literal;
use tksir::ir::{IRComp, IRTerm, IRVal};

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
