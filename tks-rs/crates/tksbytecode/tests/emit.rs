use tksbytecode::bytecode::{Instruction, Opcode};
use tksbytecode::emit::emit;
use tkscore::ast::Literal;
use tksir::ir::{IRComp, IRTerm, IRVal};

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
