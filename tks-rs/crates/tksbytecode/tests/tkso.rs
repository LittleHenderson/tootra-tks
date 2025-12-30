use tksbytecode::bytecode::{Instruction, Opcode};
use tksbytecode::tkso::{decode, encode};

#[test]
fn tkso_round_trip() {
    let code = vec![
        Instruction {
            opcode: Opcode::PushInt,
            flags: 0,
            operand1: Some(42),
            operand2: None,
        },
        Instruction {
            opcode: Opcode::PushBool,
            flags: 7,
            operand1: Some(1),
            operand2: None,
        },
        Instruction {
            opcode: Opcode::Jmp,
            flags: 0,
            operand1: Some(2),
            operand2: None,
        },
        Instruction {
            opcode: Opcode::PushElement,
            flags: 0,
            operand1: Some(2),
            operand2: Some(5),
        },
        Instruction {
            opcode: Opcode::Ret,
            flags: 0,
            operand1: None,
            operand2: None,
        },
    ];

    let bytes = encode(&code).expect("encode tkso");
    let decoded = decode(&bytes).expect("decode tkso");
    assert_eq!(decoded, code);
}

#[test]
fn tkso_round_trip_empty() {
    let code: Vec<Instruction> = Vec::new();
    let bytes = encode(&code).expect("encode tkso");
    let decoded = decode(&bytes).expect("decode tkso");
    assert_eq!(decoded, code);
}
