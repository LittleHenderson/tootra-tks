use tksbytecode::bytecode::{Instruction, Opcode};
use tksvm::vm::{Value, VmState};

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
fn call_returns_argument() {
    let code = vec![
        inst1(Opcode::PushClosure, 2),
        inst1(Opcode::Jmp, 4),
        inst1(Opcode::Load, 0),
        inst(Opcode::Ret),
        inst1(Opcode::PushInt, 21),
        inst(Opcode::Call),
        inst(Opcode::Ret),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("vm run");
    assert_eq!(result, Value::Int(21));
}

#[test]
fn locals_are_per_frame() {
    let code = vec![
        inst1(Opcode::PushInt, 7),
        inst1(Opcode::Store, 0),
        inst1(Opcode::PushClosure, 4),
        inst1(Opcode::Jmp, 6),
        inst1(Opcode::Load, 0),
        inst(Opcode::Ret),
        inst1(Opcode::PushInt, 1),
        inst(Opcode::Call),
        inst1(Opcode::Load, 0),
        inst(Opcode::Ret),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("vm run");
    assert_eq!(result, Value::Int(7));
}
