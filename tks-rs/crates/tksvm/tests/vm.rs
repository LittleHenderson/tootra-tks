use tksbytecode::bytecode::{Instruction, Opcode};
use tksvm::vm::{OrdinalValue, Value, VmState};

fn instr(opcode: Opcode, operand1: Option<u64>) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1,
        operand2: None,
    }
}

fn instr2(opcode: Opcode, operand1: u64, operand2: u64) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1: Some(operand1),
        operand2: Some(operand2),
    }
}

#[test]
fn run_add_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::PushInt, Some(3)),
        instr(Opcode::Add, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn run_load_store_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::Store, Some(0)),
        instr(Opcode::PushInt, Some(3)),
        instr(Opcode::Load, Some(0)),
        instr(Opcode::Add, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn run_jump_unless_program() {
    let code = vec![
        instr(Opcode::PushBool, Some(0)),
        instr(Opcode::JmpUnless, Some(4)),
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::Jmp, Some(5)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(2));
}

#[test]
fn run_push_element_program() {
    let code = vec![instr2(Opcode::PushElement, 1, 7), instr(Opcode::Ret, None)];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Element { world: 1, index: 7 });
}

#[test]
fn run_push_noetic_program() {
    let code = vec![instr(Opcode::PushNoetic, Some(4)), instr(Opcode::Ret, None)];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Noetic(4));
}

#[test]
fn run_make_ket_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(9)),
        instr(Opcode::MakeKet, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Ket(Box::new(Value::Int(9))));
}

#[test]
fn run_ord_succ_program() {
    let code = vec![
        instr(Opcode::PushOrd, Some(1)),
        instr(Opcode::OrdSucc, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::Ordinal(OrdinalValue::Succ(Box::new(OrdinalValue::Finite(1))))
    );
}

#[test]
fn run_ord_add_program() {
    let code = vec![
        instr(Opcode::PushOmega, None),
        instr(Opcode::PushOrd, Some(2)),
        instr(Opcode::OrdAdd, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::Ordinal(OrdinalValue::Add(
            Box::new(OrdinalValue::Omega),
            Box::new(OrdinalValue::Finite(2))
        ))
    );
}
