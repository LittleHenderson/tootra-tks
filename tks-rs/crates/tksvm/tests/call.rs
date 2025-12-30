use tksbytecode::bytecode::{Instruction, Opcode};
use tksbytecode::extern_id;
use tksvm::vm::{Value, VmError, VmState};

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

#[test]
fn call_extern_with_partial_application() {
    let code = vec![
        inst1(Opcode::PushInt, 10),
        inst(Opcode::Call),
        inst1(Opcode::PushInt, 20),
        inst(Opcode::Call),
        inst(Opcode::Ret),
    ];

    let mut vm = VmState::new(code);
    vm.register_extern("add2", |mut args| {
        if args.len() != 2 {
            return Err(VmError::TypeMismatch {
                expected: "two arguments",
                found: Value::Unit,
            });
        }
        let right = match args.pop().unwrap() {
            Value::Int(value) => value,
            other => {
                return Err(VmError::TypeMismatch {
                    expected: "int",
                    found: other,
                })
            }
        };
        let left = match args.pop().unwrap() {
            Value::Int(value) => value,
            other => {
                return Err(VmError::TypeMismatch {
                    expected: "int",
                    found: other,
                })
            }
        };
        Ok(Value::Int(left + right))
    });

    vm.stack.push(Value::Extern {
        id: extern_id("add2"),
        arity: 2,
        args: Vec::new(),
    });

    let result = vm.run().expect("vm run");
    assert_eq!(result, Value::Int(30));
}
