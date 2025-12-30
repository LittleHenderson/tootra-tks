use tksbytecode::bytecode::{Instruction, Opcode};
use tksvm::vm::{OrdinalValue, Value, VmError, VmState};

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
fn run_apply_noetic_program() {
    let code = vec![
        instr(Opcode::PushNoetic, Some(4)),
        instr(Opcode::PushInt, Some(9)),
        instr(Opcode::ApplyNoetic, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::NoeticApplied {
            index: 4,
            value: Box::new(Value::Int(9))
        }
    );
}

#[test]
fn run_rpm_acquire_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(7)),
        instr(Opcode::RpmAcquire, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::RpmState {
            ok: true,
            value: Box::new(Value::Int(7))
        }
    );
}

#[test]
fn run_rpm_fail_program() {
    let code = vec![instr(Opcode::RpmFail, None), instr(Opcode::Ret, None)];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::RpmState {
            ok: false,
            value: Box::new(Value::Unit)
        }
    );
}

#[test]
fn run_rpm_check_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::RpmAcquire, None),
        instr(Opcode::RpmCheck, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn run_rpm_return_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(5)),
        instr(Opcode::RpmReturn, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::RpmState {
            ok: true,
            value: Box::new(Value::Int(5))
        }
    );
}

#[test]
fn run_rpm_is_success_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::RpmReturn, None),
        instr(Opcode::RpmIsSuccess, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Bool(true));
}

#[test]
fn run_rpm_unwrap_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(7)),
        instr(Opcode::RpmReturn, None),
        instr(Opcode::RpmUnwrap, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(7));
}

#[test]
fn run_rpm_unwrap_fail_program() {
    let code = vec![instr(Opcode::RpmFail, None), instr(Opcode::RpmUnwrap, None)];

    let mut vm = VmState::new(code);
    let err = vm.run().expect_err("unwrap should fail");
    assert!(matches!(err, VmError::RpmUnwrapFailed));
}

#[test]
fn run_rpm_bind_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(5)),
        instr(Opcode::RpmReturn, None),
        instr(Opcode::PushClosure, Some(4)),
        instr(Opcode::Jmp, Some(6)),
        instr(Opcode::Load, Some(0)),
        instr(Opcode::Ret, None),
        instr(Opcode::RpmBind, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::RpmState {
            ok: true,
            value: Box::new(Value::Int(5))
        }
    );
}

#[test]
fn run_superpose_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PushInt, Some(10)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::PushInt, Some(20)),
        instr(Opcode::Superpose, Some(2)),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::Superpose(vec![
            (Value::Int(1), Value::Int(10)),
            (Value::Int(2), Value::Int(20)),
        ])
    );
}

#[test]
fn run_measure_superpose_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PushInt, Some(10)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::PushInt, Some(20)),
        instr(Opcode::Superpose, Some(2)),
        instr(Opcode::Measure, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(10));
}

#[test]
fn run_entangle_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(4)),
        instr(Opcode::PushInt, Some(5)),
        instr(Opcode::Entangle, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::Entangle(Box::new(Value::Int(4)), Box::new(Value::Int(5)))
    );
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

#[test]
fn run_perform_effect_unhandled() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PerformEffect, Some(0)),
    ];

    let mut vm = VmState::new(code);
    let err = vm.run().expect_err("expected unhandled effect");
    assert_eq!(err, VmError::UnhandledEffect);
}

#[test]
fn run_handle_perform_with_resume() {
    let code = vec![
        instr(Opcode::PushClosure, Some(2)),
        instr(Opcode::Jmp, Some(4)),
        instr(Opcode::Load, Some(0)),
        instr(Opcode::Ret, None),
        instr(Opcode::PushInt, Some(0)),
        instr(Opcode::PushClosure, Some(7)),
        instr(Opcode::Jmp, Some(11)),
        instr(Opcode::Load, Some(1)),
        instr(Opcode::Load, Some(0)),
        instr(Opcode::Call, None),
        instr(Opcode::Ret, None),
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::InstallHandler, None),
        instr(Opcode::PushInt, Some(7)),
        instr(Opcode::PerformEffect, Some(0)),
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::Add, None),
        instr(Opcode::HandlerReturn, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(8));
}
