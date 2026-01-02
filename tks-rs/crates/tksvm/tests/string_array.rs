use tksbytecode::bytecode::{Instruction, Opcode};
use tksvm::vm::{Value, VmError, VmState};

fn instr(opcode: Opcode, operand1: Option<u64>) -> Instruction {
    Instruction {
        opcode,
        flags: 0,
        operand1,
        operand2: None,
    }
}

// String opcode tests

#[test]
fn run_push_str_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("hello".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Str("hello".to_string()));
}

#[test]
fn run_str_concat_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::PushStr, Some(1)),
        instr(Opcode::StrConcat, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("hello".to_string());
    vm.strings.push(" world".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Str("hello world".to_string()));
}

#[test]
fn run_str_concat_empty_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::PushStr, Some(1)),
        instr(Opcode::StrConcat, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("test".to_string());
    vm.strings.push("".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Str("test".to_string()));
}

#[test]
fn run_str_len_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::StrLen, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("hello".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(5));
}

#[test]
fn run_str_len_empty_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::StrLen, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(0));
}

#[test]
fn run_str_concat_multiple_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::PushStr, Some(1)),
        instr(Opcode::StrConcat, None),
        instr(Opcode::PushStr, Some(2)),
        instr(Opcode::StrConcat, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("a".to_string());
    vm.strings.push("b".to_string());
    vm.strings.push("c".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Str("abc".to_string()));
}

#[test]
fn run_str_len_type_mismatch() {
    let code = vec![
        instr(Opcode::PushInt, Some(42)),
        instr(Opcode::StrLen, None),
    ];

    let mut vm = VmState::new(code);
    let err = vm.run().expect_err("expected type mismatch");
    assert!(matches!(err, VmError::TypeMismatch { expected: "string", .. }));
}

// Array opcode tests

#[test]
fn run_make_array_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::PushInt, Some(3)),
        instr(Opcode::MakeArray, Some(3)),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(
        result,
        Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
}

#[test]
fn run_make_empty_array_program() {
    let code = vec![
        instr(Opcode::MakeArray, Some(0)),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Array(vec![]));
}

#[test]
fn run_array_get_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(10)),
        instr(Opcode::PushInt, Some(20)),
        instr(Opcode::PushInt, Some(30)),
        instr(Opcode::MakeArray, Some(3)),
        instr(Opcode::PushInt, Some(1)), // index
        instr(Opcode::ArrayGet, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(20));
}

#[test]
fn run_array_get_first_element() {
    let code = vec![
        instr(Opcode::PushInt, Some(100)),
        instr(Opcode::PushInt, Some(200)),
        instr(Opcode::MakeArray, Some(2)),
        instr(Opcode::PushInt, Some(0)), // index
        instr(Opcode::ArrayGet, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(100));
}

#[test]
fn run_array_set_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::PushInt, Some(3)),
        instr(Opcode::MakeArray, Some(3)),
        instr(Opcode::PushInt, Some(1)),  // index
        instr(Opcode::PushInt, Some(99)), // new value
        instr(Opcode::ArraySet, None),
        instr(Opcode::PushInt, Some(1)),  // index
        instr(Opcode::ArrayGet, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(99));
}

#[test]
fn run_array_len_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::PushInt, Some(3)),
        instr(Opcode::PushInt, Some(4)),
        instr(Opcode::MakeArray, Some(4)),
        instr(Opcode::ArrayLen, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(4));
}

#[test]
fn run_array_len_empty_program() {
    let code = vec![
        instr(Opcode::MakeArray, Some(0)),
        instr(Opcode::ArrayLen, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(0));
}

#[test]
fn run_array_push_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::PushInt, Some(2)),
        instr(Opcode::MakeArray, Some(2)),
        instr(Opcode::PushInt, Some(3)),
        instr(Opcode::ArrayPush, None),
        instr(Opcode::ArrayLen, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(3));
}

#[test]
fn run_array_push_get_program() {
    let code = vec![
        instr(Opcode::PushInt, Some(10)),
        instr(Opcode::MakeArray, Some(1)),
        instr(Opcode::PushInt, Some(20)),
        instr(Opcode::ArrayPush, None),
        instr(Opcode::PushInt, Some(1)), // index
        instr(Opcode::ArrayGet, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Int(20));
}

#[test]
fn run_array_of_strings_program() {
    let code = vec![
        instr(Opcode::PushStr, Some(0)),
        instr(Opcode::PushStr, Some(1)),
        instr(Opcode::MakeArray, Some(2)),
        instr(Opcode::PushInt, Some(0)), // index
        instr(Opcode::ArrayGet, None),
        instr(Opcode::Ret, None),
    ];

    let mut vm = VmState::new(code);
    vm.strings.push("first".to_string());
    vm.strings.push("second".to_string());
    let result = vm.run().expect("run program");
    assert_eq!(result, Value::Str("first".to_string()));
}

#[test]
fn run_array_get_out_of_bounds() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::MakeArray, Some(1)),
        instr(Opcode::PushInt, Some(5)), // out of bounds index
        instr(Opcode::ArrayGet, None),
    ];

    let mut vm = VmState::new(code);
    let err = vm.run().expect_err("expected out of bounds");
    assert!(matches!(err, VmError::InvalidOperand { opcode: Opcode::ArrayGet, .. }));
}

#[test]
fn run_array_set_out_of_bounds() {
    let code = vec![
        instr(Opcode::PushInt, Some(1)),
        instr(Opcode::MakeArray, Some(1)),
        instr(Opcode::PushInt, Some(5)),  // out of bounds index
        instr(Opcode::PushInt, Some(99)), // value
        instr(Opcode::ArraySet, None),
    ];

    let mut vm = VmState::new(code);
    let err = vm.run().expect_err("expected out of bounds");
    assert!(matches!(err, VmError::InvalidOperand { opcode: Opcode::ArraySet, .. }));
}

#[test]
fn run_array_len_type_mismatch() {
    let code = vec![
        instr(Opcode::PushInt, Some(42)),
        instr(Opcode::ArrayLen, None),
    ];

    let mut vm = VmState::new(code);
    let err = vm.run().expect_err("expected type mismatch");
    assert!(matches!(err, VmError::TypeMismatch { expected: "array", .. }));
}
