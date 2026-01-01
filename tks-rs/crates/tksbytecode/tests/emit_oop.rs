//! OOP / Record bytecode emission integration tests
//!
//! Tests the bytecode emitter for class-like constructs:
//! - Multi-field records (class instances)
//! - Record get/set sequences (member access patterns)
//! - Nested records (composition)
//! - Closures with record access (method-like patterns)

use tksbytecode::bytecode::{field_id, Opcode};
use tksbytecode::emit::emit;
use tkscore::ast::Literal;
use tksir::ir::{IRComp, IRTerm, IRVal};

/// Test emitting a record with multiple fields (class-like constructor pattern)
#[test]
fn emit_record_multi_field() {
    let term = IRTerm::Let(
        "obj".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec![
            ("x".to_string(), IRVal::Lit(Literal::Int(10))),
            ("y".to_string(), IRVal::Lit(Literal::Int(20))),
        ]))),
        Box::new(IRTerm::Return(IRVal::Var("obj".to_string()))),
    );

    let code = emit(&term).expect("emit multi-field record");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    // MakeRecord, then pairs of (PushInt, RecordSet) for each field, then Store, Load, Ret
    assert_eq!(
        opcodes,
        vec![
            Opcode::MakeRecord,
            Opcode::PushInt,
            Opcode::RecordSet,
            Opcode::PushInt,
            Opcode::RecordSet,
            Opcode::Store,
            Opcode::Load,
            Opcode::Ret
        ]
    );
}

/// Test emitting record get and set in sequence (simulates member access patterns)
#[test]
fn emit_record_get_then_set() {
    // let r = { x: 1 }; let v = r.x; r.x = 2
    let term = IRTerm::Let(
        "r".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec![(
            "x".to_string(),
            IRVal::Lit(Literal::Int(1)),
        )]))),
        Box::new(IRTerm::Let(
            "v".to_string(),
            Box::new(IRComp::Effect(Box::new(IRTerm::RecordGet(
                IRVal::Var("r".to_string()),
                "x".to_string(),
            )))),
            Box::new(IRTerm::RecordSet(
                IRVal::Var("r".to_string()),
                "x".to_string(),
                IRVal::Lit(Literal::Int(2)),
            )),
        )),
    );

    let code = emit(&term).expect("emit record get then set");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    assert_eq!(
        opcodes,
        vec![
            Opcode::MakeRecord,
            Opcode::PushInt,
            Opcode::RecordSet,
            Opcode::Store,       // store r
            Opcode::Load,        // load r for RecordGet
            Opcode::RecordGet,
            Opcode::Store,       // store v
            Opcode::Load,        // load r for RecordSet
            Opcode::PushInt,
            Opcode::RecordSet,
            Opcode::Ret
        ]
    );
}

/// Test emitting nested record access (simulates method access on self.field)
#[test]
fn emit_nested_record_access() {
    // let obj = { inner: { val: 42 } }; obj.inner
    let term = IRTerm::Let(
        "obj".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec![(
            "inner".to_string(),
            IRVal::Record(vec![("val".to_string(), IRVal::Lit(Literal::Int(42)))]),
        )]))),
        Box::new(IRTerm::RecordGet(
            IRVal::Var("obj".to_string()),
            "inner".to_string(),
        )),
    );

    let code = emit(&term).expect("emit nested record");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    // Outer MakeRecord, inner MakeRecord/PushInt/RecordSet, outer RecordSet, Store, Load, RecordGet
    assert_eq!(
        opcodes,
        vec![
            Opcode::MakeRecord,  // outer record
            Opcode::MakeRecord,  // inner record
            Opcode::PushInt,     // 42
            Opcode::RecordSet,   // inner.val = 42
            Opcode::RecordSet,   // outer.inner = inner
            Opcode::Store,       // store obj
            Opcode::Load,        // load obj
            Opcode::RecordGet,   // obj.inner
            Opcode::Ret
        ]
    );
}

/// Test emitting a closure that accesses a record field (method-like pattern)
#[test]
fn emit_closure_with_record_access() {
    // let obj = { x: 5 }; let getter = \self -> self.x; getter(obj)
    let term = IRTerm::Let(
        "obj".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec![(
            "x".to_string(),
            IRVal::Lit(Literal::Int(5)),
        )]))),
        Box::new(IRTerm::Let(
            "getter".to_string(),
            Box::new(IRComp::Pure(IRVal::Lam(
                "self".to_string(),
                Box::new(IRTerm::RecordGet(
                    IRVal::Var("self".to_string()),
                    "x".to_string(),
                )),
            ))),
            Box::new(IRTerm::App(
                IRVal::Var("getter".to_string()),
                IRVal::Var("obj".to_string()),
            )),
        )),
    );

    let code = emit(&term).expect("emit closure with record access");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    // Record creation, closure creation with jump, closure body with Load/RecordGet/Ret,
    // then call the closure
    assert!(opcodes.contains(&Opcode::MakeRecord));
    assert!(opcodes.contains(&Opcode::PushClosure));
    assert!(opcodes.contains(&Opcode::RecordGet));
    assert!(opcodes.contains(&Opcode::Call));
}

/// Test emitting record field assignment with field_id consistency
#[test]
fn emit_record_field_ids_match() {
    // Verify that field_id("x") in RecordSet matches field_id("x") in RecordGet
    let term = IRTerm::Let(
        "r".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec![(
            "counter".to_string(),
            IRVal::Lit(Literal::Int(0)),
        )]))),
        Box::new(IRTerm::RecordGet(
            IRVal::Var("r".to_string()),
            "counter".to_string(),
        )),
    );

    let code = emit(&term).expect("emit record with counter field");

    // Find the RecordSet and RecordGet instructions
    let set_instr = code.iter().find(|i| i.opcode == Opcode::RecordSet).unwrap();
    let get_instr = code.iter().find(|i| i.opcode == Opcode::RecordGet).unwrap();

    // Both should use the same field_id for "counter"
    assert_eq!(set_instr.operand1, Some(field_id("counter")));
    assert_eq!(get_instr.operand1, Some(field_id("counter")));
    assert_eq!(set_instr.operand1, get_instr.operand1);
}

/// Test emitting a setter closure (method with mutation pattern)
#[test]
fn emit_setter_closure() {
    // let setter = \self -> \val -> (self.x = val; self); setter
    let term = IRTerm::Let(
        "setter".to_string(),
        Box::new(IRComp::Pure(IRVal::Lam(
            "self".to_string(),
            Box::new(IRTerm::Return(IRVal::Lam(
                "val".to_string(),
                Box::new(IRTerm::RecordSet(
                    IRVal::Var("self".to_string()),
                    "x".to_string(),
                    IRVal::Var("val".to_string()),
                )),
            ))),
        ))),
        Box::new(IRTerm::Return(IRVal::Var("setter".to_string()))),
    );

    let code = emit(&term).expect("emit setter closure");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    // Should have nested closures and RecordSet
    assert!(opcodes.contains(&Opcode::PushClosure));
    assert!(opcodes.contains(&Opcode::RecordSet));
}

/// Test simulating a class-like constructor that creates a record with fields
#[test]
fn emit_class_constructor_pattern() {
    // Simulate: class Point { x: Int; y: Int }
    // Constructor: \x -> \y -> { x: x, y: y }
    let ctor = IRVal::Lam(
        "x".to_string(),
        Box::new(IRTerm::Return(IRVal::Lam(
            "y".to_string(),
            Box::new(IRTerm::Return(IRVal::Record(vec![
                ("x".to_string(), IRVal::Var("x".to_string())),
                ("y".to_string(), IRVal::Var("y".to_string())),
            ]))),
        ))),
    );

    // let Point = ctor; Point(10)(20)
    let term = IRTerm::Let(
        "Point".to_string(),
        Box::new(IRComp::Pure(ctor)),
        Box::new(IRTerm::Let(
            "_t0".to_string(),
            Box::new(IRComp::Effect(Box::new(IRTerm::App(
                IRVal::Var("Point".to_string()),
                IRVal::Lit(Literal::Int(10)),
            )))),
            Box::new(IRTerm::App(
                IRVal::Var("_t0".to_string()),
                IRVal::Lit(Literal::Int(20)),
            )),
        )),
    );

    let code = emit(&term).expect("emit class constructor pattern");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    // Should have closures, calls, and record construction
    assert!(opcodes.contains(&Opcode::PushClosure));
    assert!(opcodes.contains(&Opcode::Call));
    assert!(opcodes.contains(&Opcode::MakeRecord));
    assert!(opcodes.contains(&Opcode::RecordSet));
}


// ============================================================================
// TKS OOP Integration Tests (Agent B additions)
// ============================================================================

/// Test emitting full Counter class pattern with specifics, details, actions
#[test]
fn emit_counter_class_full_pattern() {
    let method_inc = IRVal::Lam(
        "self".to_string(),
        Box::new(IRTerm::Return(IRVal::Lam(
            "delta".to_string(),
            Box::new(IRTerm::Let(
                "_val".to_string(),
                Box::new(IRComp::Effect(Box::new(IRTerm::RecordGet(
                    IRVal::Var("self".to_string()),
                    "value".to_string(),
                )))),
                Box::new(IRTerm::IntOp(
                    tksir::ir::IntOp::Add,
                    IRVal::Var("_val".to_string()),
                    IRVal::Var("delta".to_string()),
                )),
            )),
        ))),
    );

    let term = IRTerm::Let(
        "c".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec\![
            ("value".to_string(), IRVal::Lit(Literal::Int(10))),
            ("doubled".to_string(), IRVal::Lit(Literal::Int(20))),
            ("inc".to_string(), method_inc),
        ]))),
        Box::new(IRTerm::RecordGet(
            IRVal::Var("c".to_string()),
            "value".to_string(),
        )),
    );

    let code = emit(&term).expect("emit counter class");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    assert\!(opcodes.contains(&Opcode::MakeRecord));
    assert\!(opcodes.contains(&Opcode::RecordSet));
    assert\!(opcodes.contains(&Opcode::RecordGet));
    assert\!(opcodes.contains(&Opcode::PushClosure));

    let record_sets: Vec<_> = code.iter()
        .filter(|i| i.opcode == Opcode::RecordSet)
        .collect();
    assert_eq\!(record_sets.len(), 3);
}

/// Test emitting method invocation pattern
#[test]
fn emit_method_invocation_pattern() {
    let getter = IRVal::Lam(
        "self".to_string(),
        Box::new(IRTerm::RecordGet(
            IRVal::Var("self".to_string()),
            "value".to_string(),
        )),
    );

    let term = IRTerm::Let(
        "obj".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec\![
            ("value".to_string(), IRVal::Lit(Literal::Int(5))),
            ("get".to_string(), getter),
        ]))),
        Box::new(IRTerm::Let(
            "_method".to_string(),
            Box::new(IRComp::Effect(Box::new(IRTerm::RecordGet(
                IRVal::Var("obj".to_string()),
                "get".to_string(),
            )))),
            Box::new(IRTerm::App(
                IRVal::Var("_method".to_string()),
                IRVal::Var("obj".to_string()),
            )),
        )),
    );

    let code = emit(&term).expect("emit method invocation");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    assert\!(opcodes.contains(&Opcode::MakeRecord));
    assert\!(opcodes.contains(&Opcode::RecordGet));
    assert\!(opcodes.contains(&Opcode::Call));
}

/// Test emitting method with argument: obj.inc(5)
#[test]
fn emit_method_with_argument() {
    let inc_method = IRVal::Lam(
        "self".to_string(),
        Box::new(IRTerm::Return(IRVal::Lam(
            "delta".to_string(),
            Box::new(IRTerm::Let(
                "_v".to_string(),
                Box::new(IRComp::Effect(Box::new(IRTerm::RecordGet(
                    IRVal::Var("self".to_string()),
                    "value".to_string(),
                )))),
                Box::new(IRTerm::IntOp(
                    tksir::ir::IntOp::Add,
                    IRVal::Var("_v".to_string()),
                    IRVal::Var("delta".to_string()),
                )),
            )),
        ))),
    );

    let term = IRTerm::Let(
        "obj".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec\![
            ("value".to_string(), IRVal::Lit(Literal::Int(10))),
            ("inc".to_string(), inc_method),
        ]))),
        Box::new(IRTerm::Let(
            "_method".to_string(),
            Box::new(IRComp::Effect(Box::new(IRTerm::RecordGet(
                IRVal::Var("obj".to_string()),
                "inc".to_string(),
            )))),
            Box::new(IRTerm::Let(
                "_bound".to_string(),
                Box::new(IRComp::Effect(Box::new(IRTerm::App(
                    IRVal::Var("_method".to_string()),
                    IRVal::Var("obj".to_string()),
                )))),
                Box::new(IRTerm::App(
                    IRVal::Var("_bound".to_string()),
                    IRVal::Lit(Literal::Int(5)),
                )),
            )),
        )),
    );

    let code = emit(&term).expect("emit method with argument");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    let call_count = opcodes.iter().filter(|&&op| op == Opcode::Call).count();
    assert_eq\!(call_count, 2);
    assert\!(opcodes.contains(&Opcode::Add));
}

/// Test emitting computed property pattern (details)
#[test]
fn emit_computed_property_pattern() {
    let term = IRTerm::Let(
        "value".to_string(),
        Box::new(IRComp::Pure(IRVal::Lit(Literal::Int(10)))),
        Box::new(IRTerm::Let(
            "doubled".to_string(),
            Box::new(IRComp::Effect(Box::new(IRTerm::IntOp(
                tksir::ir::IntOp::Mul,
                IRVal::Var("value".to_string()),
                IRVal::Lit(Literal::Int(2)),
            )))),
            Box::new(IRTerm::Return(IRVal::Record(vec\![
                ("value".to_string(), IRVal::Var("value".to_string())),
                ("doubled".to_string(), IRVal::Var("doubled".to_string())),
            ]))),
        )),
    );

    let code = emit(&term).expect("emit computed property");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    assert\!(opcodes.contains(&Opcode::Mul));
    assert\!(opcodes.contains(&Opcode::MakeRecord));
}

/// Test emitting record with method stored and later retrieved
#[test]
fn emit_record_method_storage_retrieval() {
    let method = IRVal::Lam(
        "x".to_string(),
        Box::new(IRTerm::IntOp(
            tksir::ir::IntOp::Add,
            IRVal::Var("x".to_string()),
            IRVal::Lit(Literal::Int(1)),
        )),
    );

    let term = IRTerm::Let(
        "obj".to_string(),
        Box::new(IRComp::Pure(IRVal::Record(vec\![
            ("method".to_string(), method),
        ]))),
        Box::new(IRTerm::Let(
            "m".to_string(),
            Box::new(IRComp::Effect(Box::new(IRTerm::RecordGet(
                IRVal::Var("obj".to_string()),
                "method".to_string(),
            )))),
            Box::new(IRTerm::App(
                IRVal::Var("m".to_string()),
                IRVal::Lit(Literal::Int(41)),
            )),
        )),
    );

    let code = emit(&term).expect("emit method storage retrieval");
    let opcodes: Vec<Opcode> = code.iter().map(|i| i.opcode).collect();

    assert\!(opcodes.contains(&Opcode::PushClosure));
    assert\!(opcodes.contains(&Opcode::RecordSet));
    assert\!(opcodes.contains(&Opcode::RecordGet));
    assert\!(opcodes.contains(&Opcode::Call));
}
