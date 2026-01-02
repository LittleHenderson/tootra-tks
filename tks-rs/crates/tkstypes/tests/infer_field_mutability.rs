//! Tests for field mutability enforcement in TKS OOP.
//!
//! CURRENT STATUS: Field mutability is parsed but NOT enforced.
//!
//! The `FieldDecl` in `ast.rs` has a `mutable: bool` field that is set by
//! the parser when it sees the `mut` keyword. However, this flag is currently
//! ignored during:
//! - Type checking (infer.rs): `ClassInfo.fields` only stores Type, not mutability
//! - Lowering (lower.rs): Constructor and RecordSet ignore mutability
//! - Runtime (vm.rs): RecordSet opcode has no mutability check
//!
//! This test documents the current behavior where both mutable and immutable
//! fields can be parsed and type-checked successfully.
//!
//! TODO: Implement mutability enforcement at compile-time or runtime.
//! See TKS_OOP_HYBRID.md for the planned "var" keyword support.

use tkscore::ast::Type;
use tkscore::parser::parse_program;
use tkstypes::infer::infer_program;

/// Test that the parser accepts the `mut` keyword for mutable fields.
/// Currently, mutability is parsed but not enforced at the type level.
#[test]
fn mutable_field_is_parsed() {
    // Class with a mutable field using `mut` keyword
    let source = r#"
class Counter {
  specifics {
    mut value: Int;
    increment: Int;
  }
  details { }
  actions { }
}
repeat Counter { value: 0, increment: 1 }
"#;
    let program = parse_program(source).expect("parse program with mut field");
    let out = infer_program(&program).expect("infer program");
    let entry = out.entry.expect("entry");

    // The result should be a Counter class
    match entry.ty {
        Type::Class(name) => assert_eq!(name, "Counter"),
        other => panic!("expected Counter class, got {other:?}"),
    }
}

/// Test that immutable fields (no `mut` keyword) are also accepted.
/// This is the default behavior - fields are immutable unless marked `mut`.
#[test]
fn immutable_field_is_parsed() {
    // Class with an immutable field (no `mut` keyword)
    let source = r#"
class Point {
  specifics { x: Int; y: Int; }
  details { }
  actions { }
}
repeat Point { x: 1, y: 2 }
"#;
    let program = parse_program(source).expect("parse program with immutable fields");
    let out = infer_program(&program).expect("infer program");
    let entry = out.entry.expect("entry");

    match entry.ty {
        Type::Class(name) => assert_eq!(name, "Point"),
        other => panic!("expected Point class, got {other:?}"),
    }
}

/// Test that mixed mutable and immutable fields work together.
/// Documents the current behavior where mutability is tracked in AST
/// but not enforced at type level.
#[test]
fn mixed_mutability_fields() {
    // Using Int type since it's simpler for this test
    let source = r#"
class Entity {
  specifics {
    mut health: Int;
    id: Int;
    mut score: Int;
  }
  details { }
  actions { }
}
repeat Entity { health: 100, id: 42, score: 0 }
"#;
    let program = parse_program(source).expect("parse program");
    let out = infer_program(&program).expect("infer program");
    let entry = out.entry.expect("entry");

    match entry.ty {
        Type::Class(name) => assert_eq!(name, "Entity"),
        other => panic!("expected Entity class, got {other:?}"),
    }
}

/// Documents that the AST correctly captures the mutable flag.
/// This test verifies the parser sets FieldDecl.mutable correctly.
#[test]
fn ast_captures_mutability_flag() {
    use tkscore::ast::TopDecl;

    let source = r#"
class Wallet {
  specifics {
    mut balance: Int;
    owner: Str;
  }
  details { }
  actions { }
}
"#;
    let program = parse_program(source).expect("parse program");

    // Find the class declaration
    let class_decl = program.decls.iter().find_map(|decl| {
        if let TopDecl::ClassDecl(cd) = decl {
            Some(cd)
        } else {
            None
        }
    }).expect("should have class declaration");

    assert_eq!(class_decl.name, "Wallet");
    assert_eq!(class_decl.specifics.len(), 2);

    // First field should be mutable
    let balance_field = &class_decl.specifics[0];
    assert_eq!(balance_field.name, "balance");
    assert!(balance_field.mutable, "balance field should be mutable");

    // Second field should be immutable
    let owner_field = &class_decl.specifics[1];
    assert_eq!(owner_field.name, "owner");
    assert!(!owner_field.mutable, "owner field should be immutable");
}
