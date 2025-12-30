use tkscore::ast::{EffectRow, Type};
use tkstypes::infer::{unify_rows, unify_types, Subst, TypeError};

fn row(labels: &[&str], tail: Option<&str>) -> EffectRow {
    let mut row = match tail {
        Some(name) => EffectRow::Var(name.to_string()),
        None => EffectRow::Empty,
    };
    for label in labels.iter().rev() {
        row = EffectRow::Cons((*label).to_string(), Box::new(row));
    }
    row
}

#[test]
fn unify_closed_rows_order_insensitive() {
    let mut subst = Subst::default();
    let left = row(&["A", "B"], None);
    let right = row(&["B", "A"], None);
    unify_rows(&mut subst, &left, &right).expect("rows unify");
}

#[test]
fn unify_open_row_with_closed() {
    let mut subst = Subst::default();
    let left = row(&["A"], Some("r"));
    let right = row(&["A", "B"], None);
    unify_rows(&mut subst, &left, &right).expect("rows unify");
    let bound = subst.row_binding("r").expect("row var bound");
    assert_eq!(bound, row(&["B"], None));
}

#[test]
fn unify_row_mismatch() {
    let mut subst = Subst::default();
    let left = row(&["A"], None);
    let right = row(&["B"], None);
    let err = unify_rows(&mut subst, &left, &right).expect_err("mismatch");
    assert!(matches!(err, TypeError::RowMismatch(_, _)));
}

#[test]
fn unify_type_fun() {
    let mut subst = Subst::default();
    let left = Type::Fun(Box::new(Type::Int), Box::new(Type::Bool));
    let right = Type::Fun(Box::new(Type::Int), Box::new(Type::Bool));
    unify_types(&mut subst, &left, &right).expect("types unify");
}

#[test]
fn unify_effectful_types() {
    let mut subst = Subst::default();
    let eff = row(&["RPMEffect"], None);
    let left = Type::Effectful(Box::new(Type::Int), eff.clone());
    let right = Type::Effectful(Box::new(Type::Int), eff);
    unify_types(&mut subst, &left, &right).expect("effectful unify");
}
