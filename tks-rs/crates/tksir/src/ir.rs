use tkscore::ast::{Expr, Ident, Literal, World};

#[derive(Debug, Clone, PartialEq)]
pub enum IRVal {
    Var(Ident),
    Lit(Literal),
    Lam(Ident, Box<IRTerm>),
    Element(World, u8),
    Noetic(u8),
    Ordinal(Expr),
    Ket(Box<IRVal>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum IRTerm {
    Return(IRVal),
    Let(Ident, IRComp, Box<IRTerm>),
    App(IRVal, IRVal),
    If(IRVal, Box<IRTerm>, Box<IRTerm>),
    Perform(Ident, IRVal),
    Handle(Box<IRTerm>, IRHandler),
    RPMAcquire(IRVal),
    RPMCheck(IRVal),
    RPMFail,
    OrdLimit(Ident, IRVal, Box<IRTerm>),
    OrdSucc(IRVal),
    OrdOp(OrdOp, IRVal, IRVal),
    Superpose(Vec<(IRVal, IRVal)>),
    Measure(IRVal),
    Entangle(IRVal, IRVal),
}

#[derive(Debug, Clone, PartialEq)]
pub enum IRComp {
    Pure(IRVal),
    Effect(IRTerm),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrdOp {
    Add,
    Mul,
    Exp,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IRHandler {
    pub return_clause: (Ident, IRTerm),
    pub op_clauses: Vec<IRHandlerClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IRHandlerClause {
    pub op: Ident,
    pub arg: Ident,
    pub k: Ident,
    pub body: IRTerm,
}
