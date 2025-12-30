use crate::span::Span;

pub type Ident = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum World {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aspect {
    A,
    B,
    C,
    D,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    Float(f64),
    Complex { re: f64, im: f64 },
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub decls: Vec<TopDecl>,
    pub entry: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopDecl {
    LetDecl {
        span: Span,
        name: Ident,
        scheme: Option<TypeScheme>,
        value: Expr,
    },
    TypeDecl {
        span: Span,
        name: Ident,
        params: Vec<Ident>,
        body: Type,
    },
    EffectDecl {
        span: Span,
        name: Ident,
        ops: Vec<OpSig>,
    },
    HandlerDecl {
        span: Span,
        name: Ident,
        handler_type: Option<Type>,
        def: HandlerDef,
    },
    ModuleDecl(ModuleDecl),
    ExternDecl(ExternDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpSig {
    pub span: Span,
    pub name: Ident,
    pub input: Type,
    pub output: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HandlerDef {
    pub span: Span,
    pub return_clause: (Ident, Expr),
    pub op_clauses: Vec<OpClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpClause {
    pub span: Span,
    pub op: Ident,
    pub arg: Ident,
    pub k: Ident,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HandlerRef {
    Named {
        span: Span,
        name: Ident,
    },
    Inline {
        span: Span,
        def: HandlerDef,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Var {
        span: Span,
        name: Ident,
    },
    Lit {
        span: Span,
        literal: Literal,
    },
    Lam {
        span: Span,
        param: Ident,
        param_type: Option<Type>,
        body: Box<Expr>,
    },
    App {
        span: Span,
        func: Box<Expr>,
        arg: Box<Expr>,
    },
    Let {
        span: Span,
        name: Ident,
        scheme: Option<TypeScheme>,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    If {
        span: Span,
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    Element {
        span: Span,
        world: World,
        index: u8,
    },
    Foundation {
        span: Span,
        level: u8,
        aspect: Aspect,
    },
    Noetic {
        span: Span,
        index: u8,
        expr: Box<Expr>,
    },
    Fractal {
        span: Span,
        seq: Vec<u8>,
        ellipsis: Option<u8>,
        subscript: Option<Box<Expr>>,
        expr: Box<Expr>,
    },
    RPMReturn {
        span: Span,
        expr: Box<Expr>,
    },
    RPMBind {
        span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    RPMCheck {
        span: Span,
        expr: Box<Expr>,
    },
    RPMAcquire {
        span: Span,
        expr: Box<Expr>,
    },
    ACBE {
        span: Span,
        goal: Box<Expr>,
        expr: Box<Expr>,
    },
    Handle {
        span: Span,
        expr: Box<Expr>,
        handler: HandlerRef,
    },
    Perform {
        span: Span,
        op: Ident,
        arg: Box<Expr>,
    },
    OrdLit {
        span: Span,
        value: OrdinalLiteral,
    },
    OrdOmega {
        span: Span,
    },
    OrdEpsilon {
        span: Span,
        index: u64,
    },
    OrdAleph {
        span: Span,
        index: u64,
    },
    OrdSucc {
        span: Span,
        expr: Box<Expr>,
    },
    OrdAdd {
        span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    OrdMul {
        span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    OrdExp {
        span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    OrdLimit {
        span: Span,
        binder: Ident,
        bound: Box<Expr>,
        body: Box<Expr>,
    },
    TransfiniteLoop {
        span: Span,
        index: Ident,
        ordinal: Box<Expr>,
        init: Box<Expr>,
        step_param: Ident,
        step_body: Box<Expr>,
        limit_param: Ident,
        limit_body: Box<Expr>,
    },
    Superpose {
        span: Span,
        states: Vec<(Expr, Expr)>,
    },
    Measure {
        span: Span,
        expr: Box<Expr>,
    },
    Entangle {
        span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Ket {
        span: Span,
        expr: Box<Expr>,
    },
    Bra {
        span: Span,
        expr: Box<Expr>,
    },
    BraKet {
        span: Span,
        left: Box<Expr>,
        right: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrdinalLiteral {
    Finite(u64),
    Omega,
    OmegaPlus(u64),
    OmegaTimes(u64),
    OmegaPow(u64),
    Epsilon(u64),
    Aleph(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeScheme {
    pub vars: Vec<Ident>,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Var(Ident),
    Int,
    Bool,
    Unit,
    Void,
    Element(Option<World>),
    Foundation,
    Domain,
    Noetic(Box<Type>),
    Fractal(Box<Type>),
    RPM(Box<Type>),
    Fun(Box<Type>, Box<Type>),
    Product(Box<Type>, Box<Type>),
    Sum(Box<Type>, Box<Type>),
    Effectful(Box<Type>, EffectRow),
    Handler {
        effect: EffectRow,
        input: Box<Type>,
        output: Box<Type>,
    },
    Ordinal,
    QState(Box<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EffectRow {
    Empty,
    Cons(Ident, Box<EffectRow>),
    Var(Ident),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDecl {
    pub span: Span,
    pub path: Vec<Ident>,
    pub body: ModuleBody,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleBody {
    pub span: Span,
    pub exports: Option<ExportDecl>,
    pub imports: Vec<ImportDecl>,
    pub decls: Vec<TopDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExportDecl {
    pub span: Span,
    pub items: Vec<ExportItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportItem {
    Value(Ident),
    Type { name: Ident, transparent: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportDecl {
    Qualified { span: Span, path: Vec<Ident> },
    Aliased { span: Span, path: Vec<Ident>, alias: Ident },
    Selective { span: Span, path: Vec<Ident>, items: Vec<ImportItem> },
    Wildcard { span: Span, path: Vec<Ident> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportItem {
    pub name: Ident,
    pub alias: Option<Ident>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternDecl {
    pub span: Span,
    pub convention: Convention,
    pub safety: Safety,
    pub name: Ident,
    pub params: Vec<ExternParam>,
    pub return_type: Type,
    pub effects: Option<EffectRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternParam {
    pub name: Ident,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Convention {
    C,
    StdCall,
    FastCall,
    System,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Safety {
    Safe,
    Unsafe,
}
