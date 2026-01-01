#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Nop,
    PushInt,
    PushBool,
    PushUnit,
    PushClosure,
    Pop,
    Dup,
    Swap,
    Rot,
    Load,
    Store,
    LoadGlobal,
    StoreGlobal,
    Jmp,
    JmpIf,
    JmpUnless,
    Call,
    Ret,
    TailCall,
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Lt,
    And,
    Or,
    Not,
    PushElement,
    PushFoundation,
    ElementWing,
    ElementIndex,
    FoundationLevel,
    FoundationAspect,
    PushNoetic,
    PushExtern,
    ApplyNoetic,
    NoeticCompose,
    NoeticLevel,
    MakeFractal,
    FractalLevel,
    AcbeInit,
    AcbeDescend,
    AcbeComplete,
    AcbeResult,
    RpmReturn,
    RpmBind,
    RpmCheck,
    RpmAcquire,
    RpmFail,
    RpmIsSuccess,
    RpmUnwrap,
    RpmState,
    InstallHandler,
    RemoveHandler,
    PerformEffect,
    Resume,
    ResumeWith,
    CaptureCont,
    AbortCont,
    HandlerReturn,
    PushOrd,
    PushOmega,
    PushEpsilon,
    OrdSucc,
    OrdAdd,
    OrdMul,
    OrdExp,
    OrdLt,
    OrdIsLimit,
    OrdIsFinite,
    TransLoopInit,
    TransLoopStep,
    TransLoopCheck,
    TransLoopEnd,
    OrdLimit,
    MakeKet,
    MakeBra,
    BraKet,
    Superpose,
    Measure,
    Entangle,
    Tensor,
    QApply,
    MakeRecord,
    RecordGet,
    RecordSet,
    // String and Array opcodes
    PushStr,
    MakeArray,
    ArrayGet,
    ArraySet,
    ArrayLen,
    ArrayPush,
    StrConcat,
    StrLen,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: Opcode,
    pub flags: u8,
    pub operand1: Option<u64>,
    pub operand2: Option<u64>,
}

fn name_id(name: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

pub fn extern_id(name: &str) -> u64 {
    name_id(name)
}

pub fn string_id(s: &str) -> u64 {
    name_id(s)
}

pub fn field_id(name: &str) -> u64 {
    name_id(name)
}
