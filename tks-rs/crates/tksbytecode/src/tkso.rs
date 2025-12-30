use std::fmt;

use crate::bytecode::{Instruction, Opcode};

const TKSO_MAGIC: &[u8; 4] = b"TKSO";
const TKSO_VERSION: u8 = 1;

#[derive(Debug, Clone)]
pub enum TksoError {
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u8),
    UnexpectedEof,
    InvalidOpcode(u16),
    InvalidOperandMask(u8),
    TooManyInstructions(usize),
    UnknownOpcode(Opcode),
}

impl fmt::Display for TksoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TksoError::InvalidMagic(magic) => {
                write!(f, "invalid tkso magic: {magic:?}")
            }
            TksoError::UnsupportedVersion(version) => {
                write!(f, "unsupported tkso version {version}")
            }
            TksoError::UnexpectedEof => write!(f, "unexpected end of tkso data"),
            TksoError::InvalidOpcode(value) => write!(f, "invalid opcode {value}"),
            TksoError::InvalidOperandMask(mask) => write!(f, "invalid operand mask {mask}"),
            TksoError::TooManyInstructions(count) => {
                write!(f, "too many instructions for tkso: {count}")
            }
            TksoError::UnknownOpcode(opcode) => {
                write!(f, "unknown opcode {opcode:?}")
            }
        }
    }
}

pub fn encode(instructions: &[Instruction]) -> Result<Vec<u8>, TksoError> {
    if instructions.len() > u32::MAX as usize {
        return Err(TksoError::TooManyInstructions(instructions.len()));
    }

    let mut out = Vec::with_capacity(9 + instructions.len() * 4);
    out.extend_from_slice(TKSO_MAGIC);
    out.push(TKSO_VERSION);
    out.extend_from_slice(&(instructions.len() as u32).to_le_bytes());

    for instr in instructions {
        let opcode_id = opcode_to_u16(instr.opcode)?;
        out.extend_from_slice(&opcode_id.to_le_bytes());
        out.push(instr.flags);
        let mask = operand_mask(instr);
        out.push(mask);
        if let Some(op1) = instr.operand1 {
            out.extend_from_slice(&op1.to_le_bytes());
        }
        if let Some(op2) = instr.operand2 {
            out.extend_from_slice(&op2.to_le_bytes());
        }
    }

    Ok(out)
}

pub fn decode(bytes: &[u8]) -> Result<Vec<Instruction>, TksoError> {
    let mut cursor = 0usize;
    let magic = read_exact(bytes, &mut cursor, 4)?;
    let magic = [magic[0], magic[1], magic[2], magic[3]];
    if &magic != TKSO_MAGIC {
        return Err(TksoError::InvalidMagic(magic));
    }

    let version = read_u8(bytes, &mut cursor)?;
    if version != TKSO_VERSION {
        return Err(TksoError::UnsupportedVersion(version));
    }

    let count = read_u32(bytes, &mut cursor)? as usize;
    let mut instructions = Vec::with_capacity(count);

    for _ in 0..count {
        let opcode_id = read_u16(bytes, &mut cursor)?;
        let opcode = opcode_from_u16(opcode_id).ok_or(TksoError::InvalidOpcode(opcode_id))?;
        let flags = read_u8(bytes, &mut cursor)?;
        let mask = read_u8(bytes, &mut cursor)?;
        if mask & !0b11 != 0 {
            return Err(TksoError::InvalidOperandMask(mask));
        }
        let operand1 = if mask & 0b01 != 0 {
            Some(read_u64(bytes, &mut cursor)?)
        } else {
            None
        };
        let operand2 = if mask & 0b10 != 0 {
            Some(read_u64(bytes, &mut cursor)?)
        } else {
            None
        };
        instructions.push(Instruction {
            opcode,
            flags,
            operand1,
            operand2,
        });
    }

    Ok(instructions)
}

fn operand_mask(instr: &Instruction) -> u8 {
    let mut mask = 0u8;
    if instr.operand1.is_some() {
        mask |= 0b01;
    }
    if instr.operand2.is_some() {
        mask |= 0b10;
    }
    mask
}

fn read_exact<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    len: usize,
) -> Result<&'a [u8], TksoError> {
    let end = cursor.checked_add(len).ok_or(TksoError::UnexpectedEof)?;
    if end > bytes.len() {
        return Err(TksoError::UnexpectedEof);
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, TksoError> {
    let slice = read_exact(bytes, cursor, 1)?;
    Ok(slice[0])
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, TksoError> {
    let slice = read_exact(bytes, cursor, 2)?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, TksoError> {
    let slice = read_exact(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, TksoError> {
    let slice = read_exact(bytes, cursor, 8)?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn opcode_to_u16(opcode: Opcode) -> Result<u16, TksoError> {
    OPCODES
        .iter()
        .position(|candidate| *candidate == opcode)
        .map(|index| index as u16)
        .ok_or(TksoError::UnknownOpcode(opcode))
}

fn opcode_from_u16(value: u16) -> Option<Opcode> {
    OPCODES.get(value as usize).copied()
}

// Keep this list in the same order as Opcode.
const OPCODES: &[Opcode] = &[
    Opcode::Nop,
    Opcode::PushInt,
    Opcode::PushBool,
    Opcode::PushUnit,
    Opcode::Pop,
    Opcode::Dup,
    Opcode::Swap,
    Opcode::Rot,
    Opcode::Load,
    Opcode::Store,
    Opcode::LoadGlobal,
    Opcode::StoreGlobal,
    Opcode::Jmp,
    Opcode::JmpIf,
    Opcode::JmpUnless,
    Opcode::Call,
    Opcode::Ret,
    Opcode::TailCall,
    Opcode::Add,
    Opcode::Sub,
    Opcode::Mul,
    Opcode::Div,
    Opcode::Eq,
    Opcode::Lt,
    Opcode::And,
    Opcode::Or,
    Opcode::Not,
    Opcode::PushElement,
    Opcode::PushFoundation,
    Opcode::ElementWing,
    Opcode::ElementIndex,
    Opcode::FoundationLevel,
    Opcode::FoundationAspect,
    Opcode::PushNoetic,
    Opcode::ApplyNoetic,
    Opcode::NoeticCompose,
    Opcode::NoeticLevel,
    Opcode::MakeFractal,
    Opcode::FractalLevel,
    Opcode::AcbeInit,
    Opcode::AcbeDescend,
    Opcode::AcbeComplete,
    Opcode::AcbeResult,
    Opcode::RpmReturn,
    Opcode::RpmBind,
    Opcode::RpmCheck,
    Opcode::RpmAcquire,
    Opcode::RpmFail,
    Opcode::RpmIsSuccess,
    Opcode::RpmUnwrap,
    Opcode::RpmState,
    Opcode::InstallHandler,
    Opcode::RemoveHandler,
    Opcode::PerformEffect,
    Opcode::Resume,
    Opcode::ResumeWith,
    Opcode::CaptureCont,
    Opcode::AbortCont,
    Opcode::HandlerReturn,
    Opcode::PushOrd,
    Opcode::PushOmega,
    Opcode::PushEpsilon,
    Opcode::OrdSucc,
    Opcode::OrdAdd,
    Opcode::OrdMul,
    Opcode::OrdExp,
    Opcode::OrdLt,
    Opcode::OrdIsLimit,
    Opcode::OrdIsFinite,
    Opcode::TransLoopInit,
    Opcode::TransLoopStep,
    Opcode::TransLoopCheck,
    Opcode::TransLoopEnd,
    Opcode::OrdLimit,
    Opcode::MakeKet,
    Opcode::MakeBra,
    Opcode::BraKet,
    Opcode::Superpose,
    Opcode::Measure,
    Opcode::Entangle,
    Opcode::Tensor,
    Opcode::QApply,
];
