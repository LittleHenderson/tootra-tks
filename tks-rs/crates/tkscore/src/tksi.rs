use crate::ast::{Convention, EffectRow, Safety, Type, World};
use crate::resolve::{
    EffectSignature, ExportedType, ExternSignature, ModuleExports, ResolvedProgram,
};

pub fn emit_tksi(resolved: &ResolvedProgram) -> String {
    let mut out = String::new();
    for module in &resolved.modules {
        let name = module.path.join(".");
        out.push_str("module ");
        out.push_str(&name);
        out.push('\n');
        emit_exports(&mut out, &module.exports);
        out.push('\n');
    }
    out
}

fn emit_exports(out: &mut String, exports: &ModuleExports) {
    let mut values = exports.values.clone();
    values.sort();
    for value in values {
        out.push_str("export value ");
        out.push_str(&value);
        out.push('\n');
    }

    let mut types: Vec<ExportedType> = exports.types.clone();
    types.sort_by(|left, right| left.name.cmp(&right.name));
    for ty in types {
        out.push_str("export type ");
        out.push_str(&ty.name);
        if ty.transparent {
            out.push_str(" =");
        }
        out.push('\n');
    }

    let mut effects: Vec<EffectSignature> = exports.effects.clone();
    effects.sort_by(|left, right| left.name.cmp(&right.name));
    for effect in effects {
        emit_effect(out, &effect);
    }

    let mut externs: Vec<ExternSignature> = exports.externs.clone();
    externs.sort_by(|left, right| left.name.cmp(&right.name));
    for extern_sig in externs {
        emit_extern(out, &extern_sig);
    }
}

fn emit_effect(out: &mut String, effect: &EffectSignature) {
    out.push_str("export effect ");
    out.push_str(&effect.name);
    out.push('\n');
    for op in &effect.ops {
        out.push_str("  op ");
        out.push_str(&op.name);
        out.push_str("(arg: ");
        out.push_str(&format_type(&op.input));
        out.push_str("): ");
        out.push_str(&format_type(&op.output));
        out.push('\n');
    }
}

fn emit_extern(out: &mut String, extern_sig: &ExternSignature) {
    out.push_str("export extern ");
    out.push_str(&format_convention(&extern_sig.convention));
    out.push(' ');
    out.push_str(&format_safety(&extern_sig.safety));
    out.push_str(" fn ");
    out.push_str(&extern_sig.name);
    out.push('(');
    for (idx, param) in extern_sig.params.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&param.name);
        out.push_str(": ");
        out.push_str(&format_type(&param.ty));
    }
    out.push_str("): ");
    out.push_str(&format_type(&extern_sig.return_type));
    if let Some(row) = &extern_sig.effects {
        out.push_str(" !");
        out.push_str(&format_effect_row(row));
    }
    out.push('\n');
}

fn format_convention(convention: &Convention) -> &'static str {
    match convention {
        Convention::C => "c",
        Convention::StdCall => "stdcall",
        Convention::FastCall => "fastcall",
        Convention::System => "system",
    }
}

fn format_safety(safety: &Safety) -> &'static str {
    match safety {
        Safety::Safe => "safe",
        Safety::Unsafe => "unsafe",
    }
}

fn format_type(ty: &Type) -> String {
    format_type_prec(ty, 0)
}

fn format_type_prec(ty: &Type, prec: u8) -> String {
    match ty {
        Type::Var(name) => name.clone(),
        Type::Int => "Int".to_string(),
        Type::Bool => "Bool".to_string(),
        Type::Unit => "Unit".to_string(),
        Type::Void => "Void".to_string(),
        Type::Element(world) => match world {
            Some(world) => format!("Element[{}]", format_world(*world)),
            None => "Element".to_string(),
        },
        Type::Foundation => "Foundation".to_string(),
        Type::Domain => "Domain".to_string(),
        Type::Noetic(inner) => format!("Noetic[{}]", format_type(inner)),
        Type::Fractal(inner) => format!("Fractal[{}]", format_type(inner)),
        Type::RPM(inner) => format!("RPM[{}]", format_type(inner)),
        Type::QState(inner) => format!("QState[{}]", format_type(inner)),
        Type::Handler { effect, input, output } => format!(
            "Handler[{}, {}, {}]",
            format_effect_row(effect),
            format_type(input),
            format_type(output)
        ),
        Type::Fun(left, right) => wrap_prec(
            prec,
            1,
            format!(
                "{} -> {}",
                format_type_prec(left, 2),
                format_type_prec(right, 1)
            ),
        ),
        Type::Effectful(inner, row) => wrap_prec(
            prec,
            2,
            format!("{} !{}", format_type_prec(inner, 3), format_effect_row(row)),
        ),
        Type::Sum(left, right) => wrap_prec(
            prec,
            3,
            format!(
                "{} + {}",
                format_type_prec(left, 4),
                format_type_prec(right, 4)
            ),
        ),
        Type::Product(left, right) => wrap_prec(
            prec,
            4,
            format!(
                "{} * {}",
                format_type_prec(left, 5),
                format_type_prec(right, 5)
            ),
        ),
        Type::Class(name) => name.clone(),
        Type::Ordinal => "Ordinal".to_string(),
        Type::Str => "Str".to_string(),
        Type::Array(inner) => format!("Array[{}]", format_type(inner)),
    }
}

fn wrap_prec(current: u8, required: u8, value: String) -> String {
    if current > required {
        format!("({value})")
    } else {
        value
    }
}

fn format_effect_row(row: &EffectRow) -> String {
    let mut names = Vec::new();
    let mut tail = None;
    let mut cursor = row;
    loop {
        match cursor {
            EffectRow::Empty => break,
            EffectRow::Var(name) => {
                tail = Some(name.clone());
                break;
            }
            EffectRow::Cons(name, rest) => {
                names.push(name.clone());
                cursor = rest;
            }
        }
    }

    let mut out = String::from("{");
    out.push_str(&names.join(","));
    if let Some(tail) = tail {
        out.push('|');
        out.push_str(&tail);
    }
    out.push('}');
    out
}

fn format_world(world: World) -> &'static str {
    match world {
        World::A => "A",
        World::B => "B",
        World::C => "C",
        World::D => "D",
    }
}
