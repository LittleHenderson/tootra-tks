use crate::resolve::{ExportedType, ModuleExports, ResolvedProgram};

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
}
