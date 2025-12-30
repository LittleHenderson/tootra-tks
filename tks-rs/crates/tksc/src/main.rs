use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process;

use tksbytecode::emit::{emit, EmitError};
use tksbytecode::tkso::{encode as encode_tkso, TksoError};
use tksbytecode::bytecode::Instruction;
use tkscore::ast::{Program, TopDecl};
use tkscore::parser::{parse_program, ParseError};
use tkscore::resolve::{resolve_program, ResolveError, ResolvedProgram};
use tkscore::tksi::emit_tksi;
use tksir::lower::{lower_program, LowerError};
use tkstypes::infer::{infer_program, TypeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmitKind {
    Ast,
    Ir,
    Bc,
    Tksi,
}

#[derive(Debug, Default)]
struct Stdlib {
    decls: Vec<TopDecl>,
    module_paths: Vec<Vec<String>>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(2);
    }

    match args[1].as_str() {
        "check" => {
            if let Err(message) = cmd_check(&args[2..]) {
                eprintln!("{message}");
                process::exit(1);
            }
        }
        "build" => {
            if let Err(message) = cmd_build(&args[2..]) {
                eprintln!("{message}");
                process::exit(1);
            }
        }
        "help" | "-h" | "--help" => {
            print_usage();
        }
        _ => {
            eprintln!("tksc: unknown command");
            print_usage();
            process::exit(2);
        }
    }
}

fn cmd_check(args: &[String]) -> Result<(), String> {
    let Some(path) = args.first() else {
        return Err("tksc check: missing input file".to_string());
    };
    let source = read_source(path)?;
    let program = parse_program(&source).map_err(|err| format_parse_error(path, &err))?;
    let stdlib = load_stdlib()?;
    let program = merge_program_with_stdlib(program, &stdlib);
    resolve_program(&program).map_err(|err| format_resolve_error(path, &err))?;
    infer_program(&program).map_err(|err| format_type_error(path, &err))?;
    eprintln!("ok");
    Ok(())
}

fn cmd_build(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut emit_kind = EmitKind::Ast;
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "-o" => {
                idx += 1;
                let Some(path) = args.get(idx) else {
                    return Err("tksc build: missing value after -o".to_string());
                };
                output = Some(path.as_str());
            }
            "--emit" => {
                idx += 1;
                let Some(kind) = args.get(idx) else {
                    return Err("tksc build: missing value after --emit".to_string());
                };
                emit_kind = parse_emit(kind).ok_or_else(|| {
                    format!("tksc build: unknown emit kind '{kind}' (use ast, ir, bc, tksi)")
                })?;
            }
            "-h" | "--help" => {
                print_build_usage();
                return Ok(());
            }
            value => {
                if input.is_some() {
                    return Err("tksc build: unexpected extra argument".to_string());
                }
                input = Some(value);
            }
        }
        idx += 1;
    }

    let Some(path) = input else {
        return Err("tksc build: missing input file".to_string());
    };

    let source = read_source(path)?;
    let program = parse_program(&source).map_err(|err| format_parse_error(path, &err))?;
    let stdlib = load_stdlib()?;
    let merged = merge_program_with_stdlib(program.clone(), &stdlib);
    let resolved = resolve_program(&merged).map_err(|err| format_resolve_error(path, &err))?;
    match emit_kind {
        EmitKind::Ast => {
            let out = format!("{program:#?}\n");
            write_output(output, &out)?;
            Ok(())
        }
        EmitKind::Ir => {
            let ir = lower_program(&merged).map_err(|err| format_lower_error(path, &err))?;
            let out = format!("{ir:#?}\n");
            write_output(output, &out)?;
            Ok(())
        }
        EmitKind::Tksi => {
            let out = emit_tksi(&filter_stdlib_modules(resolved, &stdlib));
            write_output(output, &out)?;
            Ok(())
        }
        EmitKind::Bc => {
            let ir = lower_program(&merged).map_err(|err| format_lower_error(path, &err))?;
            let bytecode = emit(&ir).map_err(|err| format_emit_error(path, &err))?;
            let out = format_bytecode(&bytecode);
            if let Some(path) = output {
                if is_tkso_path(path) {
                    let encoded =
                        encode_tkso(&bytecode).map_err(|err| format_tkso_error(path, &err))?;
                    fs::write(path, encoded).map_err(|err| format!("{path}: {err}"))?;
                    print!("{out}");
                    return Ok(());
                }
            }
            write_output(output, &out)?;
            Ok(())
        }
    }
}

fn parse_emit(value: &str) -> Option<EmitKind> {
    match value {
        "ast" => Some(EmitKind::Ast),
        "ir" => Some(EmitKind::Ir),
        "bc" => Some(EmitKind::Bc),
        "tksi" => Some(EmitKind::Tksi),
        _ => None,
    }
}

fn stdlib_root() -> Option<PathBuf> {
    if let Ok(value) = env::var("TKS_STDLIB_DIR") {
        if !value.trim().is_empty() {
            return Some(PathBuf::from(value));
        }
    }
    let default = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("stdlib");
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

fn collect_tks_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let entries = fs::read_dir(&dir).map_err(|err| format!("{}: {err}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|err| format!("{}: {err}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("tks") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

fn load_stdlib() -> Result<Stdlib, String> {
    let Some(root) = stdlib_root() else {
        return Ok(Stdlib::default());
    };
    let files = collect_tks_files(&root)?;
    let mut decls = Vec::new();
    let mut module_paths = Vec::new();
    for path in files {
        let source =
            fs::read_to_string(&path).map_err(|err| format!("{}: {err}", path.display()))?;
        let path_string = path.to_string_lossy();
        let program =
            parse_program(&source).map_err(|err| format_parse_error(&path_string, &err))?;
        if program.entry.is_some() {
            return Err(format!(
                "{}: stdlib modules must not include an entry expression",
                path.display()
            ));
        }
        for decl in &program.decls {
            if let TopDecl::ModuleDecl(module) = decl {
                module_paths.push(module.path.clone());
            }
        }
        decls.extend(program.decls);
    }
    module_paths.sort();
    module_paths.dedup();
    Ok(Stdlib { decls, module_paths })
}

fn merge_program_with_stdlib(mut program: Program, stdlib: &Stdlib) -> Program {
    if stdlib.decls.is_empty() {
        return program;
    }
    let mut decls = stdlib.decls.clone();
    decls.append(&mut program.decls);
    Program {
        decls,
        entry: program.entry,
    }
}

fn filter_stdlib_modules(resolved: ResolvedProgram, stdlib: &Stdlib) -> ResolvedProgram {
    if stdlib.module_paths.is_empty() {
        return resolved;
    }
    let modules = resolved
        .modules
        .into_iter()
        .filter(|module| !stdlib.module_paths.iter().any(|path| *path == module.path))
        .collect();
    ResolvedProgram { modules }
}

fn read_source(path: &str) -> Result<String, String> {
    if path == "-" {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|err| format!("stdin: {err}"))?;
        return Ok(input);
    }
    fs::read_to_string(path).map_err(|err| format!("{path}: {err}"))
}

fn write_output(path: Option<&str>, content: &str) -> Result<(), String> {
    match path {
        Some(path) => fs::write(path, content).map_err(|err| format!("{path}: {err}")),
        None => {
            print!("{content}");
            Ok(())
        }
    }
}

fn format_parse_error(path: &str, err: &ParseError) -> String {
    match err.span {
        Some(span) => format!("{path}:{line}:{col}: {message}", line = span.line, col = span.column, message = err.message),
        None => format!("{path}: {message}", message = err.message),
    }
}

fn format_type_error(path: &str, err: &TypeError) -> String {
    format!("{path}: type error: {err}")
}

fn format_resolve_error(path: &str, err: &ResolveError) -> String {
    format!("{path}: resolve error: {err}")
}

fn format_lower_error(path: &str, err: &LowerError) -> String {
    format!("{path}: lower error: {err}")
}

fn format_emit_error(path: &str, err: &EmitError) -> String {
    format!("{path}: emit error: {err}")
}

fn format_tkso_error(path: &str, err: &TksoError) -> String {
    format!("{path}: tkso error: {err}")
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  tksc check <file.tks>");
    eprintln!("  tksc build <file.tks> [--emit ast|ir|bc|tksi] [-o out]");
}

fn print_build_usage() {
    eprintln!("Usage:");
    eprintln!("  tksc build <file.tks> [--emit ast|ir|bc|tksi] [-o out]");
}

fn format_bytecode(code: &[Instruction]) -> String {
    let mut out = String::new();
    for instr in code {
        out.push_str(&format!("{:?}", instr.opcode));
        if let Some(op1) = instr.operand1 {
            out.push(' ');
            out.push_str(&op1.to_string());
        }
        if let Some(op2) = instr.operand2 {
            out.push(' ');
            out.push_str(&op2.to_string());
        }
        out.push('\n');
    }
    out
}

fn is_tkso_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        == Some("tkso")
}
