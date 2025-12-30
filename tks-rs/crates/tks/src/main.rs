use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use tksbytecode::emit::{emit, EmitError};
use tksbytecode::tkso::{decode as decode_tkso, TksoError};
use tkscore::parser::{parse_program, ParseError};
use tksir::lower::{lower_program, LowerError};
use tksvm::vm::{VmError, VmState};
use tkstypes::infer::{infer_program, TypeError};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(2);
    }

    match args[1].as_str() {
        "run" => {
            if let Err(message) = cmd_run(&args[2..]) {
                eprintln!("{message}");
                process::exit(1);
            }
        }
        "repl" => {
            eprintln!("tks repl: not implemented");
            process::exit(1);
        }
        "help" | "-h" | "--help" => {
            print_usage();
        }
        _ => {
            eprintln!("tks: unknown command");
            print_usage();
            process::exit(2);
        }
    }
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let Some(path) = args.first() else {
        return Err("tks run: missing input file".to_string());
    };

    if path != "-" {
        let ext = Path::new(path).extension().and_then(|ext| ext.to_str());
        if matches!(ext, Some("tkso")) {
            let bytes = fs::read(path).map_err(|err| format!("{path}: {err}"))?;
            let bytecode =
                decode_tkso(&bytes).map_err(|err| format_tkso_error(path, &err))?;
            let mut vm = VmState::new(bytecode);
            let result = vm.run().map_err(|err| format_vm_error(path, &err))?;
            println!("{result:?}");
            return Ok(());
        }
    }

    let source = read_source(path)?;
    let program = parse_program(&source).map_err(|err| format_parse_error(path, &err))?;
    infer_program(&program).map_err(|err| format_type_error(path, &err))?;
    let ir = lower_program(&program).map_err(|err| format_lower_error(path, &err))?;
    let bytecode = emit(&ir).map_err(|err| format_emit_error(path, &err))?;
    let mut vm = VmState::new(bytecode);
    let result = vm.run().map_err(|err| format_vm_error(path, &err))?;
    println!("{result:?}");
    Ok(())
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

fn format_parse_error(path: &str, err: &ParseError) -> String {
    match err.span {
        Some(span) => format!("{path}:{line}:{col}: {message}", line = span.line, col = span.column, message = err.message),
        None => format!("{path}: {message}", message = err.message),
    }
}

fn format_type_error(path: &str, err: &TypeError) -> String {
    format!("{path}: type error: {err}")
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

fn format_vm_error(path: &str, err: &VmError) -> String {
    format!("{path}: vm error: {err:?}")
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  tks run <file.tks|file.tkso>");
    eprintln!("  tks repl");
}
