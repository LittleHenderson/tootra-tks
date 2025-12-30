use std::env;
use std::fs;
use std::io::{self, Read};
use std::process;

use tkscore::parser::{parse_program, ParseError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmitKind {
    Ast,
    Ir,
    Bc,
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
    parse_program(&source).map_err(|err| format_parse_error(path, &err))?;
    eprintln!("ok");
    Ok(())
}

fn cmd_build(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut emit = EmitKind::Ast;
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
                emit = parse_emit(kind).ok_or_else(|| {
                    format!("tksc build: unknown emit kind '{kind}' (use ast, ir, bc)")
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
    match emit {
        EmitKind::Ast => {
            let program = parse_program(&source).map_err(|err| format_parse_error(path, &err))?;
            let out = format!("{program:#?}\n");
            write_output(output, &out)?;
            Ok(())
        }
        EmitKind::Ir => Err("tksc build: IR emission not implemented".to_string()),
        EmitKind::Bc => Err("tksc build: bytecode emission not implemented".to_string()),
    }
}

fn parse_emit(value: &str) -> Option<EmitKind> {
    match value {
        "ast" => Some(EmitKind::Ast),
        "ir" => Some(EmitKind::Ir),
        "bc" => Some(EmitKind::Bc),
        _ => None,
    }
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

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  tksc check <file.tks>");
    eprintln!("  tksc build <file.tks> [--emit ast|ir|bc] [-o out]");
}

fn print_build_usage() {
    eprintln!("Usage:");
    eprintln!("  tksc build <file.tks> [--emit ast|ir|bc] [-o out]");
}
