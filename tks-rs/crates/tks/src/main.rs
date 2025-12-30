use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use tkscore::parser::{parse_program, ParseError};

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
            return Err("tks run: bytecode execution not implemented".to_string());
        }
    }

    let source = read_source(path)?;
    parse_program(&source).map_err(|err| format_parse_error(path, &err))?;
    Err("tks run: VM execution not implemented".to_string())
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

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  tks run <file.tks|file.tkso>");
    eprintln!("  tks repl");
}
