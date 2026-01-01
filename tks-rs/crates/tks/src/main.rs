use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process;

use tksbytecode::emit::{emit, EmitError};
use tksbytecode::tkso::{decode as decode_tkso, TksoError};
use tkscore::parser::{parse_program, ParseError};
use tksir::lower::{lower_program, LowerError};
use tksvm::vm::{ExternSafety, Value, VmError, VmState};
use tkstypes::infer::{infer_program, TypeError};

#[cfg(feature = "gpu")]
use tksgpu::GpuContext;

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
        "gpu" => {
            if let Err(message) = cmd_gpu(&args[2..]) {
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
        "version" | "-V" | "--version" => {
            print_version();
        }
        _ => {
            eprintln!("tks: unknown command '{}'", args[1]);
            print_usage();
            process::exit(2);
        }
    }
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let mut enable_ffi = false;
    let mut path = None;
    for arg in args {
        if arg == "--ffi" {
            enable_ffi = true;
            continue;
        }
        if path.is_none() {
            path = Some(arg.as_str());
        } else {
            return Err("tks run: unexpected extra argument".to_string());
        }
    }

    let Some(path) = path else {
        return Err("tks run: missing input file".to_string());
    };

    if path != "-" {
        let ext = Path::new(path).extension().and_then(|ext| ext.to_str());
        if matches!(ext, Some("tkso")) {
            let bytes = fs::read(path).map_err(|err| format!("{path}: {err}"))?;
            let bytecode =
                decode_tkso(&bytes).map_err(|err| format_tkso_error(path, &err))?;
            let mut vm = VmState::new(bytecode);
            if enable_ffi {
                register_default_externs(&mut vm);
            }
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
    if enable_ffi {
        register_default_externs(&mut vm);
    }
    let result = vm.run().map_err(|err| format_vm_error(path, &err))?;
    println!("{result:?}");
    Ok(())
}

#[cfg(feature = "gpu")]
fn cmd_gpu(args: &[String]) -> Result<(), String> {
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("info");
    match cmd {
        "info" => cmd_gpu_info(),
        "add" => cmd_gpu_add(&args[1..]),
        _ => Err("tks gpu: expected 'info' or 'add <len>'".to_string()),
    }
}

#[cfg(not(feature = "gpu"))]
fn cmd_gpu(_args: &[String]) -> Result<(), String> {
    Err("tks gpu: build with --features gpu".to_string())
}

#[cfg(feature = "gpu")]
fn cmd_gpu_info() -> Result<(), String> {
    let ctx = GpuContext::new().map_err(|err| format!("gpu init: {err}"))?;
    let info = ctx.adapter_info();
    println!("name: {}", info.name);
    println!("backend: {:?}", info.backend);
    println!("device_type: {:?}", info.device_type);
    println!("vendor: 0x{:04x}", info.vendor);
    println!("device: 0x{:04x}", info.device);
    Ok(())
}

#[cfg(feature = "gpu")]
fn cmd_gpu_add(args: &[String]) -> Result<(), String> {
    let Some(len_str) = args.first() else {
        return Err("tks gpu add: missing length".to_string());
    };
    let len: usize = len_str
        .parse()
        .map_err(|_| "tks gpu add: length must be a positive integer".to_string())?;
    let ctx = GpuContext::new().map_err(|err| format!("gpu init: {err}"))?;
    let left: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let right: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
    let out = ctx.add_f32(&left, &right).map_err(|err| format!("gpu add: {err}"))?;
    for (idx, ((a, b), got)) in left
        .iter()
        .zip(right.iter())
        .zip(out.iter())
        .enumerate()
    {
        let expected = a + b;
        if (got - expected).abs() > 1e-4 {
            return Err(format!(
                "gpu add mismatch at {idx}: got {got}, expected {expected}"
            ));
        }
    }
    println!("gpu add ok: {len} elements");
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
    eprintln!("tks {}", env!("CARGO_PKG_VERSION"));
    eprintln!("Usage:");
    eprintln!("  tks run [--ffi] <file.tks|file.tkso>");
    #[cfg(feature = "gpu")]
    {
        eprintln!("  tks gpu info");
        eprintln!("  tks gpu add <len>");
    }
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("  tks gpu (build with --features gpu)");
    }
    eprintln!("  tks repl");
    eprintln!("  tks run --ffi <file.tks|file.tkso>");
    eprintln!("  tks --help");
    eprintln!("  tks --version");
}

fn print_version() {
    println!("tks {}", env!("CARGO_PKG_VERSION"));
}

fn register_default_externs(vm: &mut VmState) {
    vm.register_extern_with("print_int", 1, ExternSafety::Safe, ["IO"], ffi_print_int);
    vm.register_extern_with("print_bool", 1, ExternSafety::Safe, ["IO"], ffi_print_bool);
    
    // String I/O externs
    vm.register_extern_with("print", 1, ExternSafety::Safe, ["IO"], ffi_print);
    vm.register_extern_with("println", 1, ExternSafety::Safe, ["IO"], ffi_println);
    vm.register_extern_with("read_line", 0, ExternSafety::Safe, ["IO"], ffi_read_line);
    
    // File I/O externs
    vm.register_extern_with("read_file", 1, ExternSafety::Safe, ["IO"], ffi_read_file);
    vm.register_extern_with("write_file", 2, ExternSafety::Safe, ["IO"], ffi_write_file);
    vm.register_extern_with("append_file", 2, ExternSafety::Safe, ["IO"], ffi_append_file);
    vm.register_extern_with("file_exists", 1, ExternSafety::Safe, ["IO"], ffi_file_exists);
}

fn ffi_print_int(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeMismatch {
            expected: "int (1 arg)",
            found: Value::Unit,
        });
    }
    match args.into_iter().next().unwrap() {
        Value::Int(value) => {
            println!("{value}");
            Ok(Value::Unit)
        }
        other => Err(VmError::TypeMismatch {
            expected: "int",
            found: other,
        }),
    }
}

fn ffi_print_bool(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeMismatch {
            expected: "bool (1 arg)",
            found: Value::Unit,
        });
    }
    match args.into_iter().next().unwrap() {
        Value::Bool(value) => {
            println!("{value}");
            Ok(Value::Unit)
        }
        other => Err(VmError::TypeMismatch {
            expected: "bool",
            found: other,
        }),
    }
}
fn ffi_print(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeMismatch {
            expected: "string (1 arg)",
            found: Value::Unit,
        });
    }
    match args.into_iter().next().unwrap() {
        Value::Str(s) => {
            print!("{s}");
            Ok(Value::Unit)
        }
        other => Err(VmError::TypeMismatch {
            expected: "string",
            found: other,
        }),
    }
}

fn ffi_println(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeMismatch {
            expected: "string (1 arg)",
            found: Value::Unit,
        });
    }
    match args.into_iter().next().unwrap() {
        Value::Str(s) => {
            println!("{s}");
            Ok(Value::Unit)
        }
        other => Err(VmError::TypeMismatch {
            expected: "string",
            found: other,
        }),
    }
}

fn ffi_read_line(_args: Vec<Value>) -> Result<Value, VmError> {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|_| VmError::TypeMismatch {
            expected: "stdin readable",
            found: Value::Unit,
        })?;
    // Remove trailing newline/carriage return
    let input = input.trim_end().to_string();
    Ok(Value::Str(input))
}

fn ffi_read_file(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeMismatch {
            expected: "string (1 arg)",
            found: Value::Unit,
        });
    }
    match args.into_iter().next().unwrap() {
        Value::Str(path) => {
            let content = fs::read_to_string(&path).map_err(|_| VmError::TypeMismatch {
                expected: "readable file",
                found: Value::Str(path.clone()),
            })?;
            Ok(Value::Str(content))
        }
        other => Err(VmError::TypeMismatch {
            expected: "string",
            found: other,
        }),
    }
}

fn ffi_write_file(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 2 {
        return Err(VmError::TypeMismatch {
            expected: "(path, content) (2 args)",
            found: Value::Unit,
        });
    }
    let mut iter = args.into_iter();
    let path = iter.next().unwrap();
    let content = iter.next().unwrap();
    match (path, content) {
        (Value::Str(path), Value::Str(content)) => {
            fs::write(&path, &content).map_err(|_| VmError::TypeMismatch {
                expected: "writable file",
                found: Value::Str(path.clone()),
            })?;
            Ok(Value::Unit)
        }
        (other, _) => Err(VmError::TypeMismatch {
            expected: "string",
            found: other,
        }),
    }
}

fn ffi_append_file(args: Vec<Value>) -> Result<Value, VmError> {
    use std::fs::OpenOptions;
    use std::io::Write;
    
    if args.len() != 2 {
        return Err(VmError::TypeMismatch {
            expected: "(path, content) (2 args)",
            found: Value::Unit,
        });
    }
    let mut iter = args.into_iter();
    let path = iter.next().unwrap();
    let content = iter.next().unwrap();
    match (path, content) {
        (Value::Str(path), Value::Str(content)) => {
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .map_err(|_| VmError::TypeMismatch {
                    expected: "appendable file",
                    found: Value::Str(path.clone()),
                })?;
            file.write_all(content.as_bytes()).map_err(|_| VmError::TypeMismatch {
                expected: "writable file",
                found: Value::Str(path.clone()),
            })?;
            Ok(Value::Unit)
        }
        (other, _) => Err(VmError::TypeMismatch {
            expected: "string",
            found: other,
        }),
    }
}

fn ffi_file_exists(args: Vec<Value>) -> Result<Value, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeMismatch {
            expected: "string (1 arg)",
            found: Value::Unit,
        });
    }
    match args.into_iter().next().unwrap() {
        Value::Str(path) => {
            let exists = Path::new(&path).exists();
            Ok(Value::Bool(exists))
        }
        other => Err(VmError::TypeMismatch {
            expected: "string",
            found: other,
        }),
    }
}
