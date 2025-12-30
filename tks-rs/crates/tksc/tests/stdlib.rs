use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn stdlib_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("stdlib")
}

fn temp_tks_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let mut path = std::env::temp_dir();
    path.push(format!("tksc-{tag}-{nanos}.tks"));
    path
}

fn run_tksc(args: &[&str], stdlib: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tksc"))
        .args(args)
        .env("TKS_STDLIB_DIR", stdlib)
        .output()
        .expect("run tksc")
}

fn format_output(output: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

#[test]
fn tksc_check_with_stdlib_imports() {
    let stdlib = stdlib_dir();
    assert!(stdlib.exists(), "missing stdlib at {}", stdlib.display());
    let source = r#"
module App {
  from TKS.Core import { id, const, unit };
  let one = id 1;
  let two = const 1 2;
  let u = unit;
}
"#;
    let path = temp_tks_path("check");
    fs::write(&path, source).expect("write source");
    let path_str = path.to_string_lossy().to_string();
    let output = run_tksc(&["check", &path_str], &stdlib);
    let (stdout, stderr) = format_output(&output);
    let _ = fs::remove_file(&path);
    assert!(
        output.status.success(),
        "tksc check failed\nstdout: {stdout}\nstderr: {stderr}"
    );
}

#[test]
fn tksc_tksi_filters_stdlib() {
    let stdlib = stdlib_dir();
    assert!(stdlib.exists(), "missing stdlib at {}", stdlib.display());
    let source = r#"
module App {
  export { foo }
  let foo = 1;
}
"#;
    let path = temp_tks_path("tksi");
    fs::write(&path, source).expect("write source");
    let path_str = path.to_string_lossy().to_string();
    let output = run_tksc(&["build", &path_str, "--emit", "tksi"], &stdlib);
    let (stdout, stderr) = format_output(&output);
    let _ = fs::remove_file(&path);
    assert!(
        output.status.success(),
        "tksc build failed\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(stdout.contains("module App"));
    assert!(!stdout.contains("module TKS.Core"));
    assert!(!stdout.contains("module TKS.RPM"));
    assert!(!stdout.contains("module TKS.Quantum"));
}
