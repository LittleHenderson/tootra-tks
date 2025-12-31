use std::fs;
use std::path::{Path, PathBuf};

use tksbytecode::emit::emit;
use tkscore::parser::parse_program;
use tksir::lower::lower_program;
use tksvm::vm::{Value, VmError, VmState};
use tkstypes::infer::infer_program;

fn canonical_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("canonical")
}

fn run_sample(path: &Path) -> Result<String, String> {
    let source = fs::read_to_string(path)
        .map_err(|err| format!("read {}: {err}", path.display()))?;
    let program = parse_program(&source)
        .map_err(|err| format!("parse error: {}", err.message))?;
    infer_program(&program).map_err(|err| format!("type error: {err}"))?;
    let ir = lower_program(&program).map_err(|err| format!("lower error: {err}"))?;
    let bytecode = emit(&ir).map_err(|err| format!("emit error: {err}"))?;
    let mut vm = VmState::new(bytecode);
    register_externs(&mut vm);
    let result = vm.run().map_err(|err| format!("vm error: {err:?}"))?;
    Ok(format!("{result:?}"))
}

fn register_externs(vm: &mut VmState) {
    vm.register_extern("ping", 1, |args: Vec<Value>| match args.as_slice() {
        [Value::Int(value)] => Ok(Value::Int(*value)),
        [other] => Err(VmError::TypeMismatch {
            expected: "int",
            found: other.clone(),
        }),
        _ => Err(VmError::TypeMismatch {
            expected: "one arg",
            found: Value::Unit,
        }),
    });
}

#[test]
fn canonical_samples() {
    let dir = canonical_dir();
    let mut samples: Vec<PathBuf> = fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("read {}: {err}", dir.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("tks"))
        .collect();
    samples.sort();
    assert!(
        !samples.is_empty(),
        "no canonical samples found in {}",
        dir.display()
    );

    for sample in samples {
        let expected_path = sample.with_extension("expect");
        let expected = fs::read_to_string(&expected_path).unwrap_or_else(|err| {
            panic!(
                "read {}: {err}",
                expected_path.display()
            )
        });
        let expected = expected.trim();
        let got = run_sample(&sample)
            .unwrap_or_else(|err| panic!("{}: {err}", sample.display()));
        assert_eq!(got, expected, "mismatch for {}", sample.display());
    }
}
