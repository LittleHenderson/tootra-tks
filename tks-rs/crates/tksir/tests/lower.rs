use tkscore::parser::parse_program;
use tksir::lower::lower_program;

#[test]
fn lower_simple_program() {
    let source = r#"
let x = 1;
let f = \y -> y;
f(x)
"#;
    let program = parse_program(source).expect("parse program");
    lower_program(&program).expect("lower program");
}
