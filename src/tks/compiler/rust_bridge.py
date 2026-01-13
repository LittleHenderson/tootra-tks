"""
TKS Rust Compiler Bridge - FFI interface to tks-rs

This module provides a Python interface to the Rust TKS compiler (tks-rs),
enabling proof-based validation instead of heuristic guessing.

The bridge supports two modes:
    1. Subprocess: Call the compiled tks/tksc binary
    2. FFI (future): Direct library calls via ctypes/cffi when available

Key capabilities:
    - Expression parsing and validation
    - Type checking and inference
    - Bytecode compilation
    - VM execution

Usage:
    from tks.compiler import TKSCompiler

    compiler = TKSCompiler()

    # Validate an expression
    result = compiler.validate("1+2*3")
    if result.valid:
        print(f"Valid! Type: {result.type_info}")
    else:
        print(f"Error: {result.errors}")

    # Compile and run
    bytecode = compiler.compile("let x = 42; x")
    output = compiler.run(bytecode)
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import shutil


class CompilerMode(Enum):
    """Compiler operation mode."""
    SUBPROCESS = "subprocess"  # Call tks binary via subprocess
    FFI = "ffi"                # Direct library calls (future)
    MOCK = "mock"              # Mock mode for testing


@dataclass
class ParseResult:
    """Result of parsing a TKS expression."""
    valid: bool
    ast: Optional[Dict] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    span: Optional[Dict] = None


@dataclass
class TypeCheckResult:
    """Result of type checking."""
    valid: bool
    type_info: Optional[str] = None
    inferred_types: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class CompileResult:
    """Result of compilation."""
    success: bool
    bytecode: Optional[bytes] = None
    bytecode_path: Optional[Path] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of VM execution."""
    success: bool
    output: Any = None
    return_value: Any = None
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class TKSCompiler:
    """
    Python interface to the TKS Rust compiler.

    This class wraps the tks-rs compiler, providing:
    - Expression validation (parse + type check)
    - Full compilation to bytecode
    - VM execution
    - Error reporting

    The compiler automatically finds the Rust binary or falls back to mock mode.
    """

    def __init__(
        self,
        tks_binary: Optional[Path] = None,
        tksc_binary: Optional[Path] = None,
        mode: CompilerMode = CompilerMode.SUBPROCESS,
        rust_project_root: Optional[Path] = None,
    ):
        """
        Initialize the TKS compiler bridge.

        Args:
            tks_binary: Path to the `tks` binary (REPL/runner)
            tksc_binary: Path to the `tksc` binary (compiler)
            mode: Operation mode
            rust_project_root: Root of tks-rs project
        """
        self.mode = mode
        self.rust_project_root = rust_project_root or self._find_rust_project()

        # Find binaries
        self.tks_binary = tks_binary or self._find_binary("tks")
        self.tksc_binary = tksc_binary or self._find_binary("tksc")

        # Track if we have real compiler access
        self.compiler_available = self.tksc_binary is not None and self.tksc_binary.exists()
        self.vm_available = self.tks_binary is not None and self.tks_binary.exists()

        if not self.compiler_available and mode != CompilerMode.MOCK:
            # Fall back to mock mode if no compiler found
            self.mode = CompilerMode.MOCK

    def _find_rust_project(self) -> Optional[Path]:
        """Find the tks-rs project root."""
        # Try relative paths from common locations
        search_paths = [
            Path(__file__).parent.parent.parent.parent / "tks-rs",  # From src/tks/compiler
            Path.cwd() / "tks-rs",
            Path.cwd().parent / "tks-rs",
        ]

        for path in search_paths:
            if path.exists() and (path / "Cargo.toml").exists():
                return path

        return None

    def _find_binary(self, name: str) -> Optional[Path]:
        """Find a compiled Rust binary."""
        if not self.rust_project_root:
            return None

        # Check common binary locations
        search_paths = [
            self.rust_project_root / "target" / "release" / name,
            self.rust_project_root / "target" / "release" / f"{name}.exe",
            self.rust_project_root / "target" / "debug" / name,
            self.rust_project_root / "target" / "debug" / f"{name}.exe",
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Try system PATH
        system_path = shutil.which(name)
        if system_path:
            return Path(system_path)

        return None

    def _run_subprocess(
        self,
        binary: Path,
        args: List[str],
        input_text: Optional[str] = None,
        timeout: int = 30,
    ) -> Tuple[int, str, str]:
        """Run a subprocess and capture output."""
        try:
            # Build full command
            cmd = [str(binary)] + args

            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,  # Explicitly no shell to avoid arg splitting issues
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout expired"
        except FileNotFoundError:
            return -2, "", f"Binary not found: {binary}"
        except Exception as e:
            return -3, "", str(e)

    def parse(self, source: str) -> ParseResult:
        """
        Parse a TKS expression/program.

        Args:
            source: TKS source code

        Returns:
            ParseResult with AST or errors
        """
        if self.mode == CompilerMode.MOCK:
            return self._mock_parse(source)

        if not self.compiler_available:
            return ParseResult(
                valid=False,
                errors=["Compiler not available - using mock mode"]
            )

        # Write source to temp file and parse
        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False, mode="w") as f:
            f.write(source)
            temp_path = f.name

        try:
            # Run compiler in check mode (validates parsing + types)
            code, stdout, stderr = self._run_subprocess(
                self.tksc_binary,
                ["check", temp_path],
            )

            if code == 0:
                return ParseResult(valid=True, ast={"raw": stdout.strip()} if stdout.strip() else None)
            else:
                errors = []
                # Parse error output
                error_text = stderr.strip() or stdout.strip()
                if error_text:
                    errors = [line for line in error_text.split("\n") if line.strip()]
                return ParseResult(valid=False, errors=errors or ["Parse/check failed"])
        finally:
            os.unlink(temp_path)

    def _mock_parse(self, source: str) -> ParseResult:
        """Mock parser for testing without Rust compiler."""
        # Simple heuristic validation
        errors = []
        warnings = []

        # Check for basic syntax issues
        if source.count("(") != source.count(")"):
            errors.append("Mismatched parentheses")
        if source.count("[") != source.count("]"):
            errors.append("Mismatched brackets")
        if source.count("{") != source.count("}"):
            errors.append("Mismatched braces")

        # Check for TKS-specific syntax
        tks_operators = [":", "^", "+", "-", "*", "/", "→", "⊕", "⊗"]
        has_tks_syntax = any(op in source for op in tks_operators)

        if errors:
            return ParseResult(valid=False, errors=errors)

        return ParseResult(
            valid=True,
            ast={"type": "mock", "source": source, "tks_syntax": has_tks_syntax},
            warnings=["Using mock parser - results may not match Rust compiler"]
        )

    def type_check(self, source: str) -> TypeCheckResult:
        """
        Type check a TKS expression/program.

        Args:
            source: TKS source code

        Returns:
            TypeCheckResult with type info or errors
        """
        if self.mode == CompilerMode.MOCK:
            return self._mock_type_check(source)

        if not self.compiler_available:
            return TypeCheckResult(
                valid=False,
                errors=["Compiler not available"]
            )

        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False, mode="w") as f:
            f.write(source)
            temp_path = f.name

        try:
            # tksc check does both parsing and type checking
            code, stdout, stderr = self._run_subprocess(
                self.tksc_binary,
                ["check", temp_path],
            )

            if code == 0:
                # Type check passed - tksc doesn't output type info, just validates
                return TypeCheckResult(
                    valid=True,
                    type_info="checked",
                )
            else:
                errors = []
                error_text = stderr.strip() or stdout.strip()
                if error_text:
                    errors = [line for line in error_text.split("\n") if line.strip()]
                return TypeCheckResult(
                    valid=False,
                    errors=errors or ["Type check failed"]
                )
        finally:
            os.unlink(temp_path)

    def _mock_type_check(self, source: str) -> TypeCheckResult:
        """Mock type checker."""
        # Simple heuristic type inference
        if source.strip().isdigit():
            return TypeCheckResult(valid=True, type_info="Int")
        if source.strip().replace(".", "").isdigit():
            return TypeCheckResult(valid=True, type_info="Float")
        if source.startswith('"') and source.endswith('"'):
            return TypeCheckResult(valid=True, type_info="String")
        if source.lower() in ("true", "false"):
            return TypeCheckResult(valid=True, type_info="Bool")

        return TypeCheckResult(
            valid=True,
            type_info="Unknown",
            errors=["Using mock type checker - type inference limited"]
        )

    def compile(self, source: str, output_path: Optional[Path] = None) -> CompileResult:
        """
        Compile TKS source to bytecode.

        Args:
            source: TKS source code
            output_path: Optional path for .tkso output

        Returns:
            CompileResult with bytecode or errors
        """
        if self.mode == CompilerMode.MOCK:
            return self._mock_compile(source)

        if not self.compiler_available:
            return CompileResult(success=False, errors=["Compiler not available"])

        with tempfile.NamedTemporaryFile(suffix=".tks", delete=False, mode="w") as f:
            f.write(source)
            temp_path = f.name

        out_path = output_path or Path(temp_path).with_suffix(".tkso")

        try:
            # Use tksc build command
            code, stdout, stderr = self._run_subprocess(
                self.tksc_binary,
                ["build", temp_path, "-o", str(out_path)],
            )

            if code == 0 and out_path.exists():
                bytecode = out_path.read_bytes()
                return CompileResult(
                    success=True,
                    bytecode=bytecode,
                    bytecode_path=out_path,
                )
            else:
                errors = []
                error_text = stderr.strip() or stdout.strip()
                if error_text:
                    errors = [line for line in error_text.split("\n") if line.strip()]
                return CompileResult(
                    success=False,
                    errors=errors or ["Compilation failed"]
                )
        finally:
            os.unlink(temp_path)
            if output_path is None and out_path.exists():
                os.unlink(out_path)

    def _mock_compile(self, source: str) -> CompileResult:
        """Mock compiler."""
        parse_result = self._mock_parse(source)
        if not parse_result.valid:
            return CompileResult(success=False, errors=parse_result.errors)

        # Return mock bytecode
        mock_bytecode = b"TKSO\x00\x01" + source.encode("utf-8")
        return CompileResult(
            success=True,
            bytecode=mock_bytecode,
            warnings=["Using mock compiler - bytecode is not executable"]
        )

    def run(
        self,
        bytecode: Optional[bytes] = None,
        bytecode_path: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> RunResult:
        """
        Run bytecode or source in the TKS VM.

        Args:
            bytecode: Bytecode bytes
            bytecode_path: Path to .tkso file
            source: TKS source (will be compiled first)

        Returns:
            RunResult with output or errors
        """
        if source:
            compile_result = self.compile(source)
            if not compile_result.success:
                return RunResult(success=False, errors=compile_result.errors)
            bytecode = compile_result.bytecode

        if self.mode == CompilerMode.MOCK:
            return self._mock_run(bytecode)

        if not self.vm_available:
            return RunResult(success=False, errors=["VM not available"])

        # Write bytecode to temp file
        with tempfile.NamedTemporaryFile(suffix=".tkso", delete=False, mode="wb") as f:
            f.write(bytecode)
            temp_path = f.name

        try:
            import time
            start = time.time()

            code, stdout, stderr = self._run_subprocess(
                self.tks_binary,
                ["run", temp_path],
            )

            elapsed = (time.time() - start) * 1000

            if code == 0:
                return RunResult(
                    success=True,
                    output=stdout.strip(),
                    return_value=stdout.strip(),
                    execution_time_ms=elapsed,
                )
            else:
                return RunResult(
                    success=False,
                    errors=[stderr.strip()] if stderr else ["Execution failed"],
                    execution_time_ms=elapsed,
                )
        finally:
            os.unlink(temp_path)

    def _mock_run(self, bytecode: Optional[bytes]) -> RunResult:
        """Mock VM execution."""
        if not bytecode:
            return RunResult(success=False, errors=["No bytecode provided"])

        # Extract source from mock bytecode
        if bytecode.startswith(b"TKSO\x00\x01"):
            source = bytecode[6:].decode("utf-8", errors="ignore")
            # Simple eval for basic expressions
            try:
                result = eval(source, {"__builtins__": {}})
                return RunResult(success=True, output=str(result), return_value=result)
            except:
                pass

        return RunResult(
            success=True,
            output="(mock execution)",
            errors=["Using mock VM - actual execution not performed"]
        )

    def validate(self, expression: str) -> ParseResult:
        """
        Validate a TKS expression (convenience method).

        This is the primary method for the Verifier module to use.

        Args:
            expression: TKS expression string

        Returns:
            ParseResult indicating validity
        """
        # First parse
        parse_result = self.parse(expression)
        if not parse_result.valid:
            return parse_result

        # Then type check
        type_result = self.type_check(expression)
        if not type_result.valid:
            return ParseResult(
                valid=False,
                ast=parse_result.ast,
                errors=type_result.errors,
            )

        return ParseResult(
            valid=True,
            ast=parse_result.ast,
            warnings=parse_result.warnings + [f"Type: {type_result.type_info}"],
        )

    @property
    def status(self) -> Dict[str, Any]:
        """Get compiler bridge status."""
        return {
            "mode": self.mode.value,
            "compiler_available": self.compiler_available,
            "vm_available": self.vm_available,
            "tks_binary": str(self.tks_binary) if self.tks_binary else None,
            "tksc_binary": str(self.tksc_binary) if self.tksc_binary else None,
            "rust_project_root": str(self.rust_project_root) if self.rust_project_root else None,
        }


# Singleton instance
_compiler: Optional[TKSCompiler] = None


def get_compiler() -> TKSCompiler:
    """Get the global compiler instance."""
    global _compiler
    if _compiler is None:
        _compiler = TKSCompiler()
    return _compiler


__all__ = [
    "CompilerMode",
    "ParseResult",
    "TypeCheckResult",
    "CompileResult",
    "RunResult",
    "TKSCompiler",
    "get_compiler",
]
