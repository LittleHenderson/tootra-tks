import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ALLOWED_EMITS = {"ast", "ir", "bc", "tksi"}


def read_version(repo_root: Path) -> str | None:
    cargo_toml = repo_root / "tks-rs" / "crates" / "tks" / "Cargo.toml"
    if not cargo_toml.exists():
        return None
    for line in cargo_toml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version") and "=" in stripped:
            _, value = stripped.split("=", 1)
            return value.strip().strip('"')
    return None


def find_binary(repo_root: Path, name: str) -> Path | None:
    version = read_version(repo_root)
    if version:
        dist_dir = repo_root / "dist" / f"tks-{version}-windows"
        dist_bin = dist_dir / f"{name}.exe"
        if dist_bin.exists():
            return dist_bin
    for profile in ("release", "debug"):
        candidate = repo_root / "tks-rs" / "target" / profile / f"{name}.exe"
        if candidate.exists():
            return candidate
    return None


def run_command(
    cmd: list[str],
    source: str,
    cwd: Path,
    stdlib_dir: Path | None,
) -> dict:
    env = os.environ.copy()
    if stdlib_dir is not None:
        env["TKS_STDLIB_DIR"] = str(stdlib_dir)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            input=source,
            text=True,
            capture_output=True,
            cwd=str(cwd),
            env=env,
            timeout=10,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "duration_ms": elapsed_ms,
        }
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"command not found: {exc}",
            "exit_code": 127,
            "duration_ms": 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "execution timed out after 10s",
            "exit_code": 124,
            "duration_ms": 10000,
        }


class TksGuiServer(HTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class,
        repo_root: Path,
        tks_path: Path,
        tksc_path: Path,
        stdlib_dir: Path | None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.repo_root = repo_root
        self.tks_path = tks_path
        self.tksc_path = tksc_path
        self.stdlib_dir = stdlib_dir


class TksGuiHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path not in ("/api/validate", "/api/run", "/api/emit"):
            self.send_error(404, "unknown endpoint")
            return
        data = self._read_json()
        if data is None:
            return
        source = data.get("source", "")
        if not isinstance(source, str):
            self._send_json(400, {"ok": False, "stderr": "source must be a string"})
            return

        if self.path == "/api/validate":
            cmd = [str(self.server.tksc_path), "check", "-"]
            payload = run_command(
                cmd, source, self.server.repo_root, self.server.stdlib_dir
            )
            self._send_json(200, payload)
            return

        if self.path == "/api/run":
            ffi = bool(data.get("ffi", False))
            cmd = [str(self.server.tks_path), "run"]
            if ffi:
                cmd.append("--ffi")
            cmd.append("-")
            payload = run_command(
                cmd, source, self.server.repo_root, self.server.stdlib_dir
            )
            self._send_json(200, payload)
            return

        if self.path == "/api/emit":
            kind = data.get("kind", "bc")
            if kind not in ALLOWED_EMITS:
                self._send_json(400, {"ok": False, "stderr": "invalid emit kind"})
                return
            cmd = [str(self.server.tksc_path), "build", "--emit", kind, "-"]
            payload = run_command(
                cmd, source, self.server.repo_root, self.server.stdlib_dir
            )
            self._send_json(200, payload)
            return

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "stderr": "invalid JSON"})
            return None

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="TKS GUI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8747)
    parser.add_argument("--tks", default="")
    parser.add_argument("--tksc", default="")
    parser.add_argument("--stdlib", default="")
    parser.add_argument("--static", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    static_dir = Path(args.static) if args.static else Path(__file__).parent / "static"
    if not static_dir.exists():
        print(f"static directory missing: {static_dir}", file=sys.stderr)
        return 2

    tks_path = Path(args.tks) if args.tks else find_binary(repo_root, "tks")
    tksc_path = Path(args.tksc) if args.tksc else find_binary(repo_root, "tksc")
    if tks_path is None or not tks_path.exists():
        print("tks.exe not found. Build with scripts/package_tks_dist.ps1.", file=sys.stderr)
        return 2
    if tksc_path is None or not tksc_path.exists():
        print("tksc.exe not found. Build with scripts/package_tks_dist.ps1.", file=sys.stderr)
        return 2

    stdlib_dir = Path(args.stdlib) if args.stdlib else repo_root / "tks-rs" / "stdlib"
    if not stdlib_dir.exists():
        stdlib_dir = None

    handler = partial(TksGuiHandler, directory=str(static_dir))
    server = TksGuiServer(
        (args.host, args.port),
        handler,
        repo_root,
        tks_path,
        tksc_path,
        stdlib_dir,
    )
    print(f"TKS GUI running at http://{args.host}:{args.port}")
    print(f"tks:  {tks_path}")
    print(f"tksc: {tksc_path}")
    if stdlib_dir:
        print(f"stdlib: {stdlib_dir}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
