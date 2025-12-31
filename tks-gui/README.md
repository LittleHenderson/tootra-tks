# TKS Equation Lab (GUI)

This is a local GUI for validating and running TKS code. It shells out to the
existing `tks.exe` and `tksc.exe` binaries.

## Step-by-step

1) Build the binaries (CPU):
```powershell
cd C:\Users\wakil\downloads\everthing-tootra-tks
.\scripts\package_tks_dist.ps1 -Configuration Release
```

2) Start the GUI server:
```powershell
python .\tks-gui\server.py
```

3) Open the GUI in your browser:
```
http://127.0.0.1:8747
```

## GPU build (optional)

```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release -GpuOnly
python .\tks-gui\server.py --tks .\dist\tks-0.1.0-windows-gpu\tks.exe
```

## Command-line options

- `--host 127.0.0.1`
- `--port 8747`
- `--tks <path to tks.exe>`
- `--tksc <path to tksc.exe>`
- `--stdlib <path to tks-rs\stdlib>`
- `--static <path to static assets>`

Example:
```powershell
python .\tks-gui\server.py --tks .\dist\tks-0.1.0-windows\tks.exe --tksc .\dist\tks-0.1.0-windows\tksc.exe
```

## Notes

- Validate uses `tksc check`.
- Run uses `tks run`.
- `tks run` does not resolve modules yet; use `tksc check` for module validation.
- The GUI can save/load snippets and projects in local storage.
- Run Bytecode compiles to `.tkso` and executes the bytecode in one step.
