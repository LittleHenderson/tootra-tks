# Packaging (Windows)

This project ships two executables:

- `tks.exe` runs `.tks` source or `.tkso` bytecode (`tks run ...`).
- `tksc.exe` is the compiler CLI for check/build (`tksc check|build ...`).

## Packaging Scripts

Run these from the repo root in PowerShell.

### Stage a single executable

```powershell
.\scripts\package_tks.ps1 -Configuration Release -OutDir .\packaging\windows
.\scripts\package_tksc.ps1 -Configuration Release -OutDir .\packaging\windows
```

GPU-enabled tks build:

```powershell
.\scripts\package_tks.ps1 -Configuration Release -OutDir .\packaging\windows -Gpu
```

### Stage both tks.exe + tksc.exe

```powershell
.\scripts\package_tks_bundle.ps1 -Configuration Release -OutDir .\packaging\windows
```

### Create a dist bundle + zip

```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release -OutDir .\dist
```

By default, the script reads the version from `tks-rs/crates/tks/Cargo.toml` and writes:

- `dist/tks-<version>-windows/` (staged binaries)
- `dist/tks-<version>-windows.zip`

You can override the version:

```powershell
.\scripts\package_tks_dist.ps1 -Version 0.1.0
```

## GPU Packaging

To build `tks.exe` with GPU support (enables `tks gpu ...` subcommands), pass `-Gpu`:

```powershell
.\scripts\package_tks_dist.ps1 -Configuration Release -Gpu
```

Notes:
- `-Gpu` only affects `tks.exe` (tksc has no GPU feature).
- GPU builds require the `gpu` feature and will pull in extra dependencies.
