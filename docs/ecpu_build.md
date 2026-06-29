# eCPU Build Notes

## Windows

```powershell
.\scripts\build_ecpu_windows.ps1
$env:PILM_ECPU_LIB = "D:\self-infer\src\ecpu\build_w4\libecpu.dll"
```

## Linux

```sh
sh scripts/build_ecpu_linux.sh
export PILM_ECPU_LIB="$PWD/src/ecpu/build_linux/libecpu.so"
```

## Raspberry Pi / Portable Linux

Use the portable mode when building on Raspberry Pi or when cross-compiling so
the build does not force the host's `-march=native` flags:

```sh
sh scripts/build_ecpu_linux.sh --raspi --build-dir src/ecpu/build_raspi
export PILM_ECPU_LIB="$PWD/src/ecpu/build_raspi/libecpu.so"
```

The W4A16 path has a scalar fallback and will compile on ARM before NEON-specific
microkernels are added.
