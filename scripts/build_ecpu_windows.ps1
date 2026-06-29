param(
    [string]$BuildDir = "src\ecpu\build_w4",
    [switch]$Portable,
    [int]$Jobs = 8
)

$nativeArch = if ($Portable) { "OFF" } else { "ON" }
cmake -S src\ecpu -B $BuildDir -G "MinGW Makefiles" `
    -DCMAKE_BUILD_TYPE=Release `
    "-DECPU_NATIVE_ARCH=$nativeArch"
cmake --build $BuildDir --config Release -j $Jobs

$dll = Join-Path $BuildDir "libecpu.dll"
Write-Host "Built: $dll"
Write-Host "Use with: `$env:PILM_ECPU_LIB='$((Resolve-Path $dll).Path)'"
