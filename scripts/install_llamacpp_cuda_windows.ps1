#!/usr/bin/env pwsh
# Rebuild llama-cpp-python against NVIDIA CUDA so llama_supports_gpu_offload() is True.
#
# Prerequisites (install separately):
#   - NVIDIA GPU driver
#   - CUDA Toolkit (12.x matches PyTorch cu124 in this repo; align toolkit + PATH)
#   - Visual Studio Build Tools with "Desktop development with C++" (MSVC + Windows SDK + CMake)
#
# Usage (from repo root):  .\scripts\install_llamacpp_cuda_windows.ps1

$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Error 'PowerShell 5.1 or newer required.'
}

Write-Host @'
CUDA rebuild for llama-cpp-python - this commonly takes several minutes.

If CMake cannot find CUDA, use "x64 Native Tools Command Prompt for VS" or "Developer
PowerShell for VS", and ensure CUDA Toolkit "bin" is on PATH (nvcc --version works).
'@

# Help CMake find nvcc (fixes "No CUDA toolset found" with Visual Studio generators).
function Add-CudaCompilerToCmakeArgs {
    $nvcc = $null
    if ($env:CUDA_PATH) {
        $candidate = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
        if (Test-Path $candidate) {
            $nvcc = $candidate
        }
    }
    if (-not $nvcc) {
        $roots = @(
            'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
            'C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA'
        )
        foreach ($root in $roots) {
            if (-not (Test-Path $root)) {
                continue
            }
            foreach ($verDir in (Get-ChildItem -Path $root -Directory | Sort-Object Name -Descending)) {
                $candidate = Join-Path $verDir.FullName 'bin\nvcc.exe'
                if (Test-Path $candidate) {
                    $nvcc = $candidate
                    break
                }
            }
            if ($nvcc) {
                break
            }
        }
    }
    $pathNvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if (-not $nvcc -and $pathNvcc) {
        $nvcc = $pathNvcc.Source
    }
    if ($nvcc) {
        $cudaBin = Split-Path $nvcc -Parent
        $env:PATH = "$cudaBin;$env:PATH"
        if ($env:CUDA_PATH) {
            # keep
        } elseif (Test-Path (Join-Path (Split-Path $cudaBin -Parent) 'include')) {
            $env:CUDA_PATH = Split-Path $cudaBin -Parent
        }
        # Paths with spaces must be quoted or CMake splits them into garbage tokens.
        $cudaFwd = $nvcc.Replace('\', '/')
        $flag = "-DCMAKE_CUDA_COMPILER=`"$cudaFwd`""
        if ($env:CMAKE_ARGS -notmatch 'CMAKE_CUDA_COMPILER') {
            $env:CMAKE_ARGS = "$($env:CMAKE_ARGS) $flag".Trim()
        }
        Write-Host "Using CUDA compiler: $nvcc"
    } else {
        Write-Warning 'nvcc.exe not found. Install CUDA Toolkit or add its bin directory to PATH.'
    }
}

function Import-VcVars64Environment {
    <#
      Ninja + MSVC needs cl.exe/link.exe on PATH. Import env from vcvars64.bat via vswhere.
      Prefer Visual Studio 2022 (17.x): CUDA 12.1 host_config.h rejects VS 2025/2026 (18.x).
    #>
    $vswhere = Join-Path (${env:ProgramFiles(x86)}) 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        Write-Warning "vswhere not found. Open 'x64 Native Tools Command Prompt for VS' and run this script there."
        return
    }

    $prefers22 = & $vswhere @(
        '-version', '[17.0,18.0)',
        '-products', '*',
        '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
        '-property', 'installationPath'
    ) | Select-Object -First 1

    if ($prefers22) {
        $installPath = $prefers22.Trim()
        Write-Host "Using Visual Studio 2022 (or 17.x) host for nvcc: $installPath"
    } else {
        $installPath = & $vswhere @(
            '-latest',
            '-products', '*',
            '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property', 'installationPath'
        )
        if (-not $installPath) {
            Write-Warning 'Visual Studio C++ toolchain not found (vswhere returned empty).'
            return
        }
        $installPath = $installPath.Trim()
        Write-Host "Using latest Visual Studio install: $installPath"
        Write-Warning (
            'CUDA 12.x typically rejects MSVC from VS 2025/2026. Install "Build Tools for Visual Studio 2022" ' +
            '(C++ workload) or upgrade CUDA to 12.8+, then rerun. Or use an x64 Native Tools prompt for VS 2022.'
        )
    }

    $vcvars = Join-Path $installPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) {
        Write-Warning "Missing $vcvars"
        return
    }

    Write-Host "Loading MSVC env: $vcvars"
    cmd.exe /v:on /c "call `"$vcvars`" >nul 2>nul && set" | ForEach-Object {
        $parts = $_ -split '=', 2
        if ($parts.Count -eq 2) {
            $name = $parts[0]
            $value = $parts[1]
            Set-Item -LiteralPath ('env:{0}' -f $name) -Value $value
        }
    }
}

# Force a CMake configure + build (avoid stale CPU-only binaries).
$env:FORCE_CMAKE = '1'

# GPU backend flag (must be present before optional CMAKE_CUDA_COMPILER append).
if (-not $env:CMAKE_ARGS) {
    $env:CMAKE_ARGS = '-DGGML_CUDA=ON'
}

# CUDA toolkit often lags MSVC: VS 2025 + CUDA 12.1 hits host_config.h "unsupported ... version".
if ($env:CMAKE_ARGS -notmatch 'allow-unsupported-compiler') {
    $env:CMAKE_ARGS = "$($env:CMAKE_ARGS) -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler".Trim()
}

Add-CudaCompilerToCmakeArgs

# Visual Studio + CUDA often hits "No CUDA toolset found"; Ninja + nvcc works without VS CUDA integration.
Write-Host 'Ensuring Ninja is available for CMake (pip install ninja)...'
uv pip install --quiet ninja
$venvRoot = uv run python -c "import sys; print(sys.prefix)"
$ninjaDir = Join-Path $venvRoot 'Scripts'
if (Test-Path (Join-Path $ninjaDir 'ninja.exe')) {
    $env:PATH = "$ninjaDir;$env:PATH"
}
if ($env:CMAKE_ARGS -notmatch '(^|\s)-G\s') {
    $env:CMAKE_ARGS = "-G Ninja $($env:CMAKE_ARGS)".Trim()
    Write-Host 'Using CMake generator: Ninja'
} else {
    Write-Host 'CMake -G already set in CMAKE_ARGS'
}

Import-VcVars64Environment

Write-Host "CMAKE_ARGS=$($env:CMAKE_ARGS)"
Write-Host "Running: uv pip install --force-reinstall --no-cache-dir llama-cpp-python`n"

uv pip install --force-reinstall --no-cache-dir llama-cpp-python
if ($LASTEXITCODE -ne 0) {
    Write-Error "llama-cpp-python build failed (exit $LASTEXITCODE). See CMake output above."
}

Write-Host "`nVerify offload support:"
uv run python -c "import llama_cpp; print('llama_supports_gpu_offload()', llama_cpp.llama_supports_gpu_offload())"

Write-Host @"

If False: open a VS Developer shell, confirm nvcc/CUDA Toolkit, then rerun.
After plain 'uv sync', PyPI may reinstall a CPU wheel - rerun this script to re-enable CUDA.
"@
