param (
    [int]$Generations = 100,
    [int]$Frames = 12000,
    # Path to a Sonic the Hedgehog (Genesis) ROM used to seed the retro
    # backend's data directory when no ROM has been imported yet. Override via
    # parameter or the SONIC_ROM_SOURCE environment variable.
    [string]$RomSource = $env:SONIC_ROM_SOURCE
)

Write-Host "Starting Sonic LLM Mutator Pipeline Simulation..." -ForegroundColor Cyan

# 0. Load local configuration from .env if present (see .env.example).
#    Accepts plain `KEY=value` and bash-style `export KEY='value'` lines;
#    values already present in the environment are not overwritten.
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment from .env..."
    foreach ($line in Get-Content $envFile) {
        $trimmed = $line.Trim()
        if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -match "^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$") {
            $name = $Matches[1]
            $value = $Matches[2].Trim().Trim("'").Trim('"')
            if (-not (Test-Path "Env:$name")) {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
}

# 1. Activate Environment
if (Test-Path ".\venv38\Scripts\Activate.ps1") {
    Write-Host "Activating Python 3.8 Virtual Environment..."
    . .\venv38\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found! Please run setup first." -ForegroundColor Red
    exit 1
}

# 2. Check if ROM is imported
$retroPath = python -c "import importlib.util; name = 'stable_retro' if importlib.util.find_spec('stable_retro') else 'retro'; module = __import__(name); print(module.__path__[0])"
$retroExitCode = $LASTEXITCODE
if ($retroExitCode -ne 0) {
    Write-Host "Failed to locate the retro backend." -ForegroundColor Red
    exit $retroExitCode
}

$romPath = Join-Path $retroPath "data\stable\SonicTheHedgehog-Genesis\rom.md"

if (-Not (Test-Path $romPath)) {
    Write-Host "Sonic ROM not found in retro backend data directory." -ForegroundColor Yellow
    if ($RomSource -and (Test-Path $RomSource)) {
        Write-Host "Copying ROM from $RomSource..."
        Copy-Item $RomSource $romPath -Force
        Write-Host "ROM copied successfully." -ForegroundColor Green
    } else {
        Write-Host "No ROM source configured. Either import the ROM once with:" -ForegroundColor Red
        Write-Host "    python -m retro.import /path/to/your/roms" -ForegroundColor Red
        Write-Host "or pass -RomSource <path-to-rom> (or set SONIC_ROM_SOURCE)." -ForegroundColor Red
        exit 1
    }
}

# 3. Run Pipeline
Write-Host "Executing Mutator Loop ($Generations generations, $Frames frames/gen)..." -ForegroundColor Cyan
python -u main.py --generations $Generations --frames $Frames
$pipelineExitCode = $LASTEXITCODE
if ($pipelineExitCode -ne 0) {
    Write-Host "Mutator pipeline failed with exit code $pipelineExitCode." -ForegroundColor Red
    exit $pipelineExitCode
}

Write-Host "Pipeline Simulation Complete." -ForegroundColor Green
