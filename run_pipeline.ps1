param (
    [int]$Generations = 100,
    [int]$Frames = 12000
)

Write-Host "Starting Sonic LLM Mutator Pipeline Simulation..." -ForegroundColor Cyan

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
$romPath = Join-Path $retroPath "data\stable\SonicTheHedgehog-Genesis\rom.md"

if (-Not (Test-Path $romPath)) {
    Write-Host "Sonic ROM not found in retro backend data directory." -ForegroundColor Yellow
    Write-Host "Attempting to copy from Downloads..."
    $sourceRom = "C:\Users\ericr\Downloads\Sonic_The_Hedgehog_W_REV01_h3C.bin"
    if (Test-Path $sourceRom) {
        Copy-Item $sourceRom $romPath -Force
        Write-Host "ROM copied successfully." -ForegroundColor Green
    } else {
        Write-Host "Source ROM not found at $sourceRom. Pipeline cannot continue." -ForegroundColor Red
        exit 1
    }
}

# 3. Run Pipeline
Write-Host "Executing Mutator Loop ($Generations generations, $Frames frames/gen)..." -ForegroundColor Cyan
python -u main.py --generations $Generations --frames $Frames

Write-Host "Pipeline Simulation Complete." -ForegroundColor Green
