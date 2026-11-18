$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Model contract checks failed' }
if (-not (Test-Path -LiteralPath '..\数据记录\reference_masks\per_image_class.csv')) {
    python build_reference.py --output '..\数据记录\reference_masks'
    if ($LASTEXITCODE -ne 0) { throw 'Mask record generation failed' }
}
python verify_masks.py --directory '..\数据记录\reference_masks'
if ($LASTEXITCODE -ne 0) { throw 'Mask verification failed' }
python generate_curves.py
if ($LASTEXITCODE -ne 0) { throw 'Epoch profile export failed' }
python generate_figures.py
if ($LASTEXITCODE -ne 0) { throw 'Figure export failed' }
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
foreach ($variant in @('baseline','adapter','text','full')) {
    $runOutput = '..\数据记录\pipeline_' + $variant + '_' + $runStamp
    python train.py --mode smoke --device cpu --variant $variant --output $runOutput
    if ($LASTEXITCODE -ne 0) { throw ('Training check failed: ' + $variant) }
    python verify_run.py --directory $runOutput
    if ($LASTEXITCODE -ne 0) { throw ('Checkpoint verification failed: ' + $variant) }
}
