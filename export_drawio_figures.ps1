$ErrorActionPreference='Stop'
param([string]$DrawioExecutable='E:\画图\draw.io\draw.io.exe')
$workspaceRoot=$PSScriptRoot
$imageFolder=Join-Path $workspaceRoot 'figures'
$backupFolder=Join-Path $workspaceRoot 'records\local_backups\drawio'
New-Item -ItemType Directory -Path $backupFolder -Force | Out-Null
foreach($figureNumber in 1,2){
    $sourceFile=Join-Path $imageFolder ('Figure '+$figureNumber+'.drawio')
    foreach($extension in 'png','svg'){
        $outputFile=Join-Path $imageFolder ('Figure '+$figureNumber+'.'+$extension)
        if((Test-Path -LiteralPath $outputFile) -and !(Test-Path -LiteralPath (Join-Path $backupFolder ([IO.Path]::GetFileName($outputFile))))){
            Copy-Item -LiteralPath $outputFile -Destination $backupFolder
        }
        $exportArgs=@('--export','--format',$extension,'--border','20','--output',('"'+$outputFile+'"'),('"'+$sourceFile+'"'))
        if($extension -eq 'png'){$exportArgs=@('--export','--format','png','--scale','4','--border','20','--output',('"'+$outputFile+'"'),('"'+$sourceFile+'"'))}
        $process=Start-Process -FilePath $DrawioExecutable -ArgumentList $exportArgs -WindowStyle Hidden -PassThru -Wait
        if($process.ExitCode -ne 0){throw ('draw.io export failed: '+$outputFile)}
        if(!(Test-Path -LiteralPath $outputFile)){throw ('Missing export: '+$outputFile)}
        Get-Item -LiteralPath $outputFile | Select-Object Name,Length
    }
}
