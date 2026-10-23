$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python"
}

$runId = [guid]::NewGuid().ToString("N")
$stdoutPath = Join-Path $env:TEMP "jobradar-embedding-$runId.out"
$stderrPath = Join-Path $env:TEMP "jobradar-embedding-$runId.err"

Push-Location $PSScriptRoot
try {
    while ($true) {
        Write-Host "`nStarting embedding backfill at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

        $process = Start-Process `
            -FilePath $python `
            -ArgumentList @("-m", "app.embeddings", "--entity", "all", "--batch-size", "1") `
            -WorkingDirectory $PSScriptRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $stdout = if (Test-Path -LiteralPath $stdoutPath) {
            Get-Content -LiteralPath $stdoutPath -Raw
        } else {
            ""
        }
        $stderr = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath -Raw
        } else {
            ""
        }

        if ($stdout) { Write-Host $stdout }
        if ($stderr) { Write-Host $stderr }

        $outputText = "$stdout`n$stderr"

        if ($process.ExitCode -eq 0) {
            Write-Host "Embedding backfill completed successfully."
            break
        }

        if ($outputText -match '(?i)(HTTP\s*429|429\s+Too Many Requests|exceeded your current quota)') {
            Write-Host "Gemini returned HTTP 429. Waiting 2 minutes before retrying..."
            Start-Sleep -Seconds 120
            continue
        }

        Write-Host "Backfill failed with exit code $($process.ExitCode); stopping because this was not a recognized 429 quota error." -ForegroundColor Red
        exit $process.ExitCode
    }
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
}