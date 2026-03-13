param(
    [int[]]$Ports = @(8000, 8001, 5173, 5174)
)

$killedPids = @{}

foreach ($port in $Ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Output "Port ${port}: no listening process found."
        continue
    }

    $pidsForPort = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pidValue in $pidsForPort) {
        if ($killedPids.ContainsKey($pidValue)) {
            Write-Output "Port ${port}: PID $pidValue already terminated (shared across ports)."
            continue
        }

        try {
            $processName = (Get-Process -Id $pidValue -ErrorAction SilentlyContinue).ProcessName
            Stop-Process -Id $pidValue -Force -ErrorAction Stop
            $killedPids[$pidValue] = $true
            if ($processName) {
                Write-Output "Port ${port}: terminated PID $pidValue ($processName)."
            } else {
                Write-Output "Port ${port}: terminated PID $pidValue."
            }
        } catch {
            Write-Output "Port ${port}: failed to terminate PID $pidValue. $_"
        }
    }
}

if ($killedPids.Count -eq 0) {
    Write-Output "No processes were terminated."
} else {
    $pidList = ($killedPids.Keys | Sort-Object) -join ", "
    Write-Output "Done. Terminated PID(s): $pidList"
}
