# Local PostgreSQL without Docker or admin rights (portable EDB binaries).
#   powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 init     # once: create cluster + databases, start
#   powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 start | stop | status
# Binaries: $env:BOXING_AI_PG_BIN, default %LOCALAPPDATA%\Programs\pgsql\bin.
# Cluster: data\pg (git-ignored). Listens on localhost:5432 only, user "postgres", trust auth (local dev only).
param([ValidateSet('init', 'start', 'stop', 'status')][string]$Action = 'status')

$ErrorActionPreference = 'Stop'
$PgBin = if ($env:BOXING_AI_PG_BIN) { $env:BOXING_AI_PG_BIN } else { Join-Path $env:LOCALAPPDATA 'Programs\pgsql\bin' }
$Root = Split-Path $PSScriptRoot -Parent
$Data = Join-Path $Root 'data\pg'
$Log = Join-Path $Data 'server.log'

function Pg([string]$exe) { Join-Path $PgBin "$exe.exe" }

switch ($Action) {
    'init' {
        if (Test-Path (Join-Path $Data 'PG_VERSION')) { throw "cluster already exists in $Data" }
        & (Pg 'initdb') -D $Data -U postgres -A trust -E UTF8 --no-locale
        if ($LASTEXITCODE -ne 0) { throw 'initdb failed' }
        Add-Content -Encoding ascii (Join-Path $Data 'postgresql.conf') "`nlisten_addresses = 'localhost'`nport = 5432"
        & (Pg 'pg_ctl') -D $Data -l $Log -w start
        & (Pg 'createdb') -h localhost -U postgres boxing_ai
        Write-Host 'Ready: postgresql://postgres@localhost:5432/boxing_ai'
    }
    'start' { & (Pg 'pg_ctl') -D $Data -l $Log -w start }
    'stop' { & (Pg 'pg_ctl') -D $Data -w stop }
    'status' { & (Pg 'pg_ctl') -D $Data status }
}
