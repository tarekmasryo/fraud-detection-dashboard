param(
  [string]$ApiUrl = "http://127.0.0.1:8000",
  [ValidateSet("rf","xgb")]
  [string]$Model = "rf",
  [Nullable[double]]$Threshold = $null,
  [switch]$UseQuery,
  [string]$JsonFile = "",
  [string]$ApiKey = "",
  [string]$BearerToken = ""
)

$ErrorActionPreference = "Stop"

function New-ZeroRecord {
  $feats = @("Time") + (1..28 | ForEach-Object { "V$_" }) + @("Amount")
  $rec = @{}
  foreach ($f in $feats) { $rec[$f] = 0.0 }
  return $rec
}

$record = $null
if ($JsonFile -ne "") {
  if (!(Test-Path $JsonFile)) { throw "JSON file not found: $JsonFile" }
  $record = (Get-Content -Raw $JsonFile | ConvertFrom-Json)
} else {
  $record = New-ZeroRecord
}

$payload = @{ record = $record }
if (-not $UseQuery) {
  $payload["model"] = $Model
}
if ($Threshold -ne $null) {
  $payload["threshold"] = [double]$Threshold
}
$body = $payload | ConvertTo-Json -Depth 10 -Compress

$uri = "$ApiUrl/v1/predictions"
if ($UseQuery) {
  $qs = "model=$Model"
  if ($Threshold -ne $null) { $qs = "$qs&threshold=$Threshold" }
  $uri = "$uri`?$qs"
}
Write-Host "POST $uri" -ForegroundColor Cyan
Write-Host $body

$headers = @{}
if ($ApiKey -ne "") { $headers["X-API-Key"] = $ApiKey }
if ($BearerToken -ne "") {
  $token = $BearerToken -replace "^Bearer\s+", ""
  $headers["Authorization"] = "Bearer $token"
}

$resp = Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Headers $headers -Body $body
$resp | ConvertTo-Json -Depth 10
