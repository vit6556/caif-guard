$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

docker compose up -d gateway_raw gateway_nemo gateway_llama_guard gateway_caif

docker compose run --rm evaluator python scripts/run_all.py `
  --profile full `
  --modes raw nemo llama_guard caif
