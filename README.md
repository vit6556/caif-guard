# CAIF-Guard

CAIF-Guard (**Context-Aware Information Flow and Action Guard**) is a defensive runtime gateway for tool-using AI agents.

It protects the full agent execution path:

```text
user input → retrieved/tool context → proposed tool action → tool result → final output
```

CAIF-Guard focuses on action control and data-flow control. It checks what the agent is about to do, which arguments it wants to pass to tools, where data came from, whether the data is trusted, and where the data is going.

## What CAIF-Guard checks

| Boundary | What is checked |
|---|---|
| Input | Direct requests for hidden instructions, secrets, destructive actions, or exfiltration |
| RAG / context | Trust level of retrieved content; untrusted content is treated as data, not instructions |
| Tool call | Tool name, arguments, target resource, sink, action type, and current trace state |
| Tool result | Secret-like data, untrusted data, tainted data, and prompt-injection markers returned by tools |
| Output | Canary/API-key leakage and unsafe propagation of tainted content |

The default CAIF policy is deterministic and configured here:

```text
configs/policies/caif_policy.yaml
```

## Runtime modes

| Mode | Description |
|---|---|
| `raw` | Agent without a guard (baseline) |
| `nemo` | Agent behind NVIDIA NeMo Guardrails (default self-check input/output rails) |
| `llama_guard` | Agent behind Meta Llama Guard (default classification prompts, input + tool stages) |
| `caif` | Agent behind CAIF-Guard |

The `nemo` and `llama_guard` modes use their stock configurations without project-specific tuning, so the comparison reflects each method as it ships, not a hand-tailored variant.

## Configuration

Main configuration files:

| File | Purpose |
|---|---|
| `configs/runtime.yaml` | Shared runtime settings |
| `configs/models.yaml` | Target models and gateway URLs |
| `configs/policies/caif_policy.yaml` | CAIF-Guard policy |
| `configs/eval/scenarios.yaml` | Internal agent scenarios |
| `configs/promptfoo/static_template.yaml` | Static Promptfoo evaluation template |
| `configs/garak/benchmark.yaml` | Garak benchmark settings |

## Requirements

- Docker with Compose
- Ollama reachable from containers (default `http://host.docker.internal:11434`)
- ~12 GB free VRAM recommended (target model up to 7B + guard model `llama3.2:3b` may run in parallel)

Target models used in the current benchmark matrix (`configs/models.yaml`):

```sh
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull deepseek-coder:6.7b
ollama pull qwen3.5:2b
ollama pull falcon3:3b
ollama pull hermes3:3b
ollama pull openchat:7b
ollama pull granite3.3:2b
ollama pull yi:6b
```

`llama3.2:3b` is reused as the guard model for both `llama_guard` and `nemo` (see `configs/runtime.yaml`, `guard_models`).

If Ollama is not reachable at the default URL, update `ollama_base_url` in:

```text
configs/runtime.yaml
```

## Run the benchmark

Start the gateways:

```sh
docker compose build --no-cache
docker compose up -d gateway_raw gateway_nemo gateway_llama_guard gateway_caif
```

Run the full benchmark:

```sh
docker compose run --rm evaluator python scripts/run_all.py \
  --profile full \
  --modes raw nemo llama_guard caif
```

PowerShell helper:

```sh
pwsh scripts/run_full.ps1
```

## Evaluation methodology

The benchmark combines three independent sources of measurement, all driven by the same matrix of target models × runtime modes:

| Source | Configuration | What it measures |
|---|---|---|
| Internal agent scenarios | `configs/eval/scenarios.yaml` | Real tool-using agent behaviour (benign + attack), with per-stage guard decisions and per-tool action audit |
| Promptfoo static red-team | `configs/promptfoo/static_template.yaml` | Black-box HTTP checks against the gateway with literal `not-contains` assertions; identical for every mode |
| Garak prompt-injection scan | `configs/garak/benchmark.yaml` | Standard `promptinject.HijackHateHumans` + `promptinject.HijackKillHumans` probes |

All modes are evaluated on the same scenarios, the same Promptfoo assertions, and the same Garak probes. The Promptfoo assertion set contains no checks tied to internal rules of any specific guard.

## Reports

Reports are written to:

```text
reports/run_YYYYMMDD_HHMMSS_full/
```

Main outputs:

| File or directory | Purpose |
|---|---|
| `SUMMARY.md` | Human-readable summary split by evaluation source |
| `agent_matrix.csv` / `agent_matrix.jsonl` | Internal scenario results for each model and mode |
| `summary_by_model_mode.csv` | Internal scenario summary table |
| `promptfoo_static_summary.csv` | Promptfoo static summary split into attack and benign checks |
| `garak_summary.csv` | Garak scan summary |
| `evaluation_summary_by_tool.csv` | Combined table split by evaluation source |
| `promptfoo_static/` | Promptfoo configs and raw outputs |
| `garak/` | Raw Garak logs and reports |

No aggregate scoring formula is calculated. Promptfoo static results are reported separately for attack checks and benign utility checks.

## Run only CAIF-Guard

Start the CAIF gateway:

```sh
docker compose build --no-cache gateway_caif
docker compose up -d gateway_caif
```

Health and policy:

```sh
curl http://localhost:8004/healthz
curl http://localhost:8004/caif/policy
```

CAIF endpoints:

```text
POST /caif/check/input
POST /caif/check/tool-call
POST /caif/check/tool-result
POST /caif/check/output
GET  /caif/policy
```

A generic external agent can integrate CAIF-Guard as middleware:

```text
1. Call /caif/check/input before processing user input.
2. Call /caif/check/tool-call before each tool or MCP action.
3. Execute the tool only if CAIF returns allowed=true.
4. Call /caif/check/tool-result after tool execution.
5. Call /caif/check/output before returning the final answer.
```

### Example CAIF request and response

Request:

```sh
curl -s http://localhost:8004/caif/check/tool-call \
  -H 'Content-Type: application/json' \
  -d '{
    "trace_id": "example-trace-1",
    "tool_name": "read_file",
    "args": {"path": "secrets/api_key.txt"}
  }'
```

Example response:

```json
{
  "trace_id": "example-trace-1",
  "stage": "tool_call",
  "guard_mode": "caif",
  "allowed": false,
  "reason": "CAIF-Guard action policy blocked tool call: protected/private path access is not allowed",
  "risk_tags": ["caif_action_block", "protected_path_access"],
  "cia": ["confidentiality"],
  "replacement": null,
  "metadata": {
    "stage": "tool_call",
    "caif_rule": "action_policy",
    "tool": "read_file",
    "args": {"path": "[protected]"}
  }
}
```

## Configure CAIF-Guard

Edit:

```text
configs/policies/caif_policy.yaml
```

Useful policy sections:

| Policy section | Purpose |
|---|---|
| `secret_patterns` | Regexes for canaries, API keys, and secret-like content |
| `prompt_injection_patterns` | Markers for direct and indirect prompt injection |
| `protected_path_patterns` | Resources treated as protected/private |
| `allowed_write_prefixes` | Paths where write operations are allowed |

## Safety scope

This repository is intended for defensive evaluation of local AI-agent security controls. The included tools are sandboxed mock tools and do not send real emails, perform external HTTP exfiltration, or execute arbitrary host commands.
