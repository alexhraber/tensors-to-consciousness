# Diagnostics + Logging

Diagnostics are configurable from both environment variables and `.config/config.json`.

Default behavior:

- Logging level: `INFO`
- Debug tracing: `off`

## Config file defaults

```json
{
  "framework": "numpy",
  "venv": ".venv-numpy",
  "platform": "gpu",
  "diagnostics": {
    "log_level": "INFO",
    "debug": false
  }
}
```

Notes:

- The config is normalized with sensible defaults when written by tooling.
- Diagnostics keys can be set under `diagnostics.*`.

## Environment overrides

Supported environment variables:

- `LOG_LEVEL`: one of `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`
- `DEBUG`: `1/0`, `true/false`, `on/off`, `yes/no`

Examples:

```bash
python explorer.py run --framework numpy --transforms default
DEBUG=1 LOG_LEVEL=DEBUG python explorer.py run --framework numpy --transforms default
```

## Precedence

1. Environment variables (`LOG_LEVEL`, `DEBUG`)
2. Config JSON (`diagnostics.log_level`, `diagnostics.debug`)
3. Built-in defaults (`INFO`, `debug=false`)
