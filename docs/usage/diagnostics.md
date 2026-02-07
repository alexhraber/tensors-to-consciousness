# Diagnostics and Logging

Diagnostics can be controlled by config and environment variables.

Default diagnostics profile:

- `log_level=INFO`
- `debug=false`

## Config Defaults

```json
{
  "framework": "<platform default>",
  "venv": ".venv-<framework>",
  "platform": "gpu",
  "diagnostics": {
    "log_level": "INFO",
    "debug": false
  }
}
```

Framework default by platform:

- macOS: `mlx`
- other platforms: `jax`

## Environment Overrides

- `LOG_LEVEL`: `CRITICAL|ERROR|WARNING|INFO|DEBUG`
- `DEBUG`: `1/0`, `true/false`, `on/off`, `yes/no`

Examples:

```bash
explorer run --framework jax --transforms default
DEBUG=1 LOG_LEVEL=DEBUG explorer run --framework jax --transforms default
```

## Precedence

1. environment variables
2. `.config/config.json` diagnostics keys
3. runtime defaults
