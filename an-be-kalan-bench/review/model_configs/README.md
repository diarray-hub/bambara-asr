# Released checkpoint configuration snapshots

These YAML files were extracted with `ASRModel.from_pretrained(model_name, return_config=True)` using Python 3.12 and NeMo 3.0.0 on 2026-09-29. They are read-only snapshots of the published NeMo checkpoint configs, not training run logs.

| Model | Hugging Face revision | Snapshot |
| --- | --- | --- |
| `RobotsMali/soloni-be-kalan-v0` | `a6dc1434a29f1e00d22e0eb6f3c56ee6b8746890` | `soloni-be-kalan-v0.yaml` |
| `RobotsMali/anbekalanNet` | `9ed6cbe4cd9ba2c8a6657f9b719c70834a01a662` | `anbekalanNet.yaml` |

Soloni's released Exp3 config has zero spectrogram masks but nonzero encoder dropout and batch normalization. QuartzNet's released Exp3 config has zero rectangular masks and all Jasper block dropout values set to zero. The model configs alone do not prove the exact command-line overrides used in Exp2 or Exp4.
