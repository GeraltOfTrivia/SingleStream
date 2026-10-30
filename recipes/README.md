# Measured settings and command templates

These pages document the configurations evaluated in the v0.32 campaigns. Match the model preparation, runtime, patch stack, and host containment described in [local setup and reproduction](../docs/install.md); see [validation](../docs/validation.md) for reproduction status.

| Platform | Configuration notes | Results |
|---|---|---|
| [24 GB Blackwell / dense Qwen](qwen-v032-blackwell24.md) | Initial MTP-4 command template; later native-NVFP4 MTP and YaRN settings | [Dense report](../docs/campaigns/24gb-v032.md) |
| [24 GB Blackwell / Qwen and Gemma MoE](moe-v032-blackwell24.md) | Native/extended settings, model-specific invocation requirements, reasoning-gate exceptions | [MoE report](../docs/campaigns/moe-v032.md) |
| [16 GB Ada / Qwen and Gemma](ada16-v032.md) | Measured Qwen MTP-3 and Gemma target/assistant settings | [Ada report](../docs/campaigns/ada16-v032.md) |

## 48 GB Blackwell

Selected configurations and their qualification/diagnostic boundaries are documented in the [48 GB dossiers](../docs/campaigns/48gb-v032.md). Settings are specific to each evaluated run and should not be mixed across configurations.
