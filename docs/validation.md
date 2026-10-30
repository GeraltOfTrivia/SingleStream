# Checks and reproduction status

Automated repository checks ([`.github/workflows/checks.yml`](../.github/workflows/checks.yml)) validate the patch series inventory and [`PATCHES.md`](../PATCHES.md) table generation (`scripts/patches_md.py`), patch-export round trips (`scripts/test_export_patch.sh`), shell/Python/JSON syntax, whitespace, and local Markdown links. Selected headline values have also been verified against their referenced evidence records.

Clean-environment installation and full campaign reproduction on target hardware have not been re-run from scratch outside the original experiment environments.

See the [benchmark scope](evidence-policy.md), [campaign index](campaigns/README.md), and [local setup notes](install.md).
