# Benchmark scope

The dossiers describe measured configurations and retain failed or incomplete checks alongside passing results. Engine admission, physical KV pool capacity, and tested occupied context are separate measurements. Original PASS labels are read with their numerical gates and any stated exceptions.

Comparisons apply to the tested model preparations, runtime, sampling profiles, and hardware. Different quantizations, response budgets, or scoring protocols are identified in the relevant dossier. A few retrieval probes or a 200-question reasoning subset do not establish general model equivalence.

Selected 48 GB v0.32 findings and sanitized scalar evidence are included in the [48 GB dossiers](campaigns/48gb-v032.md) and [evidence extracts](campaigns/evidence/48gb/README.md), while raw run directories for the initial 24 GB dense controls are included under [`nvfp4/results/`](../nvfp4/results/README.md). Original raw records for the remaining campaigns are not redistributed here.
