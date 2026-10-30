# Attribution and provenance

SingleStream is derived from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen). The serving launcher, patch series, and selected benchmark helpers originate upstream in `syv-ai/HyperQwen` (which in turn adapts or backports selected upstream vLLM PRs and commits identified in the patch headers).

Work conducted in this repository includes the hardware/model evaluation campaigns, three-tier single-stream verification methodology, root-cause analysis of performance and numerical failures, and the migration to vLLM 0.32.0—retiring upstreamed or unneeded v0.31 patches and porting the retained 37-patch series onto vLLM 0.32.0. The original Apache-2.0 licence is retained in [LICENSE](LICENSE).

The retained patch files preserve their topic/upstream preambles and author headers. Historical NVFP4 backports are outside the retained v0.32 distribution; functionality native to vLLM 0.32 should not be reapplied. Model and dataset licences remain separate from this repository's licence. Model weights are not redistributed here. The selected v0.32 evidence includes frozen evaluation inputs and sample records containing dataset excerpts; the repository licence does not replace those materials' source licences:

- **GSM8K** (`openai/gsm8k`, 200-question evaluation subset): MIT License.
- **WikiText-2** (`Salesforce/wikitext`, `wikitext-2-raw-v1` test split excerpts): Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) / GFDL.
- **FineWeb-2** (`HuggingFaceFW/fineweb-2`, `deu_Latn` and `ell_Grek` test split excerpts): Open Data Commons Attribution License (ODC-By) v1.0; underlying web crawl texts remain subject to Common Crawl terms of use.
- **vLLM Core Source Excerpts** (`vllm/v1/core/*.py` code perplexity snippets): Apache License 2.0.

Campaign reports identify the model/runtime configurations and experiment procedures. Repository checks and reproduction status are recorded in [validation](docs/validation.md).
