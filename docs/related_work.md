# Financial Benchmarks: Per-Benchmark Descriptions

Extended related-work notes referenced in the paper. Recent studies underscore the
necessity of financial domain-specific LLM evaluation due to the unique complexities and
risks inherent in the financial sector. A variety of benchmarks have been proposed to
assess financial reasoning and task performance:

- **Pixiu** — a comprehensive instruction-tuning dataset and evaluation suite for
  financial NLP tasks.
- **FinEval** — measures reasoning and question-answering performance across multi-domain
  financial contexts.
- **CFBenchmark** — focuses on financial assistant tasks.
- **FinanceBench** — evaluates financial question-answering tasks over corporate
  disclosure data.
- **FinBen** — benchmarks a wide range of financial tasks including QA, information
  extraction, and text generation across multiple languages.

Their core objective is to measure an LLM's capability as a financial *co-pilot*, not its
safety under **adversarial pressure** through red-teaming prompts. In contrast,
**FinRED** is the first benchmark that evaluates financial-domain LLMs from a *safety and
red-teaming* perspective, providing a uniquely expert-grounded framework for assessing
safety alignment in high-stakes finance applications.

## References

- Pixiu — Xie et al., 2023
- FinEval — Guo et al., 2025
- CFBenchmark — Lei et al., 2023
- FinanceBench — Islam et al., 2023
- FinBen — Xie et al., 2024
