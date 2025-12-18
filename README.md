# FinRED: An Expert-Guided Red-Teaming Benchmark for Financial LLM Safety

This repository contains the evaluation pipeline for **FinRED**, a benchmark designed to assess the safety of Large Language Models (LLMs) in the financial domain.

To ensure reliable and effective evaluation, we present an **expert-validated finance domain-specific rubric** for judging response safety, moving beyond simple disclaimer checks used in prior benchmarks. Our rubric demonstrates substantially higher alignment with human expert judgments compared to static, one-size-fits-all rubrics found in existing safety benchmarks.

---

## 1. Directory Structure
```text
FinRED-eval/
├── data/                   # Input datasets (e.g., sample_dataset.csv)
├── logs/                   # Execution logs and error reports
│   └── judge_errors/       # Detailed JSON logs for failed judgments
├── result/                 # Evaluation results (CSV, JSON, ASR scores)
├── run/                    # Execution scripts
│   └── judge_finred.py     # Main evaluation script
├── template/               # Prompt templates and rubrics
│   ├── __FinRED-eval/
├── data/                   # Input datasets (e.g., sample_dataset.csv)
├── logs/                   # Execution logs and error reports
│   └── judge_errors/       # Detailed JSON logs for failed judgments
├── result/                 # Evaluation results (CSV, JSON, ASR scores)
├── run/                    # Execution scripts
│   └── judge_finred.py     # Main evaluation script
├── template/               # Prompt templates and rubrics
│   └── rubric_financial.py # Financial safety rubric definition
└── utils/ # Utility functions
└── logger.py
```

---

## 2. Getting Started
Follow the steps below to set up the environment.

### Create Environment

**Using Conda:**
```bash
conda create -n finred python=3.10 -y
conda activate finred
```

**Or using venv:**
```bash
python3.10 -m venv finred_env
source finred_env/bin/activate
```

### Install Packages

```bash
pip install -r requirements.txt
```

---

## 3. Usage

You can run the evaluation using the `judge_finred.py` script located in the `run/` directory.

### Command Line Arguments

| Argument | Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| **Input File** | `-i`, `--input_csv` | Path to the input CSV file (relative to `data/` or absolute path). | **Required** |
| **Output Name** | `-o`, `--output_csv_name` | Base name for the output files (CSV & JSON). | **Required** |
| **Output Dir** | `-d`, `--outdir` | Directory to save results. | `./result` |
| **Prompt Col** | `-p`, `--prompt_col` | Column name for the input prompt in CSV. | `prompt` |
| **Response Col** | `-r`, `--response_col` | Column name for the model response in CSV. | `response` |
| **API Key** | `-a`, `--api_key` | OpenAI API Key (optional if set in env vars). | `None` |

### Example Command
To evaluate a dataset named `sample_dataset.csv` located in the `data/` folder:

```bash
python run/judge_finred.py \
    -i sample_dataset.csv \
    -o responses_output \
    -d ./result/qwen2.5_7b \
    -a sk-proj-YOUR_OPENAI_API_KEY...
```

---

## 4. Input Data Format
The input CSV file should contain at least the following columns:
- `category_prefix` : The taxonomy category of the prompt (e.g., `R1_1`, `R2_3`).
- `prompt`: The red-teaming prompt (attack).
- `response`: The LLM's response to be evaluated.

*(Note: Column names for prompt and response can be customized via arguments)*

---

## 5. Evaluation Logic

The evaluation process uses **GPT-4.1 (gpt-4.1-2025-04-14)** as a judge, applying our specialized financial safety rubric.

1. **Rubric-Based Assessment**: The judge evaluates the response across 5 specific rubrics tailored to financial risks.
2. **Final Judgment**: A final `Safe` or `Unsafe` verdict is rendered based on the rubric analysis.
3. **ASR Calculation**: Attack Success Rate (ASR) is calculated automatically for:
   - Overall performance
   - Level 1 Categories (e.g., R1, R2)
   - Level 2 Categories (e.g., R1_1, R2_1)

---

## 6. Output

The script generates the following files in the specified output directory:

1. `{output_name}.csv`: Contains original data plus individual rubric scores and final judgments.
2. `{output_name}.json`: Detailed JSON format of the results.
3. `{output_name}_asr_lv1_oursrubric.json`: ASR scores aggregated by Level 1 categories.
4. `{output_name}_asr_lv2_oursrubric.json`: ASR scores aggregated by Level 2 categories.

---

## Citation

If you use FinRED in your research, please cite our paper:

```bibtex
@article{finred2025,
  title={FinRED: An Expert-Guided Red-Teaming Benchmark for Financial LLM Safety},
  author={},
  journal={},
  year={2025}
}
```