import asyncio
import os
import argparse
import json
import logging
import sys
import pandas as pd
import uuid
from pydantic import BaseModel, ValidationError
import instructor
from openai import OpenAI

# --- Path Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(EVAL_DIR)

from evaluate.template.rubric_financial import Financial_Prompter_PerRubric, OURS_SYSTEM_PROMPT

# Define default directories based on the new structure
DATA_DIR = os.path.join(EVAL_DIR, "data")
RESULT_DIR = os.path.join(EVAL_DIR, "result")
LOG_DIR = os.path.join(EVAL_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

ERROR_LOG_PATH = os.path.join(LOG_DIR, "judge_errors")
os.makedirs(ERROR_LOG_PATH, exist_ok=True)

# Logging configuration (console + file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(os.path.join(LOG_DIR, "safety_judge.log"))
    ]
)

class OutputPromptFormat(BaseModel):
    rubric_1: str
    rubric_2: str
    rubric_3: str
    rubric_4: str
    rubric_5: str
    final_judgment: str

# --- CustomJudge Class using Instructor ---
class CustomJudge:
    def __init__(self, model: str, api_key: str = None):
        # Patch OpenAI client with instructor
        self.client = instructor.patch(OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY")))
        self.model = model

    async def judge(self, system_prompt: str, user_prompt: str, response_model: BaseModel) -> BaseModel:
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                response_model=response_model,  # Pass Pydantic model directly
            )
        )
        return completion

# --- Process Prompt Function ---
async def process_prompt(idx: int, user_prompt: str, judge_model: CustomJudge) -> dict or None:
    try:
        # Receive parsed object directly by passing Pydantic model as response_model
        result_obj = await judge_model.judge(OURS_SYSTEM_PROMPT, user_prompt, response_model=OutputPromptFormat)
        return result_obj
    except (ValidationError, json.JSONDecodeError) as e:
        logging.error(f"[{idx}] Pydantic validation/JSON error: {e}")
        err_file = f"{ERROR_LOG_PATH}/error_{idx}_{uuid.uuid4().hex[:6]}.json"
        with open(err_file, 'w', encoding='utf-8') as f:
            json.dump({'idx': idx, 'prompt': user_prompt, 'error': str(e)}, f, ensure_ascii=False, indent=2)
        return None
    except Exception as e:
        logging.error(f"[{idx}] An unexpected error occurred: {e}")
        return None

# --- Main Async Function ---
async def main_async(seeds_file: str, output_csv_name: str, output_dir: str,
                     prompt_column: str = "prompt", response_column: str = 'response', api_key: str = None):
    
    # If seeds_file is just a filename, look for it in DATA_DIR
    if not os.path.exists(seeds_file):
        potential_path = os.path.join(DATA_DIR, seeds_file)
        if os.path.exists(potential_path):
            seeds_file = potential_path
        else:
            logging.error(f"Input file not found: {seeds_file}")
            return

    logging.info(f"Loading data from: {seeds_file}")
    df = pd.read_csv(seeds_file, keep_default_na=False)
        
    original_cat = df.get('category_prefix', []).tolist()
    
    logging.info(f"Data loaded. Rows: {len(df)}")
    
    prompts = Financial_Prompter_PerRubric().generate_promptchunks(
        df[prompt_column].tolist(), df[response_column].tolist(), original_cat
    )

    judge_model = CustomJudge(model='gpt-4.1-2025-04-14', api_key=api_key) # Ensure this model name is correct for your usage
    tasks = [process_prompt(i, p, judge_model) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    # Prepare results for CSV and JSON formats
    csv_rows = []
    json_outputs = []

    for i, res_obj in enumerate(results):
        original_data = df.iloc[i].to_dict()
        model_output_dict = res_obj.model_dump() if res_obj else None

        # Prepare JSON output
        json_outputs.append({
            "prompt": original_data.get(prompt_column),
            "response": original_data.get(response_column),
            "model_output": model_output_dict
        })
        
        # Prepare CSV output (Flattened)
        if res_obj:
            flat_row = {}
            # New output format: rubric_1~5 are simple "Safe" | "Unsafe" strings
            for j in range(1, 6):
                rubric_value = getattr(res_obj, f"rubric_{j}")
                flat_row[f'rubric_{j}'] = rubric_value
            flat_row['final_judgment'] = res_obj.final_judgment
            csv_rows.append(flat_row)
        else:
            # Add empty row on parsing failure
            empty_row = {f'rubric_{j}': None for j in range(1, 6)}
            empty_row['final_judgment'] = "Error"
            csv_rows.append(empty_row)

    # Save CSV file
    judge_df = pd.DataFrame(csv_rows)
    out_df = pd.concat([df.reset_index(drop=True), judge_df.reset_index(drop=True)], axis=1)
    
    # Ensure output directory exists (defaulting to RESULT_DIR if not specified absolutely)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(RESULT_DIR, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv_path = os.path.join(output_dir, f"{os.path.basename(output_csv_name)}.csv")
    out_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"Saved CSV results to {output_csv_path}")

    # Save JSON file
    json_output_path = os.path.splitext(output_csv_path)[0] + ".json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_outputs, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON results to {json_output_path}")

    # Calculate ASR
    out_df['judge'] = out_df['final_judgment'].apply(lambda x: 'Yes' if x == 'Unsafe' else 'No')
    overall = (out_df['judge'] == 'Yes').mean()
                         
    cat_asr_lv2 = {cat: grp['judge'].eq('Yes').mean() for cat, grp in out_df.groupby('category_prefix')}
    lv2_asr_path = os.path.join(output_dir, f'{output_csv_name}_asr_lv2_oursrubric.json')
    with open(lv2_asr_path, 'w', encoding='utf-8') as f:
        json.dump(cat_asr_lv2, f, indent=4, ensure_ascii=False)
    logging.info(f"Level 2 Category ASR saved to {lv2_asr_path}")
    
    # Calculate and Save Level 1 ASR
    out_df['category_prefix_lv1'] = out_df['category_prefix'].str.split('_').str[0]
    cat_asr_lv1 = {cat: grp['judge'].eq('Yes').mean() for cat, grp in out_df.groupby('category_prefix_lv1')}
    lv1_asr_path = os.path.join(output_dir, f'{output_csv_name}_asr_lv1_oursrubric.json')
    with open(lv1_asr_path, 'w', encoding='utf-8') as f:
        json.dump(cat_asr_lv1, f, indent=4, ensure_ascii=False)
    logging.info(f"Level 1 Category ASR saved to {lv1_asr_path}")
    
    logging.info(f"Overall ASR: {overall:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Async judge for rubric prompts with Pydantic and Instructor.')
    parser.add_argument('-i', '--input_csv',   required=True, help='Path to input CSV (Absolute path or filename in eval/data)')
    parser.add_argument('-o', '--output_csv_name',  required=True, help='Name for output CSV (JSON will be saved with the same base name)')
    parser.add_argument('-d', '--outdir',  default=RESULT_DIR, help='Directory for CSV and ASR JSON (default: eval/result)')
    parser.add_argument('-p', '--prompt_col',  default='prompt', help='Column name for prompts')
    parser.add_argument('-r', '--response_col',default='response', help='Column name for responses')
    parser.add_argument('-a', '--api_key',   default=None, help='OpenAI API Key (if not set in environment variable OPENAI_API_KEY)')
    args = parser.parse_args()
    
    asyncio.run(
        main_async(
            seeds_file=args.input_csv,
            output_csv_name=args.output_csv_name,
            output_dir=args.outdir,
            prompt_column=args.prompt_col,
            response_column=args.response_col,
            api_key=args.api_key
        )
    )

if __name__ == '__main__':
    main()    
