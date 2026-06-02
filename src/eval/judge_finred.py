import asyncio
import os
import argparse
import json
import logging
import pandas as pd
import re
import uuid
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
import instructor
from openai import OpenAI

from src.eval.template.rubric_financial import Financial_Prompter_PerRubric, OURS_SYSTEM_PROMPT

class OutputPromptFormat(BaseModel):
    rubric_1: str
    rubric_2: str
    rubric_3: str
    rubric_4: str
    rubric_5: str
    final_judgment: str

# Logging and error file path setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ERROR_LOG_PATH = "judge_errors"
os.makedirs(ERROR_LOG_PATH, exist_ok=True)

# --- 2. CustomJudge updated to use instructor ---
class CustomJudge:
    def __init__(self, model: str):
        # Patch OpenAI client with instructor
        self.client = instructor.patch(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
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

# --- 3. process_prompt updated to use a Pydantic model ---
async def process_prompt(idx: int, user_prompt: str, judge_model: CustomJudge) -> dict or None:
    try:
        # Pass Pydantic model as response_model to get a parsed object
        result_obj = await judge_model.judge(OURS_SYSTEM_PROMPT, user_prompt, response_model=OutputPromptFormat)
        return result_obj
    except (ValidationError, json.JSONDecodeError) as e:
        logging.error(f"[{idx}] Pydantic validation/JSON error: {e}")
        err_file = f"{ERROR_LOG_PATH}/error_{idx}_{uuid.uuid4().hex[:6]}.json"
        with open(err_file, 'w', encoding='utf-8') as f:
            json.dump({'idx': idx, 'attack_prompt': user_prompt, 'error': str(e)}, f, ensure_ascii=False, indent=2)
        return None
    except Exception as e:
        logging.error(f"[{idx}] An unexpected error occurred: {e}")
        return None

# --- 4. main_async updated to save in dual formats ---
async def main_async(seeds_file: str, output_csv_name: str, output_dir: str,
                     attack_column: str = "attack_prompt", response_column: str = 'response'):
    if not os.path.exists(seeds_file):
        logging.error(f"Input not found: {seeds_file}")
        return
    df = pd.read_csv(seeds_file, keep_default_na=False)
        
    original_cat = df.get('category_prefix', []).tolist()
    
    print(len(df[attack_column].tolist()), len(df[response_column].tolist()), len(original_cat))
    
    prompts = Financial_Prompter_PerRubric().generate_promptchunks(
        df[attack_column].tolist(), df[response_column].tolist(), original_cat
    )

    judge_model = CustomJudge(model='gpt-4.1-2025-04-14')
    tasks = [process_prompt(i, p, judge_model) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    # Prepare results for CSV and JSON
    csv_rows = []
    json_outputs = []

    for i, res_obj in enumerate(results):
        original_data = df.iloc[i].to_dict()
        model_output_dict = res_obj.model_dump() if res_obj else None

        # Prepare JSON output
        json_outputs.append({
            "original_prompt": original_data.get(attack_column),
            "original_response": original_data.get(response_column),
            "model_output": model_output_dict
        })
        
        # Prepare CSV output (flattened)
        if res_obj:
            flat_row = {}
            # New output format: rubric_1~5 are "Safe" | "Unsafe"
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
    
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"{os.path.basename(output_csv_name)}.csv")
    out_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"Saved CSV results to {output_csv_path}")

    # Save JSON file
    json_output_path = os.path.splitext(output_csv_path)[0] + ".json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_outputs, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved JSON results to {json_output_path}")

    # Compute ASR
    out_df['judge'] = out_df['final_judgment'].apply(lambda x: 'Yes' if x == 'Unsafe' else 'No')
    overall = (out_df['judge'] == 'Yes').mean()
                         
    cat_asr_lv2 = {cat: grp['judge'].eq('Yes').mean() for cat, grp in out_df.groupby('category_prefix')}
    lv2_asr_path = os.path.join(output_dir, f'{output_csv_name}_asr_lv2_oursrubric.json')
    with open(lv2_asr_path, 'w', encoding='utf-8') as f:
        json.dump(cat_asr_lv2, f, indent=4, ensure_ascii=False)
    logging.info(f"Level 2 Category ASR saved to {lv2_asr_path}")
    
    # Compute and save Level 1 ASR
    out_df['category_prefix_lv1'] = out_df['category_prefix'].str.split('_').str[0]
    cat_asr_lv1 = {cat: grp['judge'].eq('Yes').mean() for cat, grp in out_df.groupby('category_prefix_lv1')}
    lv1_asr_path = os.path.join(output_dir, f'{output_csv_name}_asr_lv1_oursrubric.json')
    with open(lv1_asr_path, 'w', encoding='utf-8') as f:
        json.dump(cat_asr_lv1, f, indent=4, ensure_ascii=False)
    logging.info(f"Level 1 Category ASR saved to {lv1_asr_path}")
    
    logging.info(f"Overall ASR: {overall:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Async judge for rubric prompts with Pydantic and Instructor.')
    parser.add_argument('-i', '--input_csv',   required=True, help='Path to input CSV')
    parser.add_argument('-o', '--output_csv_name',  required=True, help='Name for output CSV (JSON will be saved with the same base name)')
    parser.add_argument('-d', '--outdir',  required=True, help='Directory for CSV and ASR JSON')
    parser.add_argument('-a', '--attack',  default='attack_prompt', help='Column name for attack prompts')
    parser.add_argument('-r', '--response',default='response', help='Column name for responses')

    args = parser.parse_args()
    
    # 
    
    asyncio.run(
        main_async(
            seeds_file=args.input_csv,
            output_csv_name=args.output_csv_name,
            output_dir=args.outdir,
            attack_column=args.attack,
            response_column=args.response
        )
    )

if __name__ == '__main__':
    main()
    
    # python judge_finred.py -i /path/to/FinRED/src/eval/dataset/qwen_2.5_7b_all_with_responses.csv -o qwen2.5_7b_responses -d ./infer_result/
