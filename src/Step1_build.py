import os
import json
import pandas as pd
import random
import glob
import instructor
from pydantic import BaseModel, create_model
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, List
import argparse
import yaml

# Prompt mapping (load relative to project root)
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
with open(PROMPTS_DIR / "step1.yaml", "r", encoding="utf-8") as f:
    PROMPT_FACTORY = yaml.safe_load(f)
    
PROMPT_MAP = {
    "R1": (PROMPT_FACTORY["R1"]["system_prompt"], PROMPT_FACTORY["R1"]["user_prompt"]),
    "R2": (PROMPT_FACTORY["R2"]["system_prompt"], PROMPT_FACTORY["R2"]["user_prompt"]),
    "R3": (PROMPT_FACTORY["R3"]["system_prompt"], PROMPT_FACTORY["R3"]["user_prompt"]),
    "R4": (PROMPT_FACTORY["R4"]["system_prompt"], PROMPT_FACTORY["R4"]["user_prompt"]),
    "R5": (PROMPT_FACTORY["R5"]["system_prompt"], PROMPT_FACTORY["R5"]["user_prompt"]),
}

class ScenarioGenerator:
    def __init__(
        self,
        project_root: str,
        output_path: str,
        category_name: str,
        api_key: str = None,
        model_name: str = "gpt-4.1-2025-04-14",
        lang: str = "ko",
    ):
        self.project_root = Path(project_root)
        self.output_path = Path(output_path)
        self.category_name = category_name
        self.category_group = category_name.split('_')[0] # R1, R2, ...
        self.api_key = api_key
        self.model_name = model_name
        self.lang = lang
        self.client = self._init_client(api_key)

        # --- Paths ---
        self.src_root = self.project_root / "src"
        
        # 1. Schema path (language-specific)
        if self.lang == 'ko':
            self.schema_path = self.src_root / "data" / "schemas" / "ko"
        else:
            self.schema_path = self.src_root / "data" / "schemas" / "en"
        
        # 2. Query path
        self.query_path = self.src_root / "data" / "queries"

        # 3. Context and product paths (R3 has a different layout)
        if self.category_group == 'R3':
            self.context_path = self.src_root / "data/contexts/retrieved_chunks/common_R3"
            self.product_path = self.src_root / "data/contexts/R3_products"
        else:
            self.context_path = self.src_root / "data/contexts/retrieved_chunks/per_taxonomy_chunks"
            self.product_path = None

    def _init_client(self, api_key):
        if not api_key:
            raise ValueError("API key is required.")
        return instructor.from_openai(OpenAI(api_key=api_key))

    def _create_dynamic_model(self, model_name: str, schema_dict: Dict[str, Any]) -> BaseModel:
        """Create a dynamic Pydantic model from a JSON schema dictionary."""
        fields = {key: (Any, ...) for key in schema_dict.keys()}
        return create_model(model_name, **fields)

    def _load_schema(self, category_prefix: str) -> tuple[dict, str, list]:
        """Load schema files."""
        schema_file = self.schema_path / f"{category_prefix}.json"
        if not schema_file.exists():
            print(f"[Error] Schema file not found: {schema_file}")
            return {}, "", ["", ""]
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        few_shot_good = data.get('few_shot_good_examples', '').rstrip()
        few_shot_schema = data.get('few_shot_schema_examples', '').rstrip()
        
        return data['Schema'], data['Description'], [few_shot_good, few_shot_schema]

    def _load_queries(self, category_prefix: str) -> list:
        """Load query list for R3."""
        query_file = self.query_path / f"{category_prefix}_queries.csv"
        if not query_file.exists():
            print(f"[Error] Query file not found: {query_file}")
            return []
        query_df = pd.read_csv(query_file)
        return self._select_query_column(query_df)

    def _select_query_column(self, query_df: pd.DataFrame) -> list:
        """Select the best query column without relying on language-specific headers."""
        if query_df.empty:
            return []

        columns = list(query_df.columns)

        # Prefer unified 'query' column if present.
        for name in columns:
            lower = str(name).lower()
            if lower == "query":
                return query_df[name].dropna().tolist()

        if self.lang == 'en':
            # Prefer explicit English column names if present.
            for name in columns:
                lower = str(name).lower()
                if lower in {"english_query", "english", "en_query"} or "english" in lower:
                    return query_df[name].dropna().tolist()
        else:
            # Prefer explicit Korean column names if present.
            for name in columns:
                lower = str(name).lower()
                if lower in {"korean_query", "korean", "ko_query"} or "korean" in lower:
                    return query_df[name].dropna().tolist()

        # Fallback: pick the column with the highest ratio of Hangul characters.
        def hangul_ratio(text: str) -> float:
            if not isinstance(text, str) or not text:
                return 0.0
            total = len(text)
            hangul = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
            return hangul / total if total else 0.0

        best_col = columns[0]
        best_score = None
        for name in columns:
            series = query_df[name].astype(str).head(50).tolist()
            if not series:
                continue
            score = sum(hangul_ratio(s) for s in series) / max(len(series), 1)
            if self.lang == 'en':
                if best_score is None or score < best_score:
                    best_score = score
                    best_col = name
            else:
                if best_score is None or score > best_score:
                    best_score = score
                    best_col = name

        return query_df[best_col].dropna().tolist()

    def _load_contexts(self, category_prefix: str):
        """Load context files (R3 returns dict, others return list)."""
        if self.category_group == 'R3':
            # R3: common context file (e.g., R3_contents.json or {category_prefix}_contents.json)
            context_file = self.context_path / f"{category_prefix}_contents.json"
            if not context_file.exists():
                # Fallback: try R3_contents.json
                context_file = self.context_path / "R3_contents.json"
        else:
            # Others: per-query chunk file
            context_file = self.context_path / f"{category_prefix}_chunks.json"

        if not context_file.exists():
            print(f"[Error] Context file not found: {context_file}")
            return [] if self.category_group != 'R3' else {}

        with open(context_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_financial_products(self) -> list:
        """Load R3 financial product summaries."""
        if not self.product_path or not self.product_path.exists():
            print(f"[Error] Product path does not exist: {self.product_path}")
            return []

        product_files = sorted(glob.glob(str(self.product_path / '*_sum.json')))
        summaries = []  
        for p_file in product_files:
            with open(p_file, 'r', encoding='utf-8') as f:
                product_data = json.load(f)
            summaries.append(product_data.get('summary', 'No summary available'))
        return summaries

    def generate(self, system_prompt, usr_prompt, formatting_fn):
        response = self.client.chat.completions.create(
            model=self.model_name, # Keep the latest available model name
            max_tokens=16384,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": usr_prompt}
            ],
            response_model=formatting_fn
        )
        return response.model_dump()

    def run(self, category_prefix: str):
        print(f"--- {category_prefix} scenario generation started ---")
        
        # 1. Prepare prompts and schema
        sys_tmpl, usr_tmpl = PROMPT_MAP.get(self.category_group, (None, None))
        if not sys_tmpl:
            print(f"[Error] Unsupported category group: {self.category_group}")
            return

        schema_template, schema_description, few_shot_examples = self._load_schema(category_prefix)
        if not schema_template: return

        # Create dynamic model
        DynamicScenarioFormat = self._create_dynamic_model(f"{category_prefix}Model", schema_template)
        
        # Build system prompt
        system_prompt = sys_tmpl.format(
            few_shot_good_examples=few_shot_examples[0],
            few_shot_schema_examples=few_shot_examples[1]
        )

        # Create output directory
        category_output_path = self.output_path / category_prefix
        category_output_path.mkdir(parents=True, exist_ok=True)

        # 2. Load data and run generation (R3 vs others)
        if self.category_group == 'R3':
            self._run_r3_logic(category_prefix, system_prompt, usr_tmpl, schema_description, schema_template, DynamicScenarioFormat, category_output_path)
        else:
            self._run_general_logic(category_prefix, system_prompt, usr_tmpl, schema_description, schema_template, DynamicScenarioFormat, category_output_path)

        print(f"--- {category_prefix} scenario generation completed ---\n")

    def _run_r3_logic(self, category_prefix, system_prompt, usr_tmpl, schema_desc, schema_json, model, output_path):
        """R3 logic: product summaries + query mapping + shared context."""
        queries = self._load_queries(category_prefix)
        contexts = self._load_contexts(category_prefix) # expected dict
        product_summaries = self._load_financial_products()
        
        if not product_summaries:
            print("[Warning] No financial product summaries to process.")
            return

        required_contexts = [c['text'] for c in contexts.get('required', [])] if isinstance(contexts, dict) else []

        for i, prod_sum in enumerate(tqdm(product_summaries, desc=f"[{category_prefix}] per-product generation")):
            # Query mapping (round-robin)
            current_query = queries[i % len(queries)] if queries else "Generate a mis-selling scenario for a financial product."
            
            # Context sampling
            combined_context = list(required_contexts)
            if isinstance(contexts, dict):
                if contexts.get('first_chunk'):
                    combined_context.append(random.choice(contexts['first_chunk'])['text'])
                if contexts.get('second_chunk'):
                    combined_context.append(random.choice(contexts['second_chunk'])['text'])
            
            context_text = "\n\n---\n".join(combined_context)

            # Build prompt
            final_prompt = usr_tmpl.format(
                product_summary=prod_sum,
                mapped_query=current_query,
                sampled_context=context_text,
                schema_description=schema_desc,
                schema_structure_json=json.dumps(schema_json, indent=4, ensure_ascii=False)
            )

            self._generate_and_save(i, category_prefix, system_prompt, final_prompt, model, output_path)

    def _run_general_logic(self, category_prefix, system_prompt, usr_tmpl, schema_desc, schema_json, model, output_path):
        """R1, R2, R4, R5 logic: chunk files per query."""
        query_context_data = self._load_contexts(category_prefix) # expected list
        
        if not query_context_data:
            print(f"[Warning] No context data for {category_prefix}.")
            return

        for i, item in enumerate(tqdm(query_context_data, desc=f"[{category_prefix}] per-query generation")):
            all_chunks = item.get('extracted_texts', [])
            if not all_chunks: continue

            # Context sampling (keep original logic)
            if len(all_chunks) < 2:
                sampled_chunks = all_chunks
            else:
                initial = all_chunks[:4]
                later = all_chunks[4:]
                mandatory = random.sample(initial, min(2, len(initial)))
                remaining = [c for c in initial if c not in mandatory] + later
                random_sel = random.sample(remaining, min(random.randint(1, 2), len(remaining))) if remaining else []
                sampled_chunks = mandatory + random_sel
            
            context_text = "\n\n---\n\n".join(sampled_chunks)

            # Build prompt
            final_prompt = usr_tmpl.format(
                korean_query=item.get('korean_query', ''),
                english_query=item.get('english_query', ''),
                sampled_context=context_text,
                schema_description=schema_desc,
                schema_structure_json=json.dumps(schema_json, indent=4, ensure_ascii=False)
            )

            self._generate_and_save(i, category_prefix, system_prompt, final_prompt, model, output_path)

    def _generate_and_save(self, index, prefix, sys_prompt, usr_prompt, model, output_path):
        output_file = output_path / f"{index:04d}_{prefix}_scenario.json"
        try:
            result = self.generate(sys_prompt, usr_prompt, model)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"\n[Error] Failed to generate item {index}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_project_root = Path(__file__).resolve().parents[1]
    default_output_path = default_project_root / "src" / "outputs" / "scenarios"
    parser.add_argument('--project_root', type=str, default=str(default_project_root), help='Project root path')
    parser.add_argument('--output_path', type=str, default=str(default_output_path), help='Output directory')
    parser.add_argument('--category', type=str, required=True, help='Target category (e.g., R1, R2_1)')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--lang', type=str, default='ko', choices=['ko', 'en'], help='Schema language (ko/en)')
    
    args = parser.parse_args()

    
    # Expand category group (e.g., R1 -> R1_1 ... R1_6)
    if args.category in ['R1', 'R2', 'R3', 'R4', 'R5']:
        if args.category == 'R1': targets = [f"R1_{k}" for k in range(1, 7)]
        elif args.category == 'R2': targets = [f"R2_{k}" for k in range(1, 6)]
        elif args.category == 'R3': targets = [f"R3_{k}" for k in range(1, 4)]
        elif args.category == 'R4': targets = [f"R4_{k}" for k in range(1, 6)]
        elif args.category == 'R5': targets = [f"R5_{k}" for k in range(1, 8)]
    else:
        targets = [args.category]

    generator = ScenarioGenerator(
        project_root=args.project_root,
        output_path=args.output_path,
        category_name=args.category, # Pass the group name
        api_key=args.api_key,
        lang=args.lang
    )

    for target in targets:
        generator.run(target)
        
    
    # Example:
    # python src/Step1_build.py --category R1 --api_key YOUR_API_KEY
