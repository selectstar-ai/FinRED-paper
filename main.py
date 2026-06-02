import argparse
import sys
from pathlib import Path

# Import modules from src
from src.Step1_build import ScenarioGenerator
from src.Step2_build import RedTeamingPromptGenerator

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "src"))

def get_project_paths():
    """
    Set project root and output paths based on the location of main.py.
    """
    from pathlib import Path
    
    # The directory containing main.py is the project root (FinRED/)
    project_root = Path(__file__).resolve().parent
    
    return {
        "project_root": project_root,
        "output_scenarios": project_root / "src" / "outputs" / "scenarios",
        "output_prompts": project_root / "src" / "outputs" / "prompts"
    }

def get_target_categories(category_group):
    """Return the list of subcategories for a category group (R1, R2, ...)."""
    if category_group == 'R1': return [f"R1_{k}" for k in range(1, 7)]
    elif category_group == 'R2': return [f"R2_{k}" for k in range(1, 6)]
    elif category_group == 'R3': return [f"R3_{k}" for k in range(1, 4)]
    elif category_group == 'R4': return [f"R4_{k}" for k in range(1, 6)]
    elif category_group == 'R5': return [f"R5_{k}" for k in range(1, 8)]
    else: return [category_group] # Single category input (e.g., R1_1)

def main():
    parser = argparse.ArgumentParser(description="FinRED unified pipeline runner")
    
    # Execution mode
    parser.add_argument(
        '--step',
        type=str,
        choices=['1', '2', 'all'],
        required=True,
        help='Step to run (1: scenario generation, 2: prompt generation, all: sequential)'
    )
    
    # Shared settings
    parser.add_argument('--category', type=str, required=True, help='Target category group (e.g., R1, R2, R5)')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key for Step 1')
    parser.add_argument('--gemini_api_key', type=str, help='Gemini API key for Step 2')
    
    # Step 2 settings
    parser.add_argument('--lang', type=str, default='en', choices=['ko', 'en'], help='Prompt language for Step 2 (default: en)')
    parser.add_argument('--num_prompts', type=int, default=3, help='Number of prompts to generate in Step 2 (K)')

    # Model settings
    parser.add_argument('--step1_model', type=str, default='gpt-4.1-2025-04-14', help='Model for Step 1 (OpenAI)')
    parser.add_argument('--step2_model', type=str, default='models/gemini-2.5-pro', help='Model for Step 2 (Gemini)')
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='[Deprecated] Alias for --step1_model. Use --step1_model and --step2_model instead.'
    )

    args = parser.parse_args()

    # Paths and targets
    paths = get_project_paths()
    print(paths)
    target_categories = get_target_categories(args.category)
    
    print(f"[*] Project Root: {paths['project_root']}")
    print(f"[*] Targets: {target_categories}")

    # --- Step 1: Scenario generation ---
    if args.step in ['1', 'all']:
        if not args.openai_api_key:
            raise ValueError("--openai_api_key is required to run Step 1.")
            
        print(f"\n[Step 1] Starting scenario generation (Category: {args.category})")
        
        for cat in target_categories:
            # Step 1 expects project_root and output_path on initialization.
            step1_model = args.model_name or args.step1_model
            step1_runner = ScenarioGenerator(
                project_root=str(paths['project_root']),
                output_path=str(paths['output_scenarios']),
                category_name=cat,
                api_key=args.openai_api_key,
                model_name=step1_model,
                lang=args.lang
            )
            step1_runner.run(cat)

    # --- Step 2: Seed prompt generation ---
    if args.step in ['2', 'all']:
        if not args.gemini_api_key:
            raise ValueError("--gemini_api_key is required to run Step 2.")
            
        print(f"\n[Step 2] Starting seed prompt generation (Category: {args.category}, Lang: {args.lang})")
        
        for cat in target_categories:
            # Step 2 instance
            step2_runner = RedTeamingPromptGenerator(
                project_root=str(paths['project_root']),
                output_path=str(paths['output_prompts']),
                category_name=cat,
                num_prompts=args.num_prompts,
                api_key=args.gemini_api_key,
                lang=args.lang,
                model_name=args.step2_model
            )
            step2_runner.run(cat)

if __name__ == "__main__":
    main()
    # Example usage:
    # python main.py --step all --category R1 --openai_api_key "sk-..." --gemini_api_key "AIza..."
    # python main.py --step 2 --category R5 --lang en --num_prompts 5 --gemini_api_key "AIza..."
