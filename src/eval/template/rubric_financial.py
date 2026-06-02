from typing import List
from pathlib import Path
import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
with open(PROMPTS_DIR / "judge.yaml", "r", encoding="utf-8") as f:
    PROMPT_FACTORY = yaml.safe_load(f)
    

BASE_FORMAT = PROMPT_FACTORY['BASE_FORMAT']

############################ end of prompts ############################

cat_to_prompt = {
    'R1_1': PROMPT_FACTORY["R1_1"],
    'R1_2': PROMPT_FACTORY["R1_2"],
    'R1_3': PROMPT_FACTORY["R1_3"],
    'R1_4': PROMPT_FACTORY["R1_4"],
    'R1_5': PROMPT_FACTORY["R1_5"],
    'R1_6': PROMPT_FACTORY["R1_6"],

    'R2_1': PROMPT_FACTORY["R2_1"],
    'R2_2': PROMPT_FACTORY["R2_2"],
    'R2_3': PROMPT_FACTORY["R2_3"],
    'R2_4': PROMPT_FACTORY["R2_4"],
    'R2_5': PROMPT_FACTORY["R2_5"],

    'R3_1': PROMPT_FACTORY["R3_1"],
    'R3_2': PROMPT_FACTORY["R3_2"],
    'R3_3': PROMPT_FACTORY["R3_3"],

    'R4_1': PROMPT_FACTORY["R4_1"],
    'R4_2': PROMPT_FACTORY["R4_2"],
    'R4_3': PROMPT_FACTORY["R4_3"],
    'R4_4': PROMPT_FACTORY["R4_4"],
    'R4_5': PROMPT_FACTORY["R4_5"],

    'R5_1': PROMPT_FACTORY["R5_1"],
    'R5_2': PROMPT_FACTORY["R5_2"],
    'R5_3': PROMPT_FACTORY["R5_3"],
    'R5_4': PROMPT_FACTORY["R5_4"],
    'R5_5': PROMPT_FACTORY["R5_5"],
    'R5_6': PROMPT_FACTORY["R5_6"],
    'R5_7': PROMPT_FACTORY["R5_7"],
}

HARMBENCH_SYSTEM_PROMPT = PROMPT_FACTORY["HARMBENCH_SYSTEM_PROMPT"]
HARMBENCH_BASE = PROMPT_FACTORY["HARMBENCH_BASE"]
OURS_SYSTEM_PROMPT=PROMPT_FACTORY["OURS_SYSTEM_PROMPT"]

class Financial_Prompter_PerRubric(object):
    def __init__(self):
        self.policy = "lv2"
        self.cat_to_prompt = cat_to_prompt
        self.base_format = BASE_FORMAT
        self.harmbench_format = HARMBENCH_BASE
      
    def harmbench_generate_promptchunks(
        self,
        questions,
        answers,
        category
    ):
        prompts = []
        for q, a in zip(questions, answers):
            prompts.append(self.harmbench_format % (q.strip(), a.strip()))
        return prompts

    def generate_promptchunks(
        self,
        questions: List[str],
        answers: List[str],
        category: List[str]) -> List[str]:
        try:
            assert questions and answers, f"questions : {questions}\nanswers : {answers}"
            assert len(questions) == len(answers)
        except AssertionError as e:
            raise ValueError(
                "Questions and answers must be non-empty and of the same length"
            )
        cat_rubric = [self.cat_to_prompt[cat] for cat in category]
        print(len(cat_rubric), len(questions), len(answers))
        
        # we have to map cat_rubric, questions, answers in base_format
        mapped_prompts = [self.base_format % (cat_rubric[idx], q.strip(), a.strip())
                          for idx, (q, a) in enumerate(zip(questions, answers))]

        return mapped_prompts
      
  
