from typing import List
import json
import random
from app.utils.prompts import SYSTEM_MESSAGE
class LamaFactoryFormatter:
    def __init__(self,SFT_data_path: str):
        self.SFT_data_path = SFT_data_path
   
    def BuildLamaFactoryData(self):
        llm_finetunning_data = []
        
        for line in open(self.SFT_data_path):
            if line.strip() == "":
                continue

            rec = json.loads(line.strip())

            llm_finetunning_data.append({
                "system": SYSTEM_MESSAGE,
                "instruction": "\n".join([
                    "# Story:",
                    rec["story"],

                    "# Task:",
                    rec["task"],

                    "# Output Scheme:",
                    rec["output_scheme"],
                    "",

                    "# Output JSON:",
                    "```json"

                ]),
                "input": "",
                "output": "\n".join([
                    "```json",
                    json.dumps(rec["response"], ensure_ascii=False, default=str),
                    "```"
                ]),
                "history": []
            })
        return random.Random(101).shuffle(llm_finetunning_data)