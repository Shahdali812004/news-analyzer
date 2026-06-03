import json
from os.path import join
from urllib import response
from openai import OpenAI
from typing import List
#CUSTOM MODULES
from app.api.inference import json_parser
class SFTFormatter:

    def __init__(self,sft_path: str,api_key: str,raw_data:List[dict]
                 ,model_id: str = "gpt-4o-mini"):

        self.sft_path = sft_path
        self.model_id = model_id
        self.raw_data = raw_data
        self.client = OpenAI(api_key=api_key )

        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.price_per_1m_input_tokens = 0.150
        self.price_per_1m_output_tokens = 0.600

    ##################################################
    # Generic Prompt Builder
    ##################################################

    def build_messages(self,system_prompt: str,story: str,output_schema
                       ,additional_context: dict = None):

        user_sections = [
            "## Story:",
            story.strip(),
            "",

            "## Pydantic Details:",
            json.dumps(
                output_schema.model_json_schema(),
                ensure_ascii=False
            ),
            ""
        ]

        if additional_context:

            for key, value in additional_context.items():

                user_sections.extend([
                    f"## {key}:",
                    str(value),
                    ""
                ])

        user_sections.extend([
            "## Output JSON:",
            "```json"
        ])

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "\n".join(user_sections)
            }
        ]

        return messages



    def BuildSFTData(self,system_prompt: str,output_schema,
                       task:str,additional_context: dict = None):
        id = 0
        for new in self.raw_data:

            messages = self.build_messages(system_prompt,new['content'],output_schema,additional_context)

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )

            if response.choices[0].finish_reason != "stop":
                self.prompt_tokens += response.usage.prompt_tokens
                continue

            llm_response = response.choices[0].message.content
            llm_resp_dict = json_parser(llm_response)
            if not llm_resp_dict:
                continue
            with open(self.sft_path, "a", encoding="utf8") as dest:
                dest.write(json.dumps({
                    "id": id,
                    "story": new['content'].strip(),
                    "task": str(task).strip(),
                    "output_scheme": json.dumps( output_schema.model_json_schema(), ensure_ascii=False ),
                    "response": llm_resp_dict,
                }, ensure_ascii=False, default=str)  + "\n" )
                id+=1
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            if(id % 3) == 0:
                cost_input = (self.prompt_tokens / 1_000_000) * self.price_per_1m_input_tokens
                cost_output = (self.completion_tokens / 1_000_000) * self.price_per_1m_output_tokens
                total_cost = cost_input + cost_output

                print(f"Iteration {id}: Total Cost = ${total_cost:.4f} ")

         